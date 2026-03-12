"""GRPO training entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from src.data.loaders import dataset_from_records, maybe_load_processed_dataset
from src.data.preprocess_rl import preprocess_rl_from_config
from src.rewards.combined import CombinedReward, RewardWeights
from src.rewards.rlaif_reward import RLAIFReward
from src.utils.gpu_monitor import reset_peak_memory, snapshot_gpu_state
from src.utils.io import (
    dataclass_from_dict,
    ensure_dir,
    load_stage_config,
    resolve_path,
    save_config_snapshot,
    save_git_state,
    save_jsonl,
)
from src.utils.logging import MetricsLogger, setup_logger
from src.utils.seed import seed_everything


@dataclass
class ModelConfig:
    policy_model_name: str
    fallback_model_name: str
    trust_remote_code: bool = True
    use_4bit: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    dataset_name: str
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str = "test"
    input_file: str = "data/processed/rl_train.jsonl"
    eval_file: str = "data/processed/rl_eval.jsonl"
    use_processed_if_available: bool = True
    cache_dir: str = "data/cache"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_proc: int = 4


@dataclass
class TrainingConfig:
    output_dir: str
    logging_dir: str
    policy_dir: str
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-5
    num_train_epochs: float = 1.0
    max_steps: int = 256
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 3
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_generations: int = 4
    group_size: int = 4
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None


@dataclass
class RewardConfig:
    reward_model_path: str
    correctness_weight: float = 0.5
    rlaif_weight: float = 0.3
    format_weight: float = 0.2
    required_sections: list[str] | None = None


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class GRPOStageConfig:
    stage_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    reward: RewardConfig
    lora: LoraConfig
    paths: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--config", required=True, help="Path to configs/grpo.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def prepare_dataset(raw_config: dict[str, Any], config: GRPOStageConfig):
    """Load or preprocess RL prompt data."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)
    eval_path = resolve_path(config.data.eval_file, project_root)

    train_dataset = maybe_load_processed_dataset(train_path) if config.data.use_processed_if_available else None
    eval_dataset = maybe_load_processed_dataset(eval_path) if config.data.use_processed_if_available else None

    if train_dataset is None or eval_dataset is None:
        train_rows, eval_rows = preprocess_rl_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(eval_path, eval_rows)
        train_dataset = dataset_from_records(train_rows)
        eval_dataset = dataset_from_records(eval_rows)

    return train_dataset, eval_dataset


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            parts.append(_completion_to_text(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def _prompt_to_text(prompt: Any) -> str:
    return _completion_to_text(prompt)


def main() -> None:
    args = parse_args()
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    config = dataclass_from_dict(GRPOStageConfig, raw_config)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    policy_dir = ensure_dir(resolve_path(config.training.policy_dir, project_root))
    logger = setup_logger("train_grpo", output_dir / "train.log")
    metrics_logger = MetricsLogger(output_dir / "metrics.jsonl")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)
    reset_peak_memory()

    from trl import GRPOConfig, GRPOTrainer

    from src.models.lora_utils import build_lora_config
    from src.models.model_utils import load_tokenizer, resolve_model_name

    policy_path = resolve_model_name(
        config.model.policy_model_name,
        fallback_model_name=config.model.fallback_model_name,
    )
    tokenizer = load_tokenizer(policy_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset, eval_dataset = prepare_dataset(raw_config, config)

    reward_model_path = resolve_path(config.reward.reward_model_path, project_root)
    rlaif = RLAIFReward(str(reward_model_path)) if reward_model_path.exists() else None
    combined_reward = CombinedReward(
        weights=RewardWeights(
            correctness=config.reward.correctness_weight,
            rlaif=config.reward.rlaif_weight,
            format=config.reward.format_weight,
        ),
        required_sections=config.reward.required_sections,
        rlaif_scorer=rlaif,
    )

    def reward_fn(prompts, completions, answer=None, answers=None, **kwargs):
        answers_list = answers or answer or kwargs.get("answer") or [""] * len(completions)
        if not isinstance(answers_list, list):
            answers_list = [answers_list] * len(completions)
        scores = []
        for prompt, completion, gold in zip(prompts, completions, answers_list):
            scores.append(combined_reward.score(_prompt_to_text(prompt), _completion_to_text(completion), gold))
        return scores

    peft_config = None
    if config.lora.enabled:
        peft_config = build_lora_config(
            r=config.lora.r,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
        )

    training_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        logging_dir=str(resolve_path(config.training.logging_dir, project_root)),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        num_generations=config.training.num_generations,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        report_to=["tensorboard"],
    )

    trainer = GRPOTrainer(
        model=policy_path,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO training. resume_from_checkpoint=%s", config.training.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(policy_dir)
    tokenizer.save_pretrained(policy_dir)
    trainer_state_path = output_dir / "checkpoints" / "trainer_state.json"
    if trainer_state_path.exists():
        (output_dir / "trainer_state.json").write_text(
            trainer_state_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    if trainer.state.log_history:
        save_jsonl(output_dir / "metrics.jsonl", trainer.state.log_history)
    metrics_logger.log({"event": "train_result", **train_result.metrics})
    metrics_logger.log({"event": "gpu_state", **snapshot_gpu_state()})
    logger.info("GRPO training finished. Policy saved to %s", policy_dir)


if __name__ == "__main__":
    main()
