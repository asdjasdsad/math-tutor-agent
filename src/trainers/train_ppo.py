"""PPO baseline training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from src.data.loaders import dataset_from_records, maybe_load_processed_dataset
from src.data.preprocess_rl import preprocess_rl_from_config
from src.rewards.combined import CombinedReward, RewardWeights
from src.rewards.rlaif_reward import RLAIFReward
from src.utils.gpu_monitor import get_peak_memory_gb, reset_peak_memory, snapshot_gpu_state
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
from src.utils.profiling import Stopwatch, safe_rate
from src.utils.seed import seed_everything


@dataclass
class ModelConfig:
    policy_model_name: str
    fallback_model_name: str
    trust_remote_code: bool = True
    use_4bit: bool = False
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
    max_prompt_length: int = 512
    max_answer_length: int = 128


@dataclass
class TrainingConfig:
    output_dir: str
    logging_dir: str
    policy_dir: str
    checkpoint_dir: str
    mini_batch_size: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 1
    learning_rate: float = 1.0e-5
    target_kl: float = 0.1
    total_episodes: int = 512
    logging_steps: int = 10
    save_steps: int = 50
    bf16: bool = True
    generation_max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    resume_from_checkpoint: str | None = None


@dataclass
class RewardConfig:
    reward_model_path: str
    correctness_weight: float = 0.5
    rlaif_weight: float = 0.3
    format_weight: float = 0.2
    required_sections: list[str] | None = None


@dataclass
class PPOStageConfig:
    stage_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    reward: RewardConfig
    paths: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO baseline training.")
    parser.add_argument("--config", required=True, help="Path to configs/ppo.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def prepare_dataset(raw_config: dict[str, Any], config: PPOStageConfig, tokenizer: Any):
    """Load or preprocess RL prompt data and tokenize prompts."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)

    train_dataset = maybe_load_processed_dataset(train_path) if config.data.use_processed_if_available else None
    if train_dataset is None:
        train_rows, eval_rows = preprocess_rl_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(resolve_path(config.data.eval_file, project_root), eval_rows)
        train_dataset = dataset_from_records(train_rows)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=config.data.max_prompt_length,
        )
        return {
            "query": example["prompt"],
            "answer": example.get("answer", ""),
            "input_ids": encoded["input_ids"],
        }

    return train_dataset.map(_tokenize, num_proc=config.data.num_proc)


def _collator(features: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [feature[key] for feature in features] for key in features[0]}


def _load_resume_step(output_dir: Path, resume_from_checkpoint: str | None) -> int:
    if resume_from_checkpoint:
        state_path = Path(resume_from_checkpoint) / "rl_state.json"
    else:
        state_path = output_dir / "rl_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return int(payload.get("step", 0))
    return 0


def _save_rl_state(path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"step": step}, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    config = dataclass_from_dict(PPOStageConfig, raw_config)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    policy_dir = ensure_dir(resolve_path(config.training.policy_dir, project_root))
    checkpoint_root = ensure_dir(resolve_path(config.training.checkpoint_dir, project_root))
    logger = setup_logger("train_ppo", output_dir / "train.log")
    metrics_logger = MetricsLogger(output_dir / "metrics.jsonl")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)
    reset_peak_memory()

    import torch
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

    from src.models.model_utils import (
        get_model_device,
        get_torch_dtype,
        load_tokenizer,
        resolve_model_name,
    )

    policy_path = resolve_model_name(
        config.model.policy_model_name,
        fallback_model_name=config.model.fallback_model_name,
    )
    if config.training.resume_from_checkpoint:
        policy_path = config.training.resume_from_checkpoint

    tokenizer = load_tokenizer(policy_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = prepare_dataset(raw_config, config, tokenizer)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=get_torch_dtype(config.model.compute_dtype),
    )
    ref_model = create_reference_model(model)

    ppo_config = PPOConfig(
        model_name=policy_path,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        mini_batch_size=config.training.mini_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        ppo_epochs=config.training.ppo_epochs,
        target_kl=config.training.target_kl,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=_collator,
    )

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

    start_step = _load_resume_step(output_dir, config.training.resume_from_checkpoint)
    total_updates = math.ceil(config.training.total_episodes / config.training.batch_size)
    dataloader = cycle(ppo_trainer.dataloader)
    generation_kwargs = {
        "max_new_tokens": config.training.generation_max_new_tokens,
        "temperature": config.training.temperature,
        "top_p": config.training.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    device = get_model_device(model)
    logger.info("Starting PPO training from step %s/%s", start_step, total_updates)

    for step in range(start_step, total_updates):
        batch = next(dataloader)
        query_tensors = [torch.tensor(ids, device=device) for ids in batch["input_ids"]]
        timer = Stopwatch()
        timer.start()
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        elapsed = timer.elapsed()

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        prompts = batch["query"]
        answers = batch["answer"]
        component_rows = combined_reward.score_batch(prompts, responses, answers)
        rewards = [torch.tensor(row["combined_reward"], device=device) for row in component_rows]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        try:
            ppo_trainer.log_stats(stats, batch, rewards)
        except Exception:
            pass

        generated_tokens = sum(len(tensor) for tensor in response_tensors)
        prompt_tokens = sum(len(tensor) for tensor in query_tensors)
        metric_row = {
            "step": step + 1,
            "mean_reward": sum(row["combined_reward"] for row in component_rows) / len(component_rows),
            "mean_correctness_reward": sum(row["correctness_reward"] for row in component_rows) / len(component_rows),
            "mean_rlaif_reward": sum(row["rlaif_reward"] for row in component_rows) / len(component_rows),
            "mean_format_reward": sum(row["format_reward"] for row in component_rows) / len(component_rows),
            "tokens_per_sec": safe_rate(prompt_tokens + generated_tokens, elapsed),
            "samples_per_sec": safe_rate(len(prompts), elapsed),
            "peak_memory_gb": get_peak_memory_gb(),
            "elapsed_sec": elapsed,
        }
        if isinstance(stats, dict):
            metric_row.update({key: float(value) for key, value in stats.items() if isinstance(value, (int, float))})
        metrics_logger.log(metric_row)

        if (step + 1) % config.training.logging_steps == 0:
            logger.info("PPO step %s/%s: mean_reward=%.4f", step + 1, total_updates, metric_row["mean_reward"])

        if (step + 1) % config.training.save_steps == 0:
            checkpoint_dir = ensure_dir(checkpoint_root / f"step-{step + 1}")
            ppo_trainer.model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            _save_rl_state(checkpoint_dir / "rl_state.json", step + 1)

    ppo_trainer.model.save_pretrained(policy_dir)
    tokenizer.save_pretrained(policy_dir)
    _save_rl_state(output_dir / "rl_state.json", total_updates)
    metrics_logger.log({"event": "gpu_state", **snapshot_gpu_state()})
    logger.info("PPO training finished. Policy saved to %s", policy_dir)


if __name__ == "__main__":
    main()
