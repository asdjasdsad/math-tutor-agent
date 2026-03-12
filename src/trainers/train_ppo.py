"""PPO training entrypoint backed by LLaMA-Factory."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.preprocess_rl import preprocess_rl_from_config
from src.models.model_utils import resolve_model_name
from src.trainers.llamafactory_runner import (
    LlamaFactoryConfig,
    detect_reward_model_type,
    infer_template,
    load_jsonl_rows,
    merge_recipe,
    prepare_dataset_dir,
    register_dataset,
    run_recipe,
    save_recipe,
    write_ppo_dataset,
)
from src.utils.io import (
    dataclass_from_dict,
    ensure_dir,
    load_stage_config,
    resolve_path,
    save_config_snapshot,
    save_git_state,
    save_jsonl,
)
from src.utils.logging import setup_logger
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
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class PPOStageConfig:
    stage_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    reward: RewardConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    llamafactory: LlamaFactoryConfig = field(default_factory=LlamaFactoryConfig)
    paths: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO training with LLaMA-Factory.")
    parser.add_argument("--config", required=True, help="Path to configs/ppo.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], PPOStageConfig]:
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    structured = dataclass_from_dict(PPOStageConfig, raw_config)
    return raw_config, structured


def prepare_datasets(raw_config: dict[str, Any], config: PPOStageConfig) -> tuple[Path, str, str]:
    """Ensure processed RL prompts exist and emit an Alpaca-style dataset."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)
    eval_path = resolve_path(config.data.eval_file, project_root)

    if not (config.data.use_processed_if_available and train_path.exists() and eval_path.exists()):
        train_rows, eval_rows = preprocess_rl_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(eval_path, eval_rows)

    dataset_dir = prepare_dataset_dir(config.llamafactory.dataset_dir, project_root)
    train_output = write_ppo_dataset(load_jsonl_rows(train_path), dataset_dir / "ppo_train.jsonl")
    eval_output = write_ppo_dataset(load_jsonl_rows(eval_path), dataset_dir / "ppo_eval.jsonl")

    train_name = f"{config.stage_name}_train"
    eval_name = f"{config.stage_name}_eval"
    definition = {
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    register_dataset(dataset_dir, train_name, {**definition, "file_name": train_output.name})
    register_dataset(dataset_dir, eval_name, {**definition, "file_name": eval_output.name})
    return dataset_dir, train_name, eval_name


def build_training_recipe(
    raw_config: dict[str, Any],
    config: PPOStageConfig,
    dataset_dir: Path,
    train_name: str,
    eval_name: str,
) -> dict[str, Any]:
    """Translate project config into a LLaMA-Factory PPO recipe."""

    project_root = raw_config["paths"]["project_root"]
    policy_path = resolve_model_name(
        config.model.policy_model_name,
        fallback_model_name=config.model.fallback_model_name,
    )
    if config.training.resume_from_checkpoint:
        policy_path = config.training.resume_from_checkpoint

    reward_candidate = resolve_path(config.reward.reward_model_path, project_root)
    reward_model = str(reward_candidate) if reward_candidate.exists() else config.reward.reward_model_path
    reward_type = detect_reward_model_type(reward_model) if Path(reward_model).exists() else "full"
    policy_dir = resolve_path(config.training.policy_dir, project_root)
    total_updates = math.ceil(config.training.total_episodes / max(config.training.batch_size, 1))

    recipe = {
        "model_name_or_path": policy_path,
        "trust_remote_code": config.model.trust_remote_code,
        "reward_model": reward_model,
        "reward_model_type": reward_type,
        "stage": "ppo",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora" if config.lora.enabled else "full",
        "template": infer_template(policy_path, config.llamafactory.template),
        "dataset_dir": str(dataset_dir),
        "dataset": train_name,
        "eval_dataset": eval_name,
        "preprocessing_num_workers": config.data.num_proc,
        "cutoff_len": config.data.max_prompt_length + config.training.generation_max_new_tokens,
        "learning_rate": config.training.learning_rate,
        "max_steps": total_updates,
        "per_device_train_batch_size": config.training.mini_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "ppo_epochs": config.training.ppo_epochs,
        "ppo_buffer_size": config.training.batch_size,
        "target_kl": config.training.target_kl,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "output_dir": str(policy_dir),
        "logging_dir": str(resolve_path(config.training.logging_dir, project_root)),
        "plot_loss": True,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": config.training.resume_from_checkpoint,
        "seed": config.seed,
        "bf16": config.training.bf16,
        "temperature": config.training.temperature,
        "top_p": config.training.top_p,
        "max_new_tokens": config.training.generation_max_new_tokens,
        "lora_rank": config.lora.r if config.lora.enabled else None,
        "lora_alpha": config.lora.alpha if config.lora.enabled else None,
        "lora_dropout": config.lora.dropout if config.lora.enabled else None,
        "lora_target": ",".join(config.lora.target_modules) if config.lora.target_modules else None,
    }
    return merge_recipe(recipe, config.llamafactory.train_args)


def main() -> None:
    args = parse_args()
    raw_config, config = load_config(args)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    policy_dir = ensure_dir(resolve_path(config.training.policy_dir, project_root))
    logger = setup_logger("train_ppo", output_dir / "train.log")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)

    logger.warning(
        "PPO training now uses LLaMA-Factory reward model scoring. "
        "Custom correctness/format/RLAIF weights are kept only for offline evaluation."
    )

    dataset_dir, train_name, eval_name = prepare_datasets(raw_config, config)
    recipe = build_training_recipe(raw_config, config, dataset_dir, train_name, eval_name)
    recipe_path = save_recipe(output_dir / "llamafactory_train.yaml", recipe)

    logger.info("Launching LLaMA-Factory PPO. output_dir=%s policy_dir=%s", output_dir, policy_dir)
    run_recipe(config.llamafactory.cli_path, "train", recipe_path, workdir=project_root)
    logger.info("PPO flow completed.")


if __name__ == "__main__":
    main()
