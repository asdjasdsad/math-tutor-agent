"""Reward-model training entrypoint backed by LLaMA-Factory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.preprocess_reward import preprocess_reward_from_config
from src.trainers.llamafactory_runner import (
    LlamaFactoryConfig,
    infer_template,
    load_jsonl_rows,
    merge_recipe,
    prepare_dataset_dir,
    register_dataset,
    run_recipe,
    save_recipe,
    write_reward_dataset,
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
    base_model_name: str
    trust_remote_code: bool = True
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    dataset_name: str
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str = "train[:512]"
    input_file: str = "data/processed/reward_train.jsonl"
    eval_file: str = "data/processed/reward_eval.jsonl"
    use_processed_if_available: bool = True
    cache_dir: str = "data/cache"
    num_proc: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_length: int = 2048
    prompt_template: str = ""


@dataclass
class TrainingConfig:
    output_dir: str
    logging_dir: str
    model_dir: str
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class RewardStageConfig:
    stage_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoraConfig
    llamafactory: LlamaFactoryConfig = field(default_factory=LlamaFactoryConfig)
    paths: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reward model with LLaMA-Factory.")
    parser.add_argument("--config", required=True, help="Path to configs/reward.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def prepare_datasets(raw_config: dict[str, Any], config: RewardStageConfig) -> tuple[Path, str, str]:
    """Ensure processed reward data exists and emit pairwise JSONL."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)
    eval_path = resolve_path(config.data.eval_file, project_root)

    if not (config.data.use_processed_if_available and train_path.exists() and eval_path.exists()):
        train_rows, eval_rows = preprocess_reward_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(eval_path, eval_rows)

    dataset_dir = prepare_dataset_dir(config.llamafactory.dataset_dir, project_root)
    train_output = write_reward_dataset(load_jsonl_rows(train_path), dataset_dir / "reward_train.jsonl")
    eval_output = write_reward_dataset(load_jsonl_rows(eval_path), dataset_dir / "reward_eval.jsonl")

    train_name = f"{config.stage_name}_train"
    eval_name = f"{config.stage_name}_eval"
    definition = {
        "formatting": "alpaca",
        "ranking": True,
        "columns": {
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    register_dataset(dataset_dir, train_name, {**definition, "file_name": train_output.name})
    register_dataset(dataset_dir, eval_name, {**definition, "file_name": eval_output.name})
    return dataset_dir, train_name, eval_name


def build_training_recipe(
    raw_config: dict[str, Any],
    config: RewardStageConfig,
    dataset_dir: Path,
    train_name: str,
    eval_name: str,
) -> dict[str, Any]:
    """Translate project config into a LLaMA-Factory reward-model recipe."""

    project_root = raw_config["paths"]["project_root"]
    model_dir = resolve_path(config.training.model_dir, project_root)
    recipe = {
        "model_name_or_path": config.model.base_model_name,
        "trust_remote_code": config.model.trust_remote_code,
        "stage": "rm",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora" if config.lora.enabled else "full",
        "template": infer_template(config.model.base_model_name, config.llamafactory.template),
        "dataset_dir": str(dataset_dir),
        "dataset": train_name,
        "eval_dataset": eval_name,
        "preprocessing_num_workers": config.data.num_proc,
        "cutoff_len": config.data.max_length,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "warmup_ratio": config.training.warmup_ratio,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "eval_steps": config.training.eval_steps,
        "save_total_limit": config.training.save_total_limit,
        "output_dir": str(model_dir),
        "logging_dir": str(resolve_path(config.training.logging_dir, project_root)),
        "report_to": config.training.report_to,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": config.training.resume_from_checkpoint,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "seed": config.seed,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "quantization_bit": 4 if config.model.use_4bit else None,
        "double_quantization": config.model.bnb_4bit_use_double_quant if config.model.use_4bit else None,
        "quantization_type": config.model.bnb_4bit_quant_type if config.model.use_4bit else None,
        "lora_rank": config.lora.r if config.lora.enabled else None,
        "lora_alpha": config.lora.alpha if config.lora.enabled else None,
        "lora_dropout": config.lora.dropout if config.lora.enabled else None,
        "lora_target": ",".join(config.lora.target_modules) if config.lora.target_modules else None,
    }
    return merge_recipe(recipe, config.llamafactory.train_args)


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], RewardStageConfig]:
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    structured = dataclass_from_dict(RewardStageConfig, raw_config)
    return raw_config, structured


def main() -> None:
    args = parse_args()
    raw_config, config = load_config(args)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    model_dir = ensure_dir(resolve_path(config.training.model_dir, project_root))
    logger = setup_logger("train_reward", output_dir / "train.log")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)

    dataset_dir, train_name, eval_name = prepare_datasets(raw_config, config)
    recipe = build_training_recipe(raw_config, config, dataset_dir, train_name, eval_name)
    recipe_path = save_recipe(output_dir / "llamafactory_train.yaml", recipe)

    logger.info("Launching LLaMA-Factory reward training. output_dir=%s model_dir=%s", output_dir, model_dir)
    run_recipe(config.llamafactory.cli_path, "train", recipe_path, workdir=project_root)
    logger.info("Reward flow completed.")


if __name__ == "__main__":
    main()
