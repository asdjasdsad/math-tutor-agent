"""SFT training entrypoint backed by LLaMA-Factory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.preprocess_sft import DEFAULT_SYSTEM_PROMPT, preprocess_sft_from_config
from src.trainers.llamafactory_runner import (
    LlamaFactoryConfig,
    infer_template,
    load_jsonl_rows,
    merge_recipe,
    prepare_dataset_dir,
    register_dataset,
    run_recipe,
    save_recipe,
    write_sft_dataset,
    build_export_recipe,
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
    attn_implementation: str | None = "sdpa"


@dataclass
class DataConfig:
    dataset_name: str
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str = "train[:512]"
    input_file: str = "data/processed/sft_train.jsonl"
    eval_file: str = "data/processed/sft_eval.jsonl"
    use_processed_if_available: bool = True
    cache_dir: str = "data/cache"
    num_proc: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    val_ratio: float = 0.02
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class TrainingConfig:
    output_dir: str
    logging_dir: str
    adapter_dir: str
    merged_dir: str
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    max_seq_length: int = 2048
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    packing: bool = False
    resume_from_checkpoint: str | None = None
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    auto_merge_after_training: bool = False


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class SFTStageConfig:
    stage_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoraConfig
    llamafactory: LlamaFactoryConfig = field(default_factory=LlamaFactoryConfig)
    paths: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SFT model with LLaMA-Factory.")
    parser.add_argument("--config", required=True, help="Path to configs/sft.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], SFTStageConfig]:
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    structured = dataclass_from_dict(SFTStageConfig, raw_config)
    return raw_config, structured


def prepare_datasets(raw_config: dict[str, Any], config: SFTStageConfig) -> tuple[Path, str, str]:
    """Ensure processed data exists and emit a ShareGPT dataset for LLaMA-Factory."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)
    eval_path = resolve_path(config.data.eval_file, project_root)

    if not (config.data.use_processed_if_available and train_path.exists() and eval_path.exists()):
        train_rows, eval_rows = preprocess_sft_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(eval_path, eval_rows)

    dataset_dir = prepare_dataset_dir(config.llamafactory.dataset_dir, project_root)
    system_prompt = config.data.system_prompt or DEFAULT_SYSTEM_PROMPT
    train_output = write_sft_dataset(load_jsonl_rows(train_path), dataset_dir / "sft_train.jsonl", system_prompt)
    eval_output = write_sft_dataset(load_jsonl_rows(eval_path), dataset_dir / "sft_eval.jsonl", system_prompt)

    train_name = f"{config.stage_name}_train"
    eval_name = f"{config.stage_name}_eval"
    definition = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
    }
    register_dataset(dataset_dir, train_name, {**definition, "file_name": train_output.name})
    register_dataset(dataset_dir, eval_name, {**definition, "file_name": eval_output.name})
    return dataset_dir, train_name, eval_name


def build_training_recipe(
    raw_config: dict[str, Any],
    config: SFTStageConfig,
    dataset_dir: Path,
    train_name: str,
    eval_name: str,
) -> dict[str, Any]:
    """Translate project config into a LLaMA-Factory SFT recipe."""

    project_root = raw_config["paths"]["project_root"]
    adapter_dir = resolve_path(config.training.adapter_dir, project_root)
    template = infer_template(config.model.base_model_name, config.llamafactory.template)
    recipe = {
        "model_name_or_path": config.model.base_model_name,
        "trust_remote_code": config.model.trust_remote_code,
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora" if config.lora.enabled else "full",
        "template": template,
        "dataset_dir": str(dataset_dir),
        "dataset": train_name,
        "eval_dataset": eval_name,
        "preprocessing_num_workers": config.data.num_proc,
        "cutoff_len": config.training.max_seq_length,
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
        "output_dir": str(adapter_dir),
        "logging_dir": str(resolve_path(config.training.logging_dir, project_root)),
        "report_to": config.training.report_to,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": config.training.resume_from_checkpoint,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "packing": config.training.packing,
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


def main() -> None:
    args = parse_args()
    raw_config, config = load_config(args)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    adapter_dir = ensure_dir(resolve_path(config.training.adapter_dir, project_root))
    merged_dir = resolve_path(config.training.merged_dir, project_root)
    logger = setup_logger("train_sft", output_dir / "train.log")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)

    dataset_dir, train_name, eval_name = prepare_datasets(raw_config, config)
    recipe = build_training_recipe(raw_config, config, dataset_dir, train_name, eval_name)
    recipe_path = save_recipe(output_dir / "llamafactory_train.yaml", recipe)

    logger.info("Launching LLaMA-Factory SFT. output_dir=%s adapter_dir=%s", output_dir, adapter_dir)
    run_recipe(config.llamafactory.cli_path, "train", recipe_path, workdir=project_root)

    if config.training.auto_merge_after_training:
        export_recipe = build_export_recipe(
            base_model_name=config.model.base_model_name,
            adapter_dir=adapter_dir,
            export_dir=merged_dir,
            template=infer_template(config.model.base_model_name, config.llamafactory.template),
            trust_remote_code=config.model.trust_remote_code,
            overrides=config.llamafactory.export_args,
        )
        export_path = save_recipe(output_dir / "llamafactory_export.yaml", export_recipe)
        logger.info("Exporting merged model to %s", merged_dir)
        run_recipe(config.llamafactory.cli_path, "export", export_path, workdir=project_root)

    logger.info("SFT flow completed.")


if __name__ == "__main__":
    main()
