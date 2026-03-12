"""Reward-model training entrypoint based on TRL RewardTrainer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.loaders import dataset_from_records, maybe_load_processed_dataset
from src.data.preprocess_reward import preprocess_reward_from_config
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
    paths: dict[str, Any]


class PairwiseDataCollator:
    """Pad chosen/rejected token pairs for RewardTrainer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_features = [
            {
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            }
            for feature in features
        ]
        rejected_features = [
            {
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            }
            for feature in features
        ]
        batch_chosen = self.tokenizer.pad(chosen_features, return_tensors="pt")
        batch_rejected = self.tokenizer.pad(rejected_features, return_tensors="pt")
        return {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reward model.")
    parser.add_argument("--config", required=True, help="Path to configs/reward.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def prepare_dataset(raw_config: dict[str, Any], config: RewardStageConfig, tokenizer: Any):
    """Load or preprocess reward data and tokenize pairwise samples."""

    project_root = raw_config["paths"]["project_root"]
    train_path = resolve_path(config.data.input_file, project_root)
    eval_path = resolve_path(config.data.eval_file, project_root)

    train_dataset = maybe_load_processed_dataset(train_path) if config.data.use_processed_if_available else None
    eval_dataset = maybe_load_processed_dataset(eval_path) if config.data.use_processed_if_available else None

    if train_dataset is None or eval_dataset is None:
        train_rows, eval_rows = preprocess_reward_from_config(raw_config)
        save_jsonl(train_path, train_rows)
        save_jsonl(eval_path, eval_rows)
        train_dataset = dataset_from_records(train_rows)
        eval_dataset = dataset_from_records(eval_rows)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        chosen_text = f"问题：{example['prompt']}\n\n回答：{example['chosen']}"
        rejected_text = f"问题：{example['prompt']}\n\n回答：{example['rejected']}"
        chosen = tokenizer(chosen_text, truncation=True, max_length=config.data.max_length)
        rejected = tokenizer(rejected_text, truncation=True, max_length=config.data.max_length)
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    train_dataset = train_dataset.map(_tokenize, num_proc=config.data.num_proc)
    eval_dataset = eval_dataset.map(_tokenize, num_proc=config.data.num_proc)
    return train_dataset, eval_dataset


def main() -> None:
    args = parse_args()
    raw_config = load_stage_config(args.config, args.paths_config, args.override)
    config = dataclass_from_dict(RewardStageConfig, raw_config)
    seed_everything(config.seed)

    project_root = raw_config["paths"]["project_root"]
    output_dir = ensure_dir(resolve_path(config.training.output_dir, project_root))
    model_dir = ensure_dir(resolve_path(config.training.model_dir, project_root))
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logger = setup_logger("train_reward", output_dir / "train.log")
    metrics_logger = MetricsLogger(output_dir / "metrics.jsonl")

    save_config_snapshot(output_dir, raw_config)
    save_git_state(output_dir, project_root)
    reset_peak_memory()

    from transformers import TrainingArguments
    from trl import RewardTrainer

    from src.models.lora_utils import build_lora_config
    from src.models.model_utils import load_sequence_classification_model, load_tokenizer

    tokenizer = load_tokenizer(
        config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    train_dataset, eval_dataset = prepare_dataset(raw_config, config, tokenizer)

    model = load_sequence_classification_model(
        model_name_or_path=config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
        use_4bit=config.model.use_4bit,
        compute_dtype=config.model.compute_dtype,
        num_labels=1,
    )

    peft_config = None
    if config.lora.enabled:
        peft_config = build_lora_config(
            r=config.lora.r,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
            task_type="SEQ_CLS",
        )

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        logging_dir=str(resolve_path(config.training.logging_dir, project_root)),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        report_to=config.training.report_to,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        logging_first_step=True,
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairwiseDataCollator(tokenizer),
        peft_config=peft_config,
    )

    logger.info("Starting reward-model training. resume_from_checkpoint=%s", config.training.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    trainer_state_path = checkpoints_dir / "trainer_state.json"
    if trainer_state_path.exists():
        (output_dir / "trainer_state.json").write_text(
            trainer_state_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if trainer.state.log_history:
        save_jsonl(output_dir / "metrics.jsonl", trainer.state.log_history)
    metrics_logger.log({"event": "train_result", **train_result.metrics})
    metrics_logger.log({"event": "gpu_state", **snapshot_gpu_state()})
    logger.info("Reward model saved to %s", model_dir)


if __name__ == "__main__":
    main()
