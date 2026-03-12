"""Prepare prompt-only RL data from GSM8K or compatible math datasets."""

from __future__ import annotations

import argparse
from typing import Any

from src.data.loaders import load_hf_dataset, maybe_load_processed_dataset
from src.data.schemas import RlPromptRecord
from src.utils.io import load_stage_config, resolve_path, save_jsonl


def _pick_first(mapping: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def build_rl_prompt(question: str) -> str:
    """Convert a math question into the tutor-style prompt used for RL."""

    return (
        "\u8bf7\u50cf\u6570\u5b66\u8f85\u5bfc\u8001\u5e08\u4e00\u6837\u89e3\u7b54\u4e0b\u9762\u7684\u95ee\u9898\u3002\n"
        "\u8981\u6c42\u6309\u201c\u601d\u8def\u3001\u6b65\u9aa4\u3001\u7b54\u6848\u201d\u7ed9\u51fa\u7ed3\u6784\u5316\u8f93\u51fa\uff0c\u5e76\u786e\u4fdd\u6700\u7ec8\u7b54\u6848\u660e\u786e\u3002\n\n"
        f"\u9898\u76ee\uff1a{question.strip()}"
    )


def normalize_rl_example(example: dict[str, Any], source_name: str) -> RlPromptRecord:
    """Normalize a raw prompt example into prompt-only RL format."""

    question = _pick_first(
        example,
        ["question", "problem", "prompt", "instruction", "query"],
    )
    answer = _pick_first(example, ["answer", "target", "final_answer"])
    return RlPromptRecord(
        prompt=build_rl_prompt(question),
        answer=answer,
        metadata={"source": source_name, "raw_keys": sorted(example.keys())},
    )


def convert_rl_dataset(raw_dataset, source_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert an HF split to prompt-only RL rows."""

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(raw_dataset):
        if limit is not None and index >= limit:
            break
        rows.append(normalize_rl_example(dict(example), source_name=source_name).to_dict())
    return rows


def preprocess_rl_from_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare train/eval RL rows from config settings."""

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    cache_dir = data_cfg.get("cache_dir")

    train_dataset = load_hf_dataset(
        dataset_name=dataset_name,
        dataset_config_name=data_cfg.get("dataset_config_name"),
        split=data_cfg["train_split"],
        cache_dir=cache_dir,
    )
    eval_dataset = load_hf_dataset(
        dataset_name=dataset_name,
        dataset_config_name=data_cfg.get("dataset_config_name"),
        split=data_cfg["eval_split"],
        cache_dir=cache_dir,
    )
    train_rows = convert_rl_dataset(train_dataset, dataset_name, data_cfg.get("max_train_samples"))
    eval_rows = convert_rl_dataset(eval_dataset, dataset_name, data_cfg.get("max_eval_samples"))
    return train_rows, eval_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RL prompt-only data.")
    parser.add_argument("--config", required=True, help="Path to PPO or GRPO config file")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage_config(args.config, args.paths_config, args.override)
    project_root = config["paths"]["project_root"]
    data_cfg = config["data"]

    train_path = resolve_path(data_cfg["input_file"], project_root)
    eval_path = resolve_path(data_cfg["eval_file"], project_root)

    if data_cfg.get("use_processed_if_available"):
        existing_train = maybe_load_processed_dataset(train_path)
        existing_eval = maybe_load_processed_dataset(eval_path)
        if existing_train is not None and existing_eval is not None:
            print(f"Processed RL data already exists at {train_path} and {eval_path}")
            return

    train_rows, eval_rows = preprocess_rl_from_config(config)
    save_jsonl(train_path, train_rows)
    save_jsonl(eval_path, eval_rows)
    print(f"Saved {len(train_rows)} train rows to {train_path}")
    print(f"Saved {len(eval_rows)} eval rows to {eval_path}")


if __name__ == "__main__":
    main()
