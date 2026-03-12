"""Prepare reward-model data into pairwise records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from src.data.loaders import load_hf_dataset, maybe_load_processed_dataset
from src.data.schemas import RewardRecord
from src.utils.io import load_stage_config, resolve_path, save_jsonl


def _pick_first(mapping: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def normalize_reward_example(example: dict[str, Any], source_name: str) -> RewardRecord | None:
    """Normalize a raw reward example into chosen/rejected form."""

    prompt = _pick_first(example, ["prompt", "instruction", "question", "query"])
    answer = _pick_first(example, ["answer", "target", "final_answer"])

    if isinstance(example.get("chosen"), str) and isinstance(example.get("rejected"), str):
        return RewardRecord(
            prompt=prompt,
            chosen=example["chosen"].strip(),
            rejected=example["rejected"].strip(),
            answer=answer,
            metadata={"source": source_name, "raw_keys": sorted(example.keys())},
        )

    completions = example.get("completions")
    if isinstance(completions, list) and len(completions) >= 2:
        ranked = []
        for item in completions:
            if not isinstance(item, dict):
                continue
            text = _pick_first(item, ["response", "text", "content", "completion"])
            score = item.get("score", item.get("rating"))
            if text:
                ranked.append((float(score) if score is not None else 0.0, text))
        if len(ranked) >= 2:
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            return RewardRecord(
                prompt=prompt,
                chosen=ranked[0][1],
                rejected=ranked[-1][1],
                answer=answer,
                score_chosen=ranked[0][0],
                score_rejected=ranked[-1][0],
                metadata={"source": source_name, "raw_keys": sorted(example.keys())},
            )

    response = _pick_first(example, ["response", "output", "completion"])
    score = example.get("score", example.get("rating"))
    if prompt and response and score is not None:
        return RewardRecord(
            prompt=prompt,
            chosen=response,
            rejected="",
            answer=answer,
            score_chosen=float(score),
            metadata={"source": source_name, "raw_keys": sorted(example.keys())},
        )

    return None


def pairwise_from_score_rows(rows: list[RewardRecord]) -> list[RewardRecord]:
    """Convert score-style rows into pairwise rows grouped by prompt."""

    grouped: dict[str, list[RewardRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt].append(row)

    pairs: list[RewardRecord] = []
    for prompt, records in grouped.items():
        ranked = sorted(
            [record for record in records if record.score_chosen is not None],
            key=lambda record: float(record.score_chosen or 0.0),
            reverse=True,
        )
        if len(ranked) < 2:
            continue
        best = ranked[0]
        worst = ranked[-1]
        pairs.append(
            RewardRecord(
                prompt=prompt,
                chosen=best.chosen,
                rejected=worst.chosen,
                answer=best.answer or worst.answer,
                score_chosen=best.score_chosen,
                score_rejected=worst.score_chosen,
                metadata={"source": best.metadata.get("source", "score_pairwise")},
            )
        )
    return pairs


def convert_reward_dataset(raw_dataset, source_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert an HF reward split into pairwise JSON rows."""

    normalized: list[RewardRecord] = []
    for index, example in enumerate(raw_dataset):
        if limit is not None and index >= limit:
            break
        record = normalize_reward_example(dict(example), source_name=source_name)
        if record is not None:
            normalized.append(record)

    if normalized and any(record.rejected == "" for record in normalized):
        normalized = pairwise_from_score_rows(normalized)
    else:
        normalized = [record for record in normalized if record.rejected]

    return [record.to_dict() for record in normalized]


def preprocess_reward_from_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare train/eval reward rows from config settings."""

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

    train_rows = convert_reward_dataset(train_dataset, dataset_name, data_cfg.get("max_train_samples"))
    eval_rows = convert_reward_dataset(eval_dataset, dataset_name, data_cfg.get("max_eval_samples"))
    return train_rows, eval_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reward-model data into pairwise JSONL.")
    parser.add_argument("--config", required=True, help="Path to configs/reward.yaml")
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
            print(f"Processed reward data already exists at {train_path} and {eval_path}")
            return

    train_rows, eval_rows = preprocess_reward_from_config(config)
    save_jsonl(train_path, train_rows)
    save_jsonl(eval_path, eval_rows)
    print(f"Saved {len(train_rows)} train rows to {train_path}")
    print(f"Saved {len(eval_rows)} eval rows to {eval_path}")


if __name__ == "__main__":
    main()
