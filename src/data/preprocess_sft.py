"""Prepare SFT data from heterogeneous math datasets into a unified JSONL format."""

from __future__ import annotations

import argparse
import re
from typing import Any

from src.data.loaders import load_hf_dataset, maybe_load_processed_dataset
from src.data.schemas import UnifiedRecord
from src.utils.io import load_stage_config, resolve_path, save_jsonl

DEFAULT_SYSTEM_PROMPT = (
    "\u4f60\u662f\u4e00\u540d\u8010\u5fc3\u3001\u4e25\u8c28\u3001\u64c5\u957f\u5206\u6b65\u8bb2\u89e3\u7684\u6570\u5b66\u8001\u5e08\u3002"
    "\u8bf7\u6309\u201c\u9898\u610f\u7406\u89e3\u3001\u601d\u8def\u3001\u6b65\u9aa4\u3001\u7b54\u6848\u3001\u603b\u7ed3\u201d\u7ed3\u6784\u4f5c\u7b54\uff0c\u5e76\u786e\u4fdd\u6700\u7ec8\u7b54\u6848\u660e\u786e\u3002"
)


def _pick_first(mapping: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _extract_answer(text: str) -> str:
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"\u7b54\u6848[:\uff1a]\s*(.+)",
        r"final answer[:\uff1a]?\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip().strip(".\u3002")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def format_tutor_response(solution: str, answer: str) -> str:
    """Wrap raw reasoning into the tutor style expected by the project."""

    cleaned_lines = [line.strip() for line in solution.splitlines() if line.strip()]
    thought = (
        cleaned_lines[0]
        if cleaned_lines
        else "\u5148\u5206\u6790\u9898\u76ee\u6761\u4ef6\uff0c\u518d\u9010\u6b65\u6c42\u89e3\u3002"
    )
    steps = (
        "\n".join(cleaned_lines)
        if cleaned_lines
        else "\u8bf7\u6839\u636e\u9898\u76ee\u6761\u4ef6\u9010\u6b65\u63a8\u5bfc\u3002"
    )
    final_answer = answer or _extract_answer(solution) or "待求"
    return (
        "\u9898\u610f\u7406\u89e3\uff1a\n"
        "\u8fd9\u9053\u9898\u9700\u8981\u6839\u636e\u5df2\u77e5\u6761\u4ef6\u8fdb\u884c\u6570\u5b66\u63a8\u5bfc\uff0c\u5e76\u7ed9\u51fa\u6e05\u6670\u7684\u7ed3\u8bba\u3002\n\n"
        "\u601d\u8def\uff1a\n"
        f"{thought}\n\n"
        "\u6b65\u9aa4\uff1a\n"
        f"{steps}\n\n"
        "\u7b54\u6848\uff1a\n"
        f"{final_answer}\n\n"
        "\u603b\u7ed3\uff1a\n"
        "\u5148\u6839\u636e\u6761\u4ef6\u5efa\u7acb\u5173\u7cfb\uff0c\u518d\u9010\u6b65\u5316\u7b80\u5e76\u68c0\u67e5\u6700\u7ec8\u7ed3\u679c\u3002"
    )


def normalize_sft_example(example: dict[str, Any], source_name: str) -> UnifiedRecord:
    """Normalize a raw SFT example into the unified internal schema."""

    prompt = _pick_first(
        example,
        ["prompt", "problem", "question", "instruction", "query", "input"],
    )
    solution = _pick_first(
        example,
        ["response", "solution", "output", "completion", "reasoning", "cot"],
    )
    answer = _pick_first(
        example,
        ["answer", "final_answer", "target", "label", "final"],
        default=_extract_answer(solution),
    )
    response = format_tutor_response(solution or answer, answer)
    return UnifiedRecord(
        prompt=prompt,
        response=response,
        answer=answer,
        metadata={"source": source_name, "raw_keys": sorted(example.keys())},
    )


def build_chat_messages(record: UnifiedRecord, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Construct chat messages for SFT chat-template rendering."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record.prompt},
        {"role": "assistant", "content": record.response},
    ]


def render_sft_text(
    record: UnifiedRecord,
    tokenizer: Any | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a record into a text sample suitable for `SFTTrainer`."""

    messages = build_chat_messages(record, system_prompt=system_prompt)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{record.prompt}\n"
        f"<|assistant|>\n{record.response}"
    )


def convert_sft_dataset(raw_dataset, source_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert an HF dataset split to unified records."""

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(raw_dataset):
        if limit is not None and index >= limit:
            break
        record = normalize_sft_example(dict(example), source_name=source_name)
        rows.append(record.to_dict())
    return rows


def preprocess_sft_from_config(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare train/eval SFT records from config settings."""

    data_cfg = config["data"]
    source_name = data_cfg["dataset_name"]
    cache_dir = data_cfg.get("cache_dir")

    train_dataset = load_hf_dataset(
        dataset_name=source_name,
        dataset_config_name=data_cfg.get("dataset_config_name"),
        split=data_cfg["train_split"],
        cache_dir=cache_dir,
    )
    eval_split = data_cfg.get("eval_split") or data_cfg["train_split"]
    eval_dataset = load_hf_dataset(
        dataset_name=source_name,
        dataset_config_name=data_cfg.get("dataset_config_name"),
        split=eval_split,
        cache_dir=cache_dir,
    )

    train_rows = convert_sft_dataset(train_dataset, source_name, data_cfg.get("max_train_samples"))
    eval_rows = convert_sft_dataset(eval_dataset, source_name, data_cfg.get("max_eval_samples"))
    return train_rows, eval_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT data into unified JSONL format.")
    parser.add_argument("--config", required=True, help="Path to configs/sft.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths.yaml")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage_config(args.config, args.paths_config, args.override)
    project_root = config["paths"]["project_root"]
    data_cfg = config["data"]

    processed_train = resolve_path(data_cfg["input_file"], project_root)
    processed_eval = resolve_path(data_cfg["eval_file"], project_root)

    if data_cfg.get("use_processed_if_available"):
        existing_train = maybe_load_processed_dataset(processed_train)
        existing_eval = maybe_load_processed_dataset(processed_eval)
        if existing_train is not None and existing_eval is not None:
            print(f"Processed SFT data already exists at {processed_train} and {processed_eval}")
            return

    train_rows, eval_rows = preprocess_sft_from_config(config)
    save_jsonl(processed_train, train_rows)
    save_jsonl(processed_eval, eval_rows)
    print(f"Saved {len(train_rows)} train rows to {processed_train}")
    print(f"Saved {len(eval_rows)} eval rows to {processed_eval}")


if __name__ == "__main__":
    main()
