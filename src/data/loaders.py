"""Dataset loading helpers for Hugging Face datasets and local JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl


def load_hf_dataset(
    dataset_name: str,
    split: str,
    dataset_config_name: str | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
):
    """Load a Hugging Face dataset with lazy imports."""

    from datasets import load_dataset

    return load_dataset(
        path=dataset_name,
        name=dataset_config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a local JSONL file and return rows."""

    return load_jsonl(path)


def dataset_from_records(records: list[dict[str, Any]]):
    """Convert in-memory records into a Hugging Face `Dataset`."""

    from datasets import Dataset

    return Dataset.from_list(records)


def maybe_load_processed_dataset(path: str | Path | None):
    """Load a JSONL dataset if the file exists."""

    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return dataset_from_records(load_jsonl_records(target))
