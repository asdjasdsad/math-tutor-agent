"""Helpers for running project training stages through LLaMA-Factory."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.preprocess_sft import DEFAULT_SYSTEM_PROMPT, build_chat_messages
from src.data.schemas import RewardRecord, RlPromptRecord, UnifiedRecord
from src.utils.io import deep_merge, ensure_dir, load_jsonl, resolve_path, save_jsonl, save_yaml


@dataclass
class LlamaFactoryConfig:
    """Runtime settings for the local LLaMA-Factory integration."""

    cli_path: str = "llamafactory-cli"
    dataset_dir: str = "data/llamafactory"
    template: str | None = None
    train_args: dict[str, Any] = field(default_factory=dict)
    export_args: dict[str, Any] = field(default_factory=dict)


def infer_template(model_name_or_path: str, override: str | None = None) -> str:
    """Infer a chat template from the model identifier when not configured."""

    if override:
        return override
    lowered = model_name_or_path.lower()
    if "qwen" in lowered:
        return "qwen"
    if "llama-3" in lowered or "llama3" in lowered:
        return "llama3"
    return "default"


def ensure_cli_available(cli_path: str) -> str:
    """Resolve the LLaMA-Factory executable or raise a clear error."""

    resolved = shutil.which(cli_path)
    if resolved:
        return resolved
    raise RuntimeError(
        f"Cannot find `{cli_path}`. Install LLaMA-Factory in the target conda environment first."
    )


def prepare_dataset_dir(dataset_dir: str | Path, project_root: str | Path = ".") -> Path:
    """Create and resolve the directory that stores LLaMA-Factory dataset assets."""

    return ensure_dir(resolve_path(dataset_dir, project_root))


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows from a local file."""

    return load_jsonl(path)


def write_sft_dataset(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Path:
    """Convert processed SFT rows into ShareGPT-style JSONL."""

    target = Path(output_path)
    converted = []
    for row in rows:
        record = UnifiedRecord(
            prompt=row["prompt"],
            response=row.get("response", ""),
            answer=row.get("answer", ""),
            metadata=row.get("metadata", {}),
        )
        converted.append(
            {
                "messages": build_chat_messages(record, system_prompt=system_prompt),
                "answer": record.answer,
                "metadata": record.metadata,
            }
        )
    save_jsonl(target, converted)
    return target


def write_reward_dataset(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Convert processed reward rows into pairwise JSONL."""

    target = Path(output_path)
    converted = []
    for row in rows:
        record = RewardRecord(
            prompt=row["prompt"],
            chosen=row["chosen"],
            rejected=row["rejected"],
            answer=row.get("answer", ""),
            score_chosen=row.get("score_chosen"),
            score_rejected=row.get("score_rejected"),
            metadata=row.get("metadata", {}),
        )
        converted.append(
            {
                "prompt": record.prompt,
                "chosen": record.chosen,
                "rejected": record.rejected,
                "answer": record.answer,
                "metadata": record.metadata,
            }
        )
    save_jsonl(target, converted)
    return target


def write_ppo_dataset(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Convert processed RL prompt rows into Alpaca-style JSONL for PPO."""

    target = Path(output_path)
    converted = []
    for row in rows:
        record = RlPromptRecord(
            prompt=row["prompt"],
            answer=row.get("answer", ""),
            metadata=row.get("metadata", {}),
        )
        converted.append(
            {
                "instruction": record.prompt,
                "input": "",
                "output": "",
                "answer": record.answer,
                "metadata": record.metadata,
            }
        )
    save_jsonl(target, converted)
    return target


def register_dataset(dataset_dir: str | Path, dataset_name: str, definition: dict[str, Any]) -> Path:
    """Upsert a dataset entry into `dataset_info.json`."""

    target_dir = Path(dataset_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    registry_path = target_dir / "dataset_info.json"
    registry: dict[str, Any] = {}
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
    registry[dataset_name] = definition
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
    return registry_path


def detect_reward_model_type(model_path: str | Path) -> str:
    """Infer whether a reward model path is a merged model or a LoRA adapter."""

    candidate = Path(model_path)
    adapter_markers = {
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
    if candidate.is_dir() and any((candidate / marker).exists() for marker in adapter_markers):
        return "lora"
    return "full"


def save_recipe(output_path: str | Path, recipe: dict[str, Any]) -> Path:
    """Persist a generated LLaMA-Factory YAML recipe."""

    cleaned = _drop_empty(recipe)
    save_yaml(output_path, cleaned)
    return Path(output_path)


def run_recipe(
    cli_path: str,
    command: str,
    recipe_path: str | Path,
    workdir: str | Path = ".",
) -> None:
    """Run `llamafactory-cli <command> <recipe>` and stream logs to the caller."""

    executable = ensure_cli_available(cli_path)
    subprocess.run(
        [executable, command, str(recipe_path)],
        cwd=Path(workdir),
        check=True,
    )


def merge_recipe(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Combine generated defaults with caller-supplied LLaMA-Factory arguments."""

    return deep_merge(base, overrides or {})


def build_export_recipe(
    base_model_name: str,
    adapter_dir: str | Path,
    export_dir: str | Path,
    template: str,
    trust_remote_code: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an export recipe used to merge a LoRA adapter."""

    recipe = {
        "model_name_or_path": str(base_model_name),
        "adapter_name_or_path": str(adapter_dir),
        "template": template,
        "finetuning_type": "lora",
        "export_dir": str(export_dir),
        "trust_remote_code": trust_remote_code,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    return merge_recipe(recipe, overrides)


def _drop_empty(value: Any) -> Any:
    """Recursively remove `None`, empty strings, and empty containers from recipes."""

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _drop_empty(item)
            if normalized is None:
                continue
            if normalized == {} or normalized == []:
                continue
            cleaned[key] = normalized
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            normalized = _drop_empty(item)
            if normalized is None:
                continue
            cleaned_list.append(normalized)
        return cleaned_list
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value
