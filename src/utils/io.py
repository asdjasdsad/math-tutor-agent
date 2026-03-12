"""I/O helpers for config loading, output management, and lightweight serialization."""

from __future__ import annotations

import json
import subprocess
import types
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(RuntimeError):
    """Raised when a configuration file is invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected mapping at {path}, got {type(payload)!r}")
    return payload


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a mapping as UTF-8 YAML."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                row = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}")
            rows.append(row)
    return rows


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write iterable rows to JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def coerce_override_value(raw: str) -> Any:
    """Parse CLI override values into bool, number, list, dict, or string."""

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted CLI overrides to a config mapping."""

    if not overrides:
        return config
    merged = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        cursor = merged
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                existing = {}
                cursor[part] = existing
            if not isinstance(existing, dict):
                raise ConfigError(f"Cannot override nested key on non-mapping: {key}")
            cursor = existing
        cursor[parts[-1]] = coerce_override_value(raw_value)
    return merged


def load_stage_config(
    config_path: str | Path,
    paths_config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a stage config and optionally inject shared path config."""

    stage_config = load_yaml(config_path)
    if paths_config_path is not None:
        stage_config = deep_merge({"paths": load_yaml(paths_config_path)}, stage_config)
    return apply_overrides(stage_config, overrides)


def resolve_path(path_like: str | Path, project_root: str | Path = ".") -> Path:
    """Resolve a path relative to the project root."""

    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return Path(project_root).joinpath(candidate).resolve()


def save_config_snapshot(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Store a config snapshot under an output directory."""

    target = Path(output_dir) / "config_snapshot.yaml"
    save_yaml(target, config)
    return target


def get_git_state(project_root: str | Path = ".") -> dict[str, Any]:
    """Collect current git commit hash if available."""

    root = Path(project_root)
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                text=True,
            )
            .strip()
        )
        return {"git_available": True, "commit": commit, "branch": branch}
    except Exception:
        return {"git_available": False, "commit": None, "branch": None}


def save_git_state(output_dir: str | Path, project_root: str | Path = ".") -> Path:
    """Persist git metadata without failing when git is unavailable."""

    target = Path(output_dir) / "git_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(get_git_state(project_root), handle, ensure_ascii=False, indent=2)
    return target


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    """Instantiate nested dataclasses from dictionaries."""

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        field_value = payload.get(field.name, MISSING)
        if field_value is MISSING:
            continue
        annotation = type_hints.get(field.name, field.type)
        kwargs[field.name] = _coerce_value(annotation, field_value)
    return cls(**kwargs)  # type: ignore[arg-type]


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
            return dataclass_from_dict(annotation, value)
        return value

    if origin in {list, tuple}:
        inner_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_coerce_value(inner_type, item) for item in value]

    if origin is dict:
        key_type, val_type = get_args(annotation) or (Any, Any)
        return {
            _coerce_value(key_type, key): _coerce_value(val_type, val)
            for key, val in value.items()
        }

    if origin in {Union, types.UnionType}:
        for candidate in get_args(annotation):
            if candidate is type(None) and value is None:
                return None
            try:
                return _coerce_value(candidate, value)
            except Exception:
                continue
        return value

    return value
