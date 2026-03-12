"""Data schemas used across preprocessing, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UnifiedRecord:
    """Canonical record format shared across datasets."""

    prompt: str
    response: str = ""
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly mapping."""

        return asdict(self)


@dataclass
class RewardRecord:
    """Canonical reward-model record."""

    prompt: str
    chosen: str
    rejected: str
    answer: str = ""
    score_chosen: float | None = None
    score_rejected: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly mapping."""

        return asdict(self)


@dataclass
class RlPromptRecord:
    """Prompt-only RL record with an optional reference answer."""

    prompt: str
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly mapping."""

        return asdict(self)
