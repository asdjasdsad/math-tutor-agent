"""Simple format reward for tutor-style structured answers."""

from __future__ import annotations

from collections.abc import Iterable


def required_section_flags(text: str, required_sections: Iterable[str]) -> dict[str, bool]:
    """Return presence flags for each required heading."""

    return {section: section in text for section in required_sections}


def format_reward(text: str, required_sections: list[str] | tuple[str, ...] | None = None) -> float:
    """Reward outputs that include the configured tutor sections."""

    sections = list(required_sections or ["\u601d\u8def", "\u6b65\u9aa4", "\u7b54\u6848"])
    if not sections:
        return 1.0
    flags = required_section_flags(text, sections)
    return sum(1.0 for is_present in flags.values() if is_present) / len(sections)
