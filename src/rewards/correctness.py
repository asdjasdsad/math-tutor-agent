"""Correctness rewards and answer extraction utilities."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
ANSWER_PATTERNS = [
    re.compile(r"\u7b54\u6848[:\uff1a]\s*(.+)", flags=re.IGNORECASE),
    re.compile(r"\u6700\u7ec8\u7b54\u6848[:\uff1a]\s*(.+)", flags=re.IGNORECASE),
    re.compile(r"final answer[:\uff1a]?\s*(.+)", flags=re.IGNORECASE),
]


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output."""

    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().splitlines()[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].strip("\u3002.")


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("$", "\\(", "\\["):
        cleaned = cleaned.replace(prefix, "")
    for suffix in ("$", "\\)", "\\]"):
        cleaned = cleaned.replace(suffix, "")
    return cleaned.strip()


def normalize_answer(text: str) -> str:
    """Normalize common math answer surface forms."""

    cleaned = _strip_wrappers(text)
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace("\uff0e", ".")
    cleaned = cleaned.strip("\u3002.")
    return cleaned.lower()


def maybe_to_number(text: str) -> float | None:
    """Convert a normalized answer to a float when possible."""

    normalized = normalize_answer(text)
    if not normalized:
        return None
    try:
        return float(Decimal(normalized))
    except (InvalidOperation, ValueError):
        pass

    if "/" in normalized:
        try:
            return float(Fraction(normalized))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def answers_match(predicted: str, target: str, atol: float = 1e-6) -> bool:
    """Check whether two math answers should be considered equivalent."""

    normalized_pred = normalize_answer(predicted)
    normalized_target = normalize_answer(target)
    if normalized_pred == normalized_target:
        return True

    pred_num = maybe_to_number(normalized_pred)
    target_num = maybe_to_number(normalized_target)
    if pred_num is not None and target_num is not None:
        return math.isclose(pred_num, target_num, abs_tol=atol, rel_tol=atol)
    return False


def correctness_reward(output: str, gold_answer: str) -> float:
    """Return 1.0 for a correct answer and 0.0 otherwise."""

    predicted = extract_final_answer(output)
    return 1.0 if answers_match(predicted, gold_answer) else 0.0
