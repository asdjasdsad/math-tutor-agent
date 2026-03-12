"""Evaluation metrics shared across GSM8K, MATH-500, and RL comparison."""

from __future__ import annotations

from typing import Iterable

from src.rewards.correctness import answers_match, extract_final_answer
from src.rewards.format_reward import format_reward


def exact_match(prediction: str, reference: str) -> float:
    """Exact-match style score on extracted final answers."""

    return 1.0 if answers_match(extract_final_answer(prediction), reference) else 0.0


def final_answer_accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Average exact match over extracted final answers."""

    pairs = list(zip(predictions, references))
    if not pairs:
        return 0.0
    scores = [exact_match(prediction, reference) for prediction, reference in pairs]
    return sum(scores) / len(scores)


def format_pass_rate(
    predictions: Iterable[str],
    required_sections: list[str] | None = None,
    threshold: float = 1.0,
) -> float:
    """Share of outputs whose format reward reaches the threshold."""

    outputs = list(predictions)
    if not outputs:
        return 0.0
    scores = [format_reward(output, required_sections) for output in outputs]
    return sum(1.0 for score in scores if score >= threshold) / len(scores)
