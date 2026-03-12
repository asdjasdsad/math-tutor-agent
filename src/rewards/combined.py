"""Composable reward aggregation for PPO and GRPO."""

from __future__ import annotations

from dataclasses import dataclass

from src.rewards.correctness import correctness_reward
from src.rewards.format_reward import format_reward


@dataclass
class RewardWeights:
    """Configurable reward weights."""

    correctness: float = 0.5
    rlaif: float = 0.3
    format: float = 0.2


class CombinedReward:
    """Combine correctness, RLAIF, and format rewards."""

    def __init__(
        self,
        weights: RewardWeights,
        required_sections: list[str] | None = None,
        rlaif_scorer: object | None = None,
    ) -> None:
        self.weights = weights
        self.required_sections = required_sections or [
            "\u601d\u8def",
            "\u6b65\u9aa4",
            "\u7b54\u6848",
        ]
        self.rlaif_scorer = rlaif_scorer

    def score_components(self, prompt: str, response: str, answer: str) -> dict[str, float]:
        """Return individual reward components for a sample."""

        correctness = correctness_reward(response, answer) if answer else 0.0
        structure = format_reward(response, self.required_sections)
        rlaif = float(self.rlaif_scorer.score(prompt, response)) if self.rlaif_scorer is not None else 0.0
        total = (
            self.weights.correctness * correctness
            + self.weights.rlaif * rlaif
            + self.weights.format * structure
        )
        return {
            "correctness_reward": correctness,
            "rlaif_reward": rlaif,
            "format_reward": structure,
            "combined_reward": total,
        }

    def score(self, prompt: str, response: str, answer: str) -> float:
        """Return only the weighted combined reward."""

        return self.score_components(prompt, response, answer)["combined_reward"]

    def score_batch(self, prompts: list[str], responses: list[str], answers: list[str]) -> list[dict[str, float]]:
        """Score a batch and keep component visibility for logging."""

        return [
            self.score_components(prompt, response, answer)
            for prompt, response, answer in zip(prompts, responses, answers)
        ]
