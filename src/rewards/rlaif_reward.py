"""Reward-model-backed RLAIF scoring."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.reward_model import RewardModelScorer


@dataclass
class RLAIFReward:
    """Thin wrapper around the reward model scorer."""

    model_path: str
    trust_remote_code: bool = True
    use_4bit: bool = False
    compute_dtype: str = "bfloat16"
    max_length: int = 2048

    def __post_init__(self) -> None:
        self.scorer = RewardModelScorer(
            model_name_or_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_4bit=self.use_4bit,
            compute_dtype=self.compute_dtype,
            max_length=self.max_length,
        )

    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt/response pair."""

        return float(self.scorer.score(prompt, response))

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        """Score a batch of prompt/response pairs."""

        return [float(x) for x in self.scorer.score_batch(prompts, responses)]
