from src.rewards.combined import CombinedReward, RewardWeights
from src.rewards.correctness import correctness_reward
from src.rewards.format_reward import format_reward


class DummyScorer:
    def score(self, prompt: str, response: str) -> float:
        return 0.8


def test_correctness_reward_exact_match() -> None:
    assert correctness_reward("\u7b54\u6848\uff1a4", "4") == 1.0
    assert correctness_reward("\u7b54\u6848\uff1a5", "4") == 0.0


def test_format_reward_sections() -> None:
    text = "\u601d\u8def\uff1a\u5148\u5217\u5f0f\n\u6b65\u9aa4\uff1a\u9010\u6b65\u6c42\u89e3\n\u7b54\u6848\uff1a4"
    assert format_reward(text) == 1.0


def test_combined_reward_uses_all_components() -> None:
    combined = CombinedReward(
        weights=RewardWeights(correctness=0.5, rlaif=0.3, format=0.2),
        required_sections=["\u601d\u8def", "\u6b65\u9aa4", "\u7b54\u6848"],
        rlaif_scorer=DummyScorer(),
    )
    result = combined.score_components(
        "1+3=?",
        "\u601d\u8def\uff1a\u76f4\u63a5\u76f8\u52a0\n\u6b65\u9aa4\uff1a1+3=4\n\u7b54\u6848\uff1a4",
        "4",
    )
    assert result["correctness_reward"] == 1.0
    assert result["format_reward"] == 1.0
    assert result["rlaif_reward"] == 0.8
    assert result["combined_reward"] > 0.9
