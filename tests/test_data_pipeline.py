from src.data.preprocess_reward import normalize_reward_example, pairwise_from_score_rows
from src.data.preprocess_rl import normalize_rl_example
from src.data.preprocess_sft import normalize_sft_example, render_sft_text
from src.data.schemas import RewardRecord


def test_normalize_sft_example() -> None:
    record = normalize_sft_example(
        {
            "question": "\u6c42 1+1",
            "solution": "\u6b65\u9aa4\uff1a1+1=2\n\u7b54\u6848\uff1a2",
            "answer": "2",
        },
        source_name="unit",
    )
    assert record.prompt == "\u6c42 1+1"
    assert record.answer == "2"
    assert "\u7b54\u6848" in record.response
    assert "\u6c42 1+1" in render_sft_text(record)


def test_normalize_reward_example_pairwise() -> None:
    record = normalize_reward_example(
        {"prompt": "1+1=?", "chosen": "\u7b54\u6848\uff1a2", "rejected": "\u7b54\u6848\uff1a3"},
        source_name="unit",
    )
    assert record is not None
    assert record.chosen == "\u7b54\u6848\uff1a2"
    assert record.rejected == "\u7b54\u6848\uff1a3"


def test_pairwise_from_score_rows() -> None:
    rows = [
        RewardRecord(prompt="1+1=?", chosen="\u7b54\u6848\uff1a2", rejected="", score_chosen=1.0),
        RewardRecord(prompt="1+1=?", chosen="\u7b54\u6848\uff1a3", rejected="", score_chosen=0.0),
    ]
    pairs = pairwise_from_score_rows(rows)
    assert len(pairs) == 1
    assert pairs[0].chosen == "\u7b54\u6848\uff1a2"
    assert pairs[0].rejected == "\u7b54\u6848\uff1a3"


def test_normalize_rl_example() -> None:
    record = normalize_rl_example({"question": "\u6c42 2+2", "answer": "4"}, source_name="unit")
    assert "\u9898\u76ee\uff1a" in record.prompt
    assert record.answer == "4"
