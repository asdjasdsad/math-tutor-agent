from src.rewards.correctness import answers_match, extract_final_answer, normalize_answer


def test_extract_boxed_answer() -> None:
    assert extract_final_answer("推导过程...\n\\boxed{7}") == "7"


def test_extract_answer_heading() -> None:
    assert extract_final_answer("\u6b65\u9aa4\uff1a\u7565\n\u7b54\u6848\uff1a42") == "42"


def test_numeric_normalization() -> None:
    assert normalize_answer(" 1,200 ") == "1200"
    assert answers_match("0.5", "1/2")
