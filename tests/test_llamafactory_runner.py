import json

from src.trainers.llamafactory_runner import (
    build_export_recipe,
    detect_reward_model_type,
    infer_template,
    register_dataset,
    write_ppo_dataset,
    write_reward_dataset,
    write_sft_dataset,
)
from src.utils.io import load_jsonl


def test_write_sft_dataset_uses_sharegpt_messages(tmp_path) -> None:
    output = write_sft_dataset(
        [
            {
                "prompt": "求 1+1",
                "response": "答案：2",
                "answer": "2",
                "metadata": {"source": "unit"},
            }
        ],
        tmp_path / "sft.jsonl",
        system_prompt="你是数学老师。",
    )
    rows = load_jsonl(output)
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["role"] == "user"
    assert rows[0]["messages"][2]["role"] == "assistant"


def test_write_reward_dataset_keeps_pairwise_columns(tmp_path) -> None:
    output = write_reward_dataset(
        [{"prompt": "1+1=?", "chosen": "答案：2", "rejected": "答案：3"}],
        tmp_path / "reward.jsonl",
    )
    rows = load_jsonl(output)
    assert rows[0]["chosen"] == "答案：2"
    assert rows[0]["rejected"] == "答案：3"


def test_write_ppo_dataset_uses_alpaca_columns(tmp_path) -> None:
    output = write_ppo_dataset(
        [{"prompt": "题目：求 2+2", "answer": "4"}],
        tmp_path / "ppo.jsonl",
    )
    rows = load_jsonl(output)
    assert rows[0]["instruction"] == "题目：求 2+2"
    assert rows[0]["output"] == ""


def test_register_dataset_merges_entries(tmp_path) -> None:
    registry_path = register_dataset(tmp_path, "sft_train", {"file_name": "sft.jsonl"})
    register_dataset(tmp_path, "reward_train", {"file_name": "reward.jsonl"})
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert sorted(payload) == ["reward_train", "sft_train"]


def test_infer_template_from_model_name() -> None:
    assert infer_template("Qwen/Qwen2.5-1.5B-Instruct") == "qwen"


def test_detect_reward_model_type_from_adapter_dir(tmp_path) -> None:
    adapter_dir = tmp_path / "reward_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert detect_reward_model_type(adapter_dir) == "lora"


def test_build_export_recipe_defaults(tmp_path) -> None:
    recipe = build_export_recipe(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_dir=tmp_path / "adapter",
        export_dir=tmp_path / "merged",
        template="qwen",
    )
    assert recipe["finetuning_type"] == "lora"
    assert recipe["template"] == "qwen"
