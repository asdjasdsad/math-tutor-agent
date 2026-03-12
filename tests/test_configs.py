from src.trainers.train_grpo import GRPOStageConfig
from src.trainers.train_ppo import PPOStageConfig
from src.trainers.train_reward import RewardStageConfig
from src.trainers.train_sft import SFTStageConfig
from src.utils.io import dataclass_from_dict, load_stage_config, load_yaml


def test_all_yaml_files_load() -> None:
    for path in [
        "configs/paths.yaml",
        "configs/sft.yaml",
        "configs/reward.yaml",
        "configs/ppo.yaml",
        "configs/grpo.yaml",
        "configs/inference.yaml",
    ]:
        payload = load_yaml(path)
        assert isinstance(payload, dict)
        assert payload


def test_sft_config_instantiation() -> None:
    config = load_stage_config("configs/sft.yaml", "configs/paths.yaml")
    structured = dataclass_from_dict(SFTStageConfig, config)
    assert structured.model.base_model_name
    assert structured.training.max_seq_length == 2048


def test_reward_config_instantiation() -> None:
    config = load_stage_config("configs/reward.yaml", "configs/paths.yaml")
    structured = dataclass_from_dict(RewardStageConfig, config)
    assert structured.training.model_dir.endswith("model")


def test_ppo_grpo_config_instantiation() -> None:
    ppo_config = dataclass_from_dict(PPOStageConfig, load_stage_config("configs/ppo.yaml", "configs/paths.yaml"))
    grpo_config = dataclass_from_dict(GRPOStageConfig, load_stage_config("configs/grpo.yaml", "configs/paths.yaml"))
    assert ppo_config.training.batch_size > 0
    assert grpo_config.training.num_generations >= 1
