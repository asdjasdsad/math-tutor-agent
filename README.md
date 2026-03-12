# Math Tutor Agent

单卡数学辅导 Agent 项目，当前训练主链路已经切到 `LLaMA-Factory`：

- `SFT` 使用 `llamafactory-cli train` 的 `sft` stage
- `Reward Model` 使用 `rm` stage
- `PPO` 使用 `ppo` stage
- 数据预处理、评测、推理服务仍保留在本仓库

`GRPO` 入口目前仍保留为兼容模式，原因是本仓库这次重构只把官方文档中稳定公开的 `SFT / RM / PPO` 流程切到 LLaMA-Factory。

## 环境

不再推荐 `venv`。默认使用 `conda`：

```bash
conda env create -f environment.yml
conda activate math-tutor-agent
```

如果你不想手动 `activate`，也可以直接：

```powershell
conda run -n math-tutor-agent python -m src.trainers.train_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

仓库还提供了一个 PowerShell 包装脚本：

```powershell
.\scripts\run_in_conda.ps1 src.trainers.train_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

`environment.yml` 会安装项目依赖，并从官方仓库安装 `LLaMA-Factory`。

## 数据流

本仓库继续负责把原始数据规整成项目内部 JSONL，然后自动生成一份给 LLaMA-Factory 使用的本地数据目录：

- 处理后的项目数据默认写入 `data/processed/`
- LLaMA-Factory 数据默认写入 `data/llamafactory/`
- 数据集注册文件为 `data/llamafactory/dataset_info.json`

你不需要手工维护 `dataset_info.json`，训练入口会自动生成或更新。

## 训练

### 1. 准备 SFT 数据

```bash
python -m src.data.preprocess_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

### 2. 训练 SFT

```bash
python -m src.trainers.train_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

训练入口会做三件事：

1. 确保 `data/processed/sft_*.jsonl` 存在
2. 生成 `data/llamafactory/sft_*.jsonl` 和 `dataset_info.json`
3. 生成 `outputs/sft/llamafactory_train.yaml` 并调用 `llamafactory-cli train`

如果 `configs/sft.yaml` 里打开了 `training.auto_merge_after_training`，还会额外生成 `outputs/sft/llamafactory_export.yaml` 并调用 `llamafactory-cli export`。

### 3. 准备 Reward 数据

```bash
python -m src.data.preprocess_reward --config configs/reward.yaml --paths-config configs/paths.yaml
```

### 4. 训练 Reward Model

```bash
python -m src.trainers.train_reward --config configs/reward.yaml --paths-config configs/paths.yaml
```

### 5. 准备 RL Prompt 数据

```bash
python -m src.data.preprocess_rl --config configs/ppo.yaml --paths-config configs/paths.yaml
```

### 6. 训练 PPO

```bash
python -m src.trainers.train_ppo --config configs/ppo.yaml --paths-config configs/paths.yaml
```

说明：

- PPO 现在使用 LLaMA-Factory 的 reward-model 路径做打分
- 仓库里的 `correctness / format / RLAIF` 奖励逻辑仍保留，但主要用于离线评测和分析，不再作为 PPO 主训练回路

### 7. 兼容 GRPO

```bash
python -m src.trainers.train_grpo --config configs/grpo.yaml --paths-config configs/paths.yaml
```

这一条仍是兼容入口，没有切到 LLaMA-Factory。

## 推理与评测

本地单轮问答：

```bash
python -m src.main --question "若 x^2 - 5x + 6 = 0，求 x 的值。"
```

启动 vLLM 服务：

```bash
python -m src.inference.serve_vllm --config configs/inference.yaml --paths-config configs/paths.yaml
```

统一评测：

```bash
bash scripts/eval_all.sh
```

## 关键配置

`configs/sft.yaml`、`configs/reward.yaml`、`configs/ppo.yaml` 现在都包含一个 `llamafactory` 段：

```yaml
llamafactory:
  cli_path: llamafactory-cli
  dataset_dir: data/llamafactory
  template: qwen
  train_args: {}
  export_args: {}
```

用途如下：

- `cli_path`: LLaMA-Factory CLI 名称或绝对路径
- `dataset_dir`: 自动生成的数据集目录
- `template`: chat template；不填时会根据模型名推断
- `train_args`: 透传给生成后的训练 recipe，用来覆盖默认参数
- `export_args`: 透传给导出 recipe

## 脚本

保留了原有 shell 脚本，但它们现在只是 Python 入口的薄包装：

- `scripts/run_sft.sh`
- `scripts/run_reward.sh`
- `scripts/run_ppo.sh`
- `scripts/run_grpo.sh`

Windows 用户推荐直接使用 `conda run` 或 `scripts/run_in_conda.ps1`。
