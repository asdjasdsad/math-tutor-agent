# Math Tutor Agent

Math Tutor Agent 是一个围绕 `Qwen/Qwen2.5-1.5B-Instruct` 构建的单卡数学辅导 PoC，覆盖从 SFT、Reward Model、PPO、GRPO 到 vLLM 部署与推理优化的完整链路。项目目标不是追求最大模型，而是在 RTX 5090 32GB 上把整条训练与部署工程跑通，并保留足够清晰的扩展结构。

## 1. 项目简介

项目提供以下能力：

- 基于数学题输入生成分步讲解、关键思路、详细推导与最终答案
- 用 `trl + transformers + peft + accelerate` 完成 SFT、Reward、PPO、GRPO 训练
- 用统一奖励模块组合正确性奖励、格式奖励与 RLAIF 奖励
- 用统一评测与对比模块输出 PPO vs GRPO 的质量、吞吐、显存和报告
- 用 `vllm` 提供推理服务，并预留 Prefix Caching、FP8 KV Cache、AWQ、Speculative Decoding 开关

## 2. 场景说明

目标场景是“数学解题辅导 / 作业讲评 Agent”。输入一道题，模型应输出：

1. 题意理解
2. 解题思路
3. 步骤推导
4. 最终答案
5. 可选总结或易错点提醒

`src/agents/` 和 `src/tools/` 已预留扩展位，后续可以接入题目分类、公式检索、错题本、学生画像等能力。

## 3. 环境安装

推荐环境：

- Ubuntu 22.04
- Python 3.10
- CUDA 12.8
- NVIDIA RTX 5090 32GB

创建环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你打算做 vLLM 推理或 AWQ 量化，建议确认 CUDA、PyTorch、vLLM 版本兼容。

## 4. 数据准备

项目默认从 Hugging Face 下载数据集，并将处理后的数据写入 `data/processed/`。

### SFT 数据

- 主数据集：`open-r1/OpenR1-Math-220k`
- 可选扩展：`AI-MO/NuminaMath-1.5`

准备命令：

```bash
bash scripts/prepare_sft_data.sh
```

或手动执行：

```bash
python -m src.data.preprocess_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

### Reward 数据

- 默认数据集：`trl-lib/ultrafeedback_binarized`

准备命令：

```bash
bash scripts/prepare_reward_data.sh
```

或手动执行：

```bash
python -m src.data.preprocess_reward --config configs/reward.yaml --paths-config configs/paths.yaml
```

### RL Prompt 数据

PPO 和 GRPO 默认使用 `openai/gsm8k`，也支持切换到本地预处理后的 prompt-only 数据：

```bash
python -m src.data.preprocess_rl --config configs/ppo.yaml --paths-config configs/paths.yaml
```

## 5. SFT 训练命令

默认策略：

- LoRA / QLoRA
- bf16
- `max_length=2048`
- 单卡安全 batch
- 保存 checkpoint 与 adapter

运行：

```bash
bash scripts/run_sft.sh
```

或：

```bash
accelerate launch -m src.trainers.train_sft --config configs/sft.yaml --paths-config configs/paths.yaml
```

输出目录默认在 `outputs/sft/`，包括：

- `config_snapshot.yaml`
- `git_state.json`
- `metrics.jsonl`
- `trainer_state.json`
- `checkpoints/`
- `adapter/`
- `merged/`（可选自动 merge）

## 6. Reward Model 训练命令

Reward 模型默认使用 `Qwen/Qwen3-0.6B`，训练目标是对清晰度、步骤完整性、教学风格与结论明确性打分。

运行：

```bash
bash scripts/run_reward.sh
```

或：

```bash
accelerate launch -m src.trainers.train_reward --config configs/reward.yaml --paths-config configs/paths.yaml
```

Reward 数据既支持原生 `chosen/rejected`，也支持 `score` 风格数据；如果输入是 score，预处理模块会先转成 pairwise 样本再喂给 `RewardTrainer`。

## 7. PPO 训练命令

PPO 部分是 baseline，重点是打通 RL 流程和可比指标，而不是追求极限分数。

运行：

```bash
bash scripts/run_ppo.sh
```

或：

```bash
accelerate launch -m src.trainers.train_ppo --config configs/ppo.yaml --paths-config configs/paths.yaml
```

PPO 奖励由以下三部分组合：

- `correctness_reward`
- `rlaif_reward`
- `format_reward`

权重通过 `configs/ppo.yaml` 配置。

## 8. GRPO 训练命令

GRPO 是正式 RL 版本，默认启用 group-based generation 和 reward aggregation。

运行：

```bash
bash scripts/run_grpo.sh
```

或：

```bash
accelerate launch -m src.trainers.train_grpo --config configs/grpo.yaml --paths-config configs/paths.yaml
```

## 9. PPO vs GRPO 对比方法

训练完成后，运行统一评测和报告生成脚本：

```bash
bash scripts/eval_all.sh
```

也可以分别执行：

```bash
python -m src.eval.eval_gsm8k --model-path outputs/ppo/policy
python -m src.eval.eval_gsm8k --model-path outputs/grpo/policy
python -m src.eval.compare_rl --ppo-metrics outputs/ppo/metrics.jsonl --grpo-metrics outputs/grpo/metrics.jsonl --report-path reports/ppo_vs_grpo_report.md
```

报告会输出：

- GSM8K / MATH-500 / final answer accuracy
- format pass rate
- tokens/sec
- samples/sec
- peak GPU memory
- wall-clock time
- Markdown 报告与 PNG 曲线图

## 10. 推理部署方法

### vLLM 服务

运行：

```bash
bash scripts/serve.sh
```

或：

```bash
python -m src.inference.serve_vllm --config configs/inference.yaml --paths-config configs/paths.yaml
```

服务默认暴露两个接口：

- `POST /generate`
- `POST /v1/chat/completions`

### 本地客户端

```bash
python -m src.inference.client --question "解方程 2x + 3 = 11"
```

## 11. 推理性能优化开关说明

`configs/inference.yaml` 中预留了以下优化项：

- `enable_prefix_caching`: 打开 vLLM Automatic Prefix Caching
- `kv_cache_dtype`: 可切换到 `fp8`，但需要确认当前 GPU / vLLM 版本支持
- `enable_speculative_decoding`: 启用后尝试加载 `Qwen/Qwen2.5-0.5B-Instruct` 作为 draft model
- `draft_model_name`: Speculative Decoding 的小模型
- `awq.enabled`: AWQ 量化开关

如果 speculative decoding 未启用，服务会回退到普通生成路径。

## 12. 常见报错排查

### 显存不足

- 降低 `per_device_train_batch_size`
- 提高 `gradient_accumulation_steps`
- 打开 `model.use_4bit`
- 保持 `gradient_checkpointing=true`
- 缩短 `max_seq_length`

### `bitsandbytes` / CUDA 不兼容

- 确认 PyTorch 与 CUDA 版本匹配
- 确认 `bitsandbytes` 已正确安装
- 先关闭 `use_4bit` 验证基础链路

### Reward / RL 训练启动失败

- 检查 `trl` 版本是否支持 `RewardTrainer`、`PPOTrainer`、`GRPOTrainer`
- 检查 `configs/reward.yaml`、`configs/ppo.yaml`、`configs/grpo.yaml` 的模型路径是否存在
- 如果使用 adapter 路径，先执行 merge 或确认脚本支持直接加载 PEFT

### vLLM 启动失败

- 确认 GPU 驱动、CUDA、vLLM 版本匹配
- 先关闭 `enable_speculative_decoding`
- 先使用原始 bf16 模型排查，再切到量化模型

## 13. 单卡 5090 调参建议

- SFT 默认从 LoRA + 4bit 开始，不要直接全参微调
- `max_seq_length=2048` 是安全起点，长上下文请同步降 batch
- Reward 模型先跑小样本验证格式和 loss 曲线
- PPO 作为 baseline，先减少 rollout 数和 update 次数
- GRPO 可以重点调 `num_generations`、`group_size` 和 reward 权重
- 如果日志吞吐异常，优先检查 `generation_max_new_tokens` 和 `dataset` 过滤长度

## 14. 目录说明

```text
configs/      训练、推理和路径配置
scripts/      数据准备、训练、评测、部署命令包装
src/data/     统一数据读取与预处理
src/models/   模型加载、LoRA、奖励模型、adapter merge
src/rewards/  可组合奖励函数
src/trainers/ 四阶段训练入口
src/eval/     评测、指标和 RL 对比报告
src/inference/ 推理服务、量化、压测和客户端
src/agents/   数学辅导 Agent 封装
src/tools/    未来扩展工具位
tests/        单元测试
```

## 15. 快速开始

```bash
bash scripts/prepare_sft_data.sh
bash scripts/run_sft.sh
bash scripts/run_reward.sh
bash scripts/run_ppo.sh
bash scripts/run_grpo.sh
bash scripts/eval_all.sh
bash scripts/serve.sh
```

如果需要直接做单轮问答，也可以：

```bash
python -m src.main --question "若 x^2 - 5x + 6 = 0，求 x 的值"
```
