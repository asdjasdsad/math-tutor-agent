# 项目名称
Math Tutor Agent: 基于 Qwen2.5-1.5B 的数学解题辅导 Agent（SFT + RLAIF + PPO + GRPO + 推理优化）

# 一、项目目标

构建一个“数学解题辅导 / 作业讲评 Agent”，具备以下能力：

1. 能进行数学题的分步讲解与答案生成
2. 支持 SFT 微调
3. 支持 Reward Model 训练（RLAIF）
4. 支持 PPO baseline
5. 支持 GRPO 正式强化学习
6. 能对比 PPO 和 GRPO 的训练效果与资源消耗
7. 支持推理部署与推理性能优化
8. 提供完整的训练脚本、评测脚本、部署脚本、配置文件、README

这个项目是一个**单卡 5090（32GB）可落地运行**的 PoC，不追求最大模型，而追求“整条链路跑通 + 工程完整”。

# 二、硬件与约束

- GPU: NVIDIA RTX 5090 32GB
- OS: Linux Ubuntu 22.04
- Python: 3.10
- CUDA: 12.8（按较新环境兼容）
- 单卡训练，不依赖多机多卡
- 所有训练脚本都必须尽量考虑显存占用
- 默认使用 LoRA / QLoRA 风格的参数高效训练
- 所有训练阶段都要支持 checkpoint、resume、日志保存
- 所有路径都必须通过 config / yaml / CLI 参数控制，不要把路径写死在代码里

# 三、技术选型（必须按这个来）

## 1. 基座模型
- `Qwen/Qwen2.5-1.5B-Instruct`

## 2. 训练框架
- `transformers`
- `trl`
- `peft`
- `accelerate`
- `bitsandbytes`
- `datasets`

## 3. 奖励模型
- `Qwen/Qwen3-0.6B` 或同级别小模型，用于 reward model / RLAIF

## 4. 推理部署
- `vllm`

## 5. 实验追踪
- 优先支持 `tensorboard`
- 可选支持 `wandb`，但默认不开启

## 6. 配置管理
- 用 `yaml` + `dataclass` 或 `pydantic`
- 提供 `configs/` 目录

# 四、业务场景定义

项目场景是“数学解题辅导 Agent”。

用户输入一道数学题，系统输出：

1. 题意理解
2. 解题思路
3. 步骤推导
4. 最终答案
5. 可选：简短总结 / 易错点提醒

这个 Agent 在推理时可以是单轮问答模式，不要求真的联网检索，但代码架构上要预留 `tools/` 和 `agents/` 目录，让后续可以扩展成：
- 题目分类工具
- 公式检索工具
- 错题本工具
- 学生画像工具

# 五、数据集要求

## 1. SFT 数据
主数据集：
- `open-r1/OpenR1-Math-220k`，优先使用 `default` 子集

可选扩展数据：
- `AI-MO/NuminaMath-1.5`

## 2. RL prompt 数据
- `openai/gsm8k`
- 可从 OpenR1 / NuminaMath 中抽样 harder subset

## 3. RLAIF / Reward 数据
- `trl-lib/ultrafeedback_binarized` 或兼容 UltraFeedback 风格数据

## 4. 数据处理要求
实现统一的数据处理模块，能够把不同来源的数据转换成项目内部统一格式：

```json
{
  "prompt": "...",
  "response": "...",
  "answer": "...",
  "metadata": {}
}
```

对于 SFT，再转换成 chat / instruction 格式。  
对于 RL，转换成 prompt-only 格式。  
对于 reward model，转换成 chosen / rejected 格式或 score 格式。

# 六、训练阶段设计

项目必须有 4 个训练阶段，每个阶段都要有独立脚本和配置。

## 阶段 A：SFT
目标：
- 将基座模型微调成“会分步讲解数学题”的教师风格模型

要求：
- 支持 LoRA / QLoRA
- 支持 bf16
- 支持 resume
- 支持保存 checkpoint
- 输出 adapter 与 merge 脚本
- 训练脚本命名为：
  - `train_sft.py`

## 阶段 B：Reward Model（RLAIF）
目标：
- 训练一个奖励模型，评估回答是否：
  - 清晰
  - 步骤完整
  - 教学风格自然
  - 结论明确

要求：
- 使用 RewardTrainer
- 支持 chosen/rejected 或 score 风格数据
- 训练脚本命名为：
  - `train_reward.py`

## 阶段 C：PPO baseline
目标：
- 基于 SFT 模型继续做 RL，对比 PPO 和 GRPO

要求：
- 使用 PPOTrainer
- 明确实现 reward 组合逻辑：
  - 数学正确性奖励
  - RLAIF 奖励
  - 格式奖励
- 训练脚本命名为：
  - `train_ppo.py`

## 阶段 D：GRPO 正式版
目标：
- 与 PPO 在相同 prompt 集、相同奖励函数下进行对照
- 使用 GRPOTrainer
- 训练脚本命名为：
  - `train_grpo.py`

# 七、奖励函数设计（非常重要）

需要在 `src/rewards/` 下实现可组合奖励函数模块。

至少实现以下奖励：

## 1. correctness_reward
- 自动抽取模型输出中的最终答案
- 支持常见数学答案匹配：
  - 精确字符串匹配
  - 数值归一化后匹配
  - boxed answer 提取
- 返回 `[0, 1]` 或实数奖励

## 2. format_reward
判断输出是否符合指定模板：
- 是否包含“思路”
- 是否包含“步骤”
- 是否包含“答案”

可设计成简单规则奖励。

## 3. rlaif_reward
- 调用训练好的 reward model 给生成结果打分

## 4. combined_reward
加权合成：

```python
reward = w1 * correctness_reward + w2 * rlaif_reward + w3 * format_reward
```

要求：
- 权重可配置
- reward 计算代码要清晰可测
- 每个 reward 单独有单元测试

# 八、PPO vs GRPO 对比要求

项目必须内置一个可重复运行的 A/B 对比实验模块。

对比项至少包括：

## 质量指标
- GSM8K exact match
- MATH-500 exact match（如果评测脚本可实现）
- final answer accuracy
- format pass rate

## 效率指标
- peak GPU memory
- tokens/sec
- samples/sec
- wall-clock time per N updates

## 日志要求
- 每轮训练保存 jsonl / csv 指标日志
- 能画出 PPO vs GRPO 曲线图
- 生成一个 markdown 报告：
  - `reports/ppo_vs_grpo_report.md`

# 九、推理性能优化要求

项目必须支持以下推理优化，并提供脚本或配置示例。

## 1. vLLM 部署
实现：
- `serve_vllm.py`
- OpenAI-compatible API 或简单 FastAPI 封装

## 2. Automatic Prefix Caching
- 在部署配置中预留开关

## 3. FP8 KV Cache
- 在部署配置中预留开关
- 写清楚注意事项

## 4. AWQ 量化推理
- 提供量化脚本或占位实现
- 文件命名：
  - `quantize_awq.py`

## 5. Speculative Decoding
- 提供可选配置
- 小 draft model 默认写为：
  - `Qwen/Qwen2.5-0.5B-Instruct`
- 需要有说明：如果不启用则回退到普通生成

# 十、代码仓库结构要求

请生成如下风格的目录结构：

```text
math-tutor-agent/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── configs/
│   ├── sft.yaml
│   ├── reward.yaml
│   ├── ppo.yaml
│   ├── grpo.yaml
│   ├── inference.yaml
│   └── paths.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── scripts/
│   ├── prepare_sft_data.sh
│   ├── prepare_reward_data.sh
│   ├── run_sft.sh
│   ├── run_reward.sh
│   ├── run_ppo.sh
│   ├── run_grpo.sh
│   ├── eval_all.sh
│   └── serve.sh
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── preprocess_sft.py
│   │   ├── preprocess_reward.py
│   │   ├── preprocess_rl.py
│   │   └── schemas.py
│   ├── models/
│   │   ├── model_utils.py
│   │   ├── lora_utils.py
│   │   ├── reward_model.py
│   │   └── merge_lora.py
│   ├── rewards/
│   │   ├── correctness.py
│   │   ├── format_reward.py
│   │   ├── rlaif_reward.py
│   │   └── combined.py
│   ├── trainers/
│   │   ├── train_sft.py
│   │   ├── train_reward.py
│   │   ├── train_ppo.py
│   │   └── train_grpo.py
│   ├── eval/
│   │   ├── eval_gsm8k.py
│   │   ├── eval_math500.py
│   │   ├── compare_rl.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── serve_vllm.py
│   │   ├── client.py
│   │   ├── quantize_awq.py
│   │   └── benchmark.py
│   ├── agents/
│   │   ├── tutor_agent.py
│   │   ├── prompts.py
│   │   └── routing.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── seed.py
│   │   ├── profiling.py
│   │   └── gpu_monitor.py
│   └── main.py
├── tests/
│   ├── test_rewards.py
│   ├── test_data_pipeline.py
│   ├── test_answer_extraction.py
│   └── test_configs.py
└── reports/
```

# 十一、实现要求

## 1. 代码风格
- Python 代码必须可运行
- 使用类型标注
- 关键函数必须有 docstring
- 错误处理要清晰
- 不要生成明显无法运行的占位符

## 2. 可配置性
- 所有训练参数通过 yaml + CLI 覆盖
- 不要把 batch size、lr、路径写死在脚本里

## 3. 可复现性
- 固定 random seed
- 保存训练配置副本
- 保存 git hash（如果拿不到就降级处理）

## 4. 单卡适配
- 默认配置必须适合单张 5090
- 默认 batch 不要太大
- 要考虑显存安全
- 尽量支持 gradient checkpointing、bf16、LoRA

## 5. 输出
- 所有训练阶段都要把日志输出到：
  - `outputs/<stage_name>/`
- 保存：
  - `config_snapshot.yaml`
  - `metrics.jsonl`
  - `trainer_state.json`（如可用）
  - `checkpoints/`

# 十二、README 要求

请生成一份高质量 README，必须包含：

1. 项目简介
2. 场景说明
3. 环境安装
4. 数据准备
5. SFT 训练命令
6. Reward Model 训练命令
7. PPO 训练命令
8. GRPO 训练命令
9. PPO vs GRPO 对比方法
10. 推理部署方法
11. 推理性能优化开关说明
12. 常见报错排查
13. 单卡 5090 调参建议

# 十三、默认训练策略（请按此实现）

## SFT 默认策略
- LoRA
- bf16
- max_length=2048
- 单卡安全 batch
- 保存 checkpoint

## Reward Model 默认策略
- 小模型
- 先跑通
- 支持 score 或 chosen/rejected

## PPO 默认策略
- 作为 baseline，小预算即可
- 重点是对比流程和指标，不追求极限结果

## GRPO 默认策略
- 作为正式 RL 版本
- 实现 group-based generation 和 reward aggregation

# 十四、你生成代码时的要求

1. 先保证**项目结构完整**
2. 再保证**SFT 跑通**
3. 再补 **Reward / PPO / GRPO**
4. 最后补 **vLLM 推理与性能优化**
5. 代码中需要给出必要注释，说明每个模块是干什么的
6. 如果某个高级功能（如 AWQ 量化或 speculative decoding）依赖环境较复杂，可以给出“可运行的占位实现 + README 中清晰说明”

# 十五、验收标准

请确保最终生成的项目满足下面这些验收点：

- `train_sft.py` 能在单卡上跑
- `train_reward.py` 能启动训练
- `train_ppo.py` 能跑 baseline
- `train_grpo.py` 能跑正式版
- 有 `eval_gsm8k.py`
- 有 PPO vs GRPO 对比报告生成逻辑
- 有 `serve_vllm.py`
- 有 `quantize_awq.py`
- 有完整 README
- 有 tests
- 代码结构清晰，不是把所有逻辑塞进一个文件

# 十六、输出方式

请直接输出完整项目代码。
如果一次输出不完，请按“文件路径 + 文件内容”的形式分批输出，保证我可以逐步保存到本地。
先从：
1. `README.md`
2. `requirements.txt`
3. `configs/*`
4. `src/data/*`
5. `src/trainers/train_sft.py`
开始输出。
