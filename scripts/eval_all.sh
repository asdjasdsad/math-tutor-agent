#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.eval_gsm8k --model-path outputs/ppo/policy --output-path outputs/eval/ppo_gsm8k.jsonl
python -m src.eval.eval_gsm8k --model-path outputs/grpo/policy --output-path outputs/eval/grpo_gsm8k.jsonl
python -m src.eval.compare_rl --ppo-metrics outputs/ppo/metrics.jsonl --grpo-metrics outputs/grpo/metrics.jsonl --report-path reports/ppo_vs_grpo_report.md
