#!/usr/bin/env bash
set -euo pipefail

python -m src.trainers.train_grpo --config configs/grpo.yaml --paths-config configs/paths.yaml "$@"
