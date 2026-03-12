#!/usr/bin/env bash
set -euo pipefail

python -m src.trainers.train_sft --config configs/sft.yaml --paths-config configs/paths.yaml "$@"
