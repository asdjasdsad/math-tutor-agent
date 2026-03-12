#!/usr/bin/env bash
set -euo pipefail

python -m src.trainers.train_ppo --config configs/ppo.yaml --paths-config configs/paths.yaml "$@"
