#!/usr/bin/env bash
set -euo pipefail

python -m src.data.preprocess_reward --config configs/reward.yaml --paths-config configs/paths.yaml "$@"
