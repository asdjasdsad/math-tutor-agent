#!/usr/bin/env bash
set -euo pipefail

accelerate launch -m src.trainers.train_reward --config configs/reward.yaml --paths-config configs/paths.yaml "$@"
