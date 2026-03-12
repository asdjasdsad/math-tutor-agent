#!/usr/bin/env bash
set -euo pipefail

python -m src.data.preprocess_sft --config configs/sft.yaml --paths-config configs/paths.yaml "$@"
