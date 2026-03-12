#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.serve_vllm --config configs/inference.yaml --paths-config configs/paths.yaml "$@"
