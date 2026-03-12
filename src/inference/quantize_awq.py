"""AWQ quantization utility.

This script is runnable when `autoawq` is installed. If the package is absent,
it exits with a clear instruction instead of silently pretending to quantize.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import load_stage_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a model with AWQ.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_calibration_texts(processed_sft_path: Path, limit: int) -> list[str]:
    """Load calibration prompts from processed SFT data when available."""

    if not processed_sft_path.exists():
        return [
            "题目：若 2x + 3 = 11，求 x。",
            "题目：已知三角形三边分别为 3、4、5，求面积。",
        ]
    import json

    rows = []
    with processed_sft_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            payload = json.loads(line)
            rows.append(payload["prompt"])
    return rows


def main() -> None:
    args = parse_args()
    config = load_stage_config(args.config, args.paths_config)
    project_root = config["paths"]["project_root"]
    server_cfg = config["server"]
    awq_cfg = config["awq"]

    model_path = args.model_path or server_cfg["model_name_or_path"]
    output_dir = args.output_dir or awq_cfg["output_dir"]
    resolved_model_path = resolve_path(model_path, project_root)
    resolved_output_dir = resolve_path(output_dir, project_root)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "AWQ quantization requires the `autoawq` package. Install it separately, then rerun this script."
        ) from exc

    model_name = str(resolved_model_path) if resolved_model_path.exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=server_cfg["trust_remote_code"])
    model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=server_cfg["trust_remote_code"])

    calibration_texts = load_calibration_texts(
        resolve_path("data/processed/sft_train.jsonl", project_root),
        limit=int(awq_cfg["calibration_samples"]),
    )
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "gemm"}
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_texts,
        max_calib_seq_len=int(awq_cfg["max_calibration_length"]),
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)
    print(f"AWQ model saved to {resolved_output_dir}")


if __name__ == "__main__":
    main()
