"""Merge a LoRA adapter back into the base model."""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    trust_remote_code: bool = True,
) -> str:
    """Merge a PEFT adapter into the base model and save the merged weights."""

    from peft import PeftModel

    from src.models.model_utils import load_causal_lm, load_tokenizer

    base_model = load_causal_lm(
        model_name_or_path=base_model_name,
        trust_remote_code=trust_remote_code,
        use_4bit=False,
        compute_dtype="bfloat16",
        use_cache=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer = load_tokenizer(base_model_name, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_path)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = merge_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Merged model saved to {path}")


if __name__ == "__main__":
    main()
