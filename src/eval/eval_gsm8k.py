"""Evaluate a local model on GSM8K exact-match and format metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data.loaders import load_hf_dataset
from src.data.preprocess_rl import build_rl_prompt
from src.eval.metrics import final_answer_accuracy, format_pass_rate
from src.rewards.correctness import extract_final_answer
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K.")
    parser.add_argument("--model-path", required=True, help="Merged model or policy path")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-path", default="outputs/eval/gsm8k_results.jsonl")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from src.models.model_utils import get_model_device, load_causal_lm, load_tokenizer

    tokenizer = load_tokenizer(args.model_path, trust_remote_code=args.trust_remote_code)
    model = load_causal_lm(
        model_name_or_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_4bit=False,
        compute_dtype="bfloat16",
        use_cache=True,
    )
    model.eval()
    device = get_model_device(model)

    dataset = load_hf_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        cache_dir="data/cache",
    )
    rows = []
    for index, example in enumerate(tqdm(dataset, total=args.max_samples, desc="gsm8k-eval")):
        if index >= args.max_samples:
            break
        question = example.get("question", "")
        answer = example.get("answer", "")
        prompt = build_rl_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        decoded = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        rows.append(
            {
                "question": question,
                "reference": answer,
                "prediction": decoded,
                "predicted_answer": extract_final_answer(decoded),
            }
        )

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "samples": len(rows),
        "final_answer_accuracy": final_answer_accuracy(
            [row["prediction"] for row in rows],
            [row["reference"] for row in rows],
        ),
        "format_pass_rate": format_pass_rate([row["prediction"] for row in rows]),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
