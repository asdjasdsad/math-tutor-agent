"""Simple HTTP benchmark for the inference server."""

from __future__ import annotations

import argparse
import statistics
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the deployed inference server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="解方程 2x + 3 = 11")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latencies = []
    token_counts = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        response = requests.post(
            f"{args.base_url.rstrip('/')}/generate",
            json={"prompt": args.prompt, "max_tokens": args.max_tokens},
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["text"]
        latency = time.perf_counter() - start
        latencies.append(latency)
        token_counts.append(len(text.split()))

    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(token_counts)
    print(
        {
            "repeats": args.repeats,
            "avg_latency_sec": avg_latency,
            "p50_latency_sec": statistics.median(latencies),
            "approx_tokens_per_sec": avg_tokens / avg_latency if avg_latency else 0.0,
        }
    )


if __name__ == "__main__":
    main()
