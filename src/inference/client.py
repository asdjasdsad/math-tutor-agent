"""Minimal client for the vLLM/FastAPI inference server."""

from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a math question to the inference server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--system-prompt",
        default="\u4f60\u662f\u4e00\u540d\u8010\u5fc3\u7684\u6570\u5b66\u8001\u5e08\uff0c\u8bf7\u6309\u601d\u8def\u3001\u6b65\u9aa4\u3001\u7b54\u6848\u7ed3\u6784\u4f5c\u7b54\u3002",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "model": "math-tutor-agent",
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.question},
        ],
        "max_tokens": args.max_tokens,
    }
    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
