"""Serve the math tutor model through vLLM with a lightweight FastAPI wrapper."""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.utils.io import load_stage_config, resolve_path


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the math tutor model with vLLM.")
    parser.add_argument("--config", required=True, help="Path to configs/inference.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Path to shared paths config")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override")
    return parser.parse_args()


def flatten_messages(messages: list[Message]) -> str:
    """Convert a chat transcript into a single prompt string."""

    parts = []
    for message in messages:
        parts.append(f"{message.role.upper()}:\n{message.content}")
    return "\n\n".join(parts) + "\n\nASSISTANT:\n"


def create_app(config: dict[str, Any]) -> FastAPI:
    """Build the FastAPI app and initialize the vLLM engine."""

    from vllm import LLM, SamplingParams

    server_cfg = config["server"]
    generation_cfg = config["generation"]
    project_root = config["paths"]["project_root"]
    local_model_path = resolve_path(server_cfg["model_name_or_path"], project_root)
    model_name = str(local_model_path) if local_model_path.exists() else server_cfg["fallback_model_name"]

    engine_kwargs = {
        "model": model_name,
        "tensor_parallel_size": server_cfg["tensor_parallel_size"],
        "dtype": server_cfg["dtype"],
        "gpu_memory_utilization": server_cfg["gpu_memory_utilization"],
        "max_model_len": server_cfg["max_model_len"],
        "trust_remote_code": server_cfg["trust_remote_code"],
        "download_dir": str(resolve_path(server_cfg["download_dir"], project_root)),
        "swap_space": server_cfg["swap_space"],
        "enable_prefix_caching": server_cfg["enable_prefix_caching"],
    }
    if server_cfg.get("kv_cache_dtype") and server_cfg["kv_cache_dtype"] != "auto":
        engine_kwargs["kv_cache_dtype"] = server_cfg["kv_cache_dtype"]
    if server_cfg.get("enable_speculative_decoding"):
        engine_kwargs["speculative_model"] = server_cfg["draft_model_name"]

    try:
        llm = LLM(**engine_kwargs)
    except TypeError:
        engine_kwargs.pop("speculative_model", None)
        llm = LLM(**engine_kwargs)

    app = FastAPI(title="Math Tutor Agent vLLM")

    def _sampling_params(
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        repetition_penalty: float | None,
    ):
        return SamplingParams(
            temperature=generation_cfg["temperature"] if temperature is None else temperature,
            top_p=generation_cfg["top_p"] if top_p is None else top_p,
            max_tokens=generation_cfg["max_tokens"] if max_tokens is None else max_tokens,
            repetition_penalty=(
                generation_cfg["repetition_penalty"]
                if repetition_penalty is None
                else repetition_penalty
            ),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        outputs = llm.generate([request.prompt], _sampling_params(
            request.temperature,
            request.top_p,
            request.max_tokens,
            request.repetition_penalty,
        ))
        text = outputs[0].outputs[0].text
        return {"text": text, "model": model_name}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        prompt = flatten_messages(request.messages)
        outputs = llm.generate([prompt], _sampling_params(
            request.temperature,
            request.top_p,
            request.max_tokens,
            request.repetition_penalty,
        ))
        text = outputs[0].outputs[0].text
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    args = parse_args()
    config = load_stage_config(args.config, args.paths_config, args.override)
    app = create_app(config)
    server_cfg = config["server"]
    uvicorn.run(app, host=server_cfg["host"], port=int(server_cfg["port"]))


if __name__ == "__main__":
    main()
