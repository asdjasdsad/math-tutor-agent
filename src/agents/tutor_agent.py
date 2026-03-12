"""Math tutor agent wrapper for local or remote inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.agents.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class TutorAgent:
    """Serve math answers from either a local model or a remote API."""

    model_name_or_path: str
    api_base: str | None = None
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _ensure_local_model(self) -> None:
        if self.api_base is not None or self._model is not None:
            return
        from src.models.model_utils import get_model_device, load_causal_lm, load_tokenizer

        self._tokenizer = load_tokenizer(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        self._model = load_causal_lm(
            model_name_or_path=self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_4bit=False,
            compute_dtype="bfloat16",
            use_cache=True,
        )
        self._device = get_model_device(self._model)
        self._model.eval()

    def answer(self, question: str, max_new_tokens: int = 512) -> str:
        """Generate a tutor-style answer."""

        if self.api_base is not None:
            payload = {
                "model": "math-tutor-agent",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question)},
                ],
                "max_tokens": max_new_tokens,
            }
            response = requests.post(
                f"{self.api_base.rstrip('/')}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"]

        self._ensure_local_model()
        import torch

        prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(question)}"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
