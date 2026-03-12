"""Reward-model scoring utilities used by RL training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RewardModelScorer:
    """Lazy wrapper around a sequence-classification reward model."""

    model_name_or_path: str
    trust_remote_code: bool = True
    use_4bit: bool = False
    compute_dtype: str = "bfloat16"
    max_length: int = 2048

    def __post_init__(self) -> None:
        from src.models.model_utils import load_sequence_classification_model, load_tokenizer

        self.tokenizer = load_tokenizer(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = load_sequence_classification_model(
            model_name_or_path=self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_4bit=self.use_4bit,
            compute_dtype=self.compute_dtype,
            num_labels=1,
        )
        self.model.eval()

    def score_batch(self, prompts: Iterable[str], responses: Iterable[str]) -> list[float]:
        """Score a batch of prompt/response pairs."""

        import torch

        texts = [f"问题：{prompt}\n\n回答：{response}" for prompt, response in zip(prompts, responses)]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
        return [float(x) for x in logits.detach().cpu().numpy().astype(np.float32)]

    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt/response pair."""

        return self.score_batch([prompt], [response])[0]
