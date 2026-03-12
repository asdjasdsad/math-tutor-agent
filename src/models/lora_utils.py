"""LoRA helpers shared across training stages."""

from __future__ import annotations


def build_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    bias: str = "none",
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
):
    """Create a PEFT LoRA config."""

    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=task_type,
    )
