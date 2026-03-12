"""Model loading utilities for causal LM and reward models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_model_name(model_name_or_path: str, fallback_model_name: str | None = None) -> str:
    """Use a local path when present, otherwise fall back to a configured model name."""

    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)
    return fallback_model_name or model_name_or_path


def get_torch_dtype(dtype_name: str | None):
    """Map string dtype names to torch dtypes."""

    import torch

    if dtype_name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def build_quantization_config(
    use_4bit: bool,
    compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
):
    """Create a BitsAndBytes quantization config when requested."""

    if not use_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_torch_dtype(compute_dtype),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    """Load a tokenizer and ensure `pad_token` is always set."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    use_4bit: bool = False,
    compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = False,
    use_cache: bool = False,
):
    """Load an AutoModelForCausalLM with optional 4-bit quantization."""

    from transformers import AutoModelForCausalLM

    quantization_config = build_quantization_config(
        use_4bit=use_4bit,
        compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=get_torch_dtype(compute_dtype) if not use_4bit else None,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    return model


def load_sequence_classification_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    use_4bit: bool = False,
    compute_dtype: str = "bfloat16",
    num_labels: int = 1,
):
    """Load a sequence-classification reward model."""

    from transformers import AutoModelForSequenceClassification

    quantization_config = build_quantization_config(
        use_4bit=use_4bit,
        compute_dtype=compute_dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=get_torch_dtype(compute_dtype) if not use_4bit else None,
        num_labels=num_labels,
        device_map="auto",
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def get_model_device(model) -> str:
    """Best-effort device detection for HF and TRL models."""

    if hasattr(model, "device"):
        return str(model.device)
    if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "device"):
        return str(model.pretrained_model.device)
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"
