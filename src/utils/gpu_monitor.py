"""GPU monitoring helpers with graceful degradation on CPU-only environments."""

from __future__ import annotations

from typing import Any


def reset_peak_memory() -> None:
    """Reset CUDA peak memory stats when CUDA is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_peak_memory_gb() -> float:
    """Return peak CUDA memory in GiB."""

    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        pass
    return 0.0


def snapshot_gpu_state() -> dict[str, Any]:
    """Get a lightweight GPU state snapshot."""

    try:
        import torch

        if torch.cuda.is_available():
            return {
                "device_name": torch.cuda.get_device_name(0),
                "allocated_gb": float(torch.cuda.memory_allocated() / (1024**3)),
                "reserved_gb": float(torch.cuda.memory_reserved() / (1024**3)),
                "peak_allocated_gb": get_peak_memory_gb(),
            }
    except Exception:
        pass
    return {
        "device_name": None,
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "peak_allocated_gb": 0.0,
    }
