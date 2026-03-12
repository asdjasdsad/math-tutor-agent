"""Profiling helpers for wall-clock and throughput reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Stopwatch:
    """Simple timer used for RL loops and evaluation."""

    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return time.perf_counter() - self.start_time


def safe_rate(numerator: float, denominator: float) -> float:
    """Return `numerator / denominator` while guarding division by zero."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator
