"""Timing utilities for performance measurement."""

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager that logs elapsed time for a block.

    Args:
        label: Description of the timed operation.
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s: %.2f ms", label, elapsed_ms)


class LatencyTracker:
    """Tracks running statistics for operation latencies."""

    def __init__(self) -> None:
        self._samples: list[float] = []

    def record(self, elapsed_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(elapsed_ms)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def mean_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def max_ms(self) -> float:
        return max(self._samples) if self._samples else 0.0

    @property
    def min_ms(self) -> float:
        return min(self._samples) if self._samples else 0.0

    def summary(self) -> dict[str, float]:
        """Return a summary dict of tracked latencies."""
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }
