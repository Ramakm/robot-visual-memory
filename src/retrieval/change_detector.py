"""Scene change detection using exponential moving average of similarity."""

import logging

from src.memory.schemas import ChangeResult

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Detects scene changes by comparing current similarity against an EMA baseline."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        change_threshold: float = 0.3,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.change_threshold = change_threshold
        self._baseline: float = 1.0

    def update(self, similarity_scores: list[float]) -> ChangeResult:
        """Update the baseline and detect change.

        Args:
            similarity_scores: List of similarity scores from retrieval.

        Returns:
            ChangeResult with change detection info.
        """
        if not similarity_scores:
            return ChangeResult(
                changed=True,
                current_similarity=0.0,
                baseline_similarity=self._baseline,
                delta=self._baseline,
            )

        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        delta = self._baseline - avg_similarity

        changed = delta >= self.change_threshold

        self._baseline = (
            self.ema_alpha * avg_similarity + (1 - self.ema_alpha) * self._baseline
        )

        if changed:
            logger.info(
                "Scene change detected: avg=%.3f baseline=%.3f delta=%.3f",
                avg_similarity,
                self._baseline,
                delta,
            )

        return ChangeResult(
            changed=changed,
            current_similarity=avg_similarity,
            baseline_similarity=self._baseline,
            delta=delta,
        )

    def reset(self) -> None:
        """Reset the baseline."""
        self._baseline = 1.0
