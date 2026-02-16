"""Keyframe selection based on embedding cosine distance."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KeyframeSelector:
    """Selects keyframes when cosine distance exceeds a threshold."""

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold
        self._last_embedding: Optional[np.ndarray] = None
        logger.info("KeyframeSelector initialized with threshold=%.3f", threshold)

    def is_keyframe(self, embedding: np.ndarray) -> tuple[bool, np.ndarray]:
        """Determine if the current embedding represents a keyframe.

        Args:
            embedding: Normalized 512-dim embedding vector.

        Returns:
            Tuple of (is_keyframe, embedding).
        """
        if self._last_embedding is None:
            self._last_embedding = embedding
            logger.debug("First frame accepted as keyframe")
            return True, embedding

        similarity = float(np.dot(self._last_embedding, embedding))
        distance = 1.0 - similarity

        if distance >= self.threshold:
            self._last_embedding = embedding
            logger.debug("Keyframe selected (distance=%.4f)", distance)
            return True, embedding

        return False, embedding

    def reset(self) -> None:
        """Reset the selector state."""
        self._last_embedding = None
