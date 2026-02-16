"""Navigation decision controller based on visual memory retrieval."""

import logging
from typing import Optional

from src.memory.schemas import (
    NavigationAction,
    NavigationDecision,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

CONFIDENT_MATCH = 0.85
PARTIAL_MATCH = 0.75


class NavigationController:
    """Makes navigation decisions based on scene retrieval results.

    Does not depend on ROS. Produces pure data decisions.
    """

    def __init__(
        self,
        confident_threshold: float = CONFIDENT_MATCH,
        partial_threshold: float = PARTIAL_MATCH,
    ) -> None:
        self.confident_threshold = confident_threshold
        self.partial_threshold = partial_threshold

    def decide(self, results: list[RetrievalResult]) -> NavigationDecision:
        """Produce a navigation decision from retrieval results.

        Args:
            results: Sorted retrieval results from SceneRetriever.

        Returns:
            NavigationDecision with action, confidence, and metadata.
        """
        if not results:
            logger.info("No matches — EXPLORE_AND_MAP")
            return NavigationDecision(
                action=NavigationAction.EXPLORE_AND_MAP,
                confidence=0.0,
                top_score=0.0,
                num_matches=0,
            )

        top_score = results[0].score
        room_id = results[0].payload.room_id
        num_matches = len(results)

        if top_score >= self.confident_threshold:
            action = NavigationAction.LOCALIZE
            logger.info("Confident match (%.3f) — LOCALIZE", top_score)
        elif top_score >= self.partial_threshold:
            action = NavigationAction.CAUTIOUS_NAVIGATE
            logger.info("Partial match (%.3f) — CAUTIOUS_NAVIGATE", top_score)
        else:
            action = NavigationAction.EXPLORE_AND_MAP
            logger.info("Low match (%.3f) — EXPLORE_AND_MAP", top_score)

        return NavigationDecision(
            action=action,
            confidence=top_score,
            top_score=top_score,
            num_matches=num_matches,
            room_id=room_id,
        )
