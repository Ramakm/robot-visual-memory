"""Scene retrieval from visual memory."""

import logging
from typing import Optional

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload, RetrievalResult

logger = logging.getLogger(__name__)


class SceneRetriever:
    """Retrieves similar scenes from visual memory with optional filtering."""

    def __init__(
        self,
        client: QdrantMemoryClient,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> None:
        self.client = client
        self.top_k = top_k
        self.score_threshold = score_threshold

    def query(
        self,
        embedding: np.ndarray,
        room_id: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Query visual memory for similar scenes.

        Args:
            embedding: 512-dim query embedding.
            room_id: Optional room filter.
            time_start: Optional timestamp lower bound.
            time_end: Optional timestamp upper bound.
            top_k: Override default top_k.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        conditions = []

        if room_id is not None:
            conditions.append(
                FieldCondition(key="room_id", match=MatchValue(value=room_id))
            )
        if time_start is not None or time_end is not None:
            range_params = {}
            if time_start is not None:
                range_params["gte"] = time_start
            if time_end is not None:
                range_params["lte"] = time_end
            conditions.append(
                FieldCondition(key="timestamp", range=Range(**range_params))
            )

        query_filter = Filter(must=conditions) if conditions else None
        k = top_k or self.top_k

        results = self.client.client.query_points(
            collection_name=self.client.collection_name,
            query=embedding.tolist(),
            query_filter=query_filter,
            limit=k,
            score_threshold=self.score_threshold,
        )

        return [
            RetrievalResult(
                point_id=str(point.id),
                score=point.score,
                payload=MemoryPayload(**point.payload),
            )
            for point in results.points
        ]
