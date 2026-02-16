"""Memory compression by pruning old, redundant frames."""

import logging
import time

from qdrant_client.models import FieldCondition, Filter, Range

from src.memory.qdrant_client import QdrantMemoryClient

logger = logging.getLogger(__name__)


class MemoryCompressor:
    """Compresses visual memory by deleting every Nth frame for old memories."""

    def __init__(
        self,
        client: QdrantMemoryClient,
        keep_every_nth: int = 3,
        age_threshold_hours: float = 24.0,
    ) -> None:
        self.client = client
        self.keep_every_nth = keep_every_nth
        self.age_threshold_hours = age_threshold_hours

    def compress(self) -> int:
        """Run compression on memories older than the age threshold.

        Returns:
            Number of points deleted.
        """
        cutoff = time.time() - (self.age_threshold_hours * 3600)
        logger.info(
            "Compressing memories older than %.1f hours (cutoff=%.0f)",
            self.age_threshold_hours,
            cutoff,
        )

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="timestamp",
                    range=Range(lt=cutoff),
                )
            ]
        )

        ids_to_delete: list[str] = []
        offset = None

        while True:
            results, next_offset = self.client.client.scroll(
                collection_name=self.client.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
            )

            for i, point in enumerate(results):
                if i % self.keep_every_nth != 0:
                    ids_to_delete.append(point.id)

            if next_offset is None:
                break
            offset = next_offset

        if ids_to_delete:
            self.client.client.delete(
                collection_name=self.client.collection_name,
                points_selector=ids_to_delete,
            )
            logger.info("Compressed: deleted %d points", len(ids_to_delete))

        return len(ids_to_delete)
