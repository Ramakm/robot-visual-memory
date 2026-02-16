"""Visual memory storage and management."""

import logging
import uuid
from typing import Optional

import numpy as np
from qdrant_client.models import PointStruct

from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload

logger = logging.getLogger(__name__)


class VisualMemory:
    """Manages storage and batched insertion of visual embeddings into Qdrant."""

    def __init__(
        self,
        client: QdrantMemoryClient,
        batch_size: int = 64,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self._buffer: list[PointStruct] = []

    def store(
        self,
        embedding: np.ndarray,
        payload: MemoryPayload,
        point_id: Optional[str] = None,
    ) -> str:
        """Store a single embedding with metadata.

        Args:
            embedding: 512-dim normalized embedding.
            payload: Metadata to attach.
            point_id: Optional UUID string. Generated if not provided.

        Returns:
            The point ID used for storage.
        """
        if point_id is None:
            point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload.model_dump(),
        )
        self._buffer.append(point)

        if len(self._buffer) >= self.batch_size:
            self.flush()

        return point_id

    def flush(self) -> int:
        """Flush the internal buffer to Qdrant.

        Returns:
            Number of points flushed.
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)
        self.client.client.upsert(
            collection_name=self.client.collection_name,
            points=self._buffer,
        )
        logger.info("Flushed %d points to Qdrant", count)
        self._buffer.clear()
        return count

    def store_immediate(
        self,
        embedding: np.ndarray,
        payload: MemoryPayload,
        point_id: Optional[str] = None,
    ) -> str:
        """Store a single embedding immediately without buffering.

        Args:
            embedding: 512-dim normalized embedding.
            payload: Metadata to attach.
            point_id: Optional UUID string.

        Returns:
            The point ID used for storage.
        """
        if point_id is None:
            point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload.model_dump(),
        )
        self.client.client.upsert(
            collection_name=self.client.collection_name,
            points=[point],
        )
        return point_id

    @property
    def buffer_size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)
