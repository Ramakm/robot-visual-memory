"""Qdrant vector database client wrapper."""

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

logger = logging.getLogger(__name__)


class QdrantMemoryClient:
    """Wrapper around Qdrant for managing visual memory collections."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "robot_visual_memory",
        vector_size: int = 512,
        quantization_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size

        logger.info("Connecting to Qdrant at %s:%d", host, port)
        self.client = QdrantClient(host=host, port=port)

        self._quantization = None
        if quantization_config and "scalar" in quantization_config:
            scalar_cfg = quantization_config["scalar"]
            self._quantization = ScalarQuantizationConfig(
                type=ScalarType(scalar_cfg.get("type", "int8")),
                always_ram=scalar_cfg.get("always_ram", True),
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info("Creating collection '%s'", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                quantization_config=self._quantization,
            )
            self._create_payload_indexes()
        else:
            logger.info("Collection '%s' already exists", self.collection_name)

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes = {
            "room_id": PayloadSchemaType.KEYWORD,
            "timestamp": PayloadSchemaType.FLOAT,
        }
        for field, schema_type in indexes.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema_type,
            )
            logger.debug("Created index on field '%s'", field)

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
