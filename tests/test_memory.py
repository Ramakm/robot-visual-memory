"""Tests for visual memory and related components."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload, Pose
from src.memory.visual_memory import VisualMemory


@pytest.fixture
def mock_qdrant():
    """Create a mocked QdrantMemoryClient."""
    with patch("src.memory.qdrant_client.QdrantClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        client = QdrantMemoryClient(
            host="localhost",
            port=6333,
            collection_name="test_collection",
        )
        return client


@pytest.fixture
def visual_memory(mock_qdrant):
    """Create a VisualMemory with mocked Qdrant."""
    return VisualMemory(client=mock_qdrant, batch_size=3)


def test_store_returns_id(visual_memory):
    """Storing an embedding should return a UUID string."""
    emb = np.random.randn(512).astype(np.float32)
    payload = MemoryPayload(timestamp=1000.0, room_id="lab")
    pid = visual_memory.store(emb, payload)
    assert isinstance(pid, str)
    assert len(pid) > 0


def test_flush_empties_buffer(visual_memory):
    """Flushing should clear the internal buffer."""
    emb = np.random.randn(512).astype(np.float32)
    payload = MemoryPayload(timestamp=1000.0)

    visual_memory.store(emb, payload)
    assert visual_memory.buffer_size == 1

    visual_memory.flush()
    assert visual_memory.buffer_size == 0


def test_auto_flush_on_batch_size(visual_memory):
    """Buffer should auto-flush when batch_size is reached."""
    for i in range(3):
        emb = np.random.randn(512).astype(np.float32)
        payload = MemoryPayload(timestamp=float(i))
        visual_memory.store(emb, payload)

    # After 3 stores (batch_size=3), buffer should have been flushed
    assert visual_memory.buffer_size == 0


def test_memory_payload_defaults():
    """MemoryPayload should have sensible defaults."""
    p = MemoryPayload(timestamp=123.0)
    assert p.pose_x == 0.0
    assert p.room_id == "unknown"
    assert p.depth_mean is None


def test_pose_model():
    """Pose model should serialize correctly."""
    pose = Pose(x=1.5, y=2.5, theta=0.785)
    d = pose.model_dump()
    assert d == {"x": 1.5, "y": 2.5, "theta": 0.785}
