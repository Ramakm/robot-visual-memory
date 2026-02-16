"""Tests for retrieval and change detection."""

import numpy as np
import pytest

from src.memory.schemas import ChangeResult, MemoryPayload, RetrievalResult
from src.retrieval.change_detector import ChangeDetector


class TestChangeDetector:
    def test_first_update_no_scores_is_change(self):
        """Empty scores should indicate a scene change."""
        detector = ChangeDetector(ema_alpha=0.1, change_threshold=0.3)
        result = detector.update([])
        assert result.changed is True
        assert result.current_similarity == 0.0

    def test_high_similarity_no_change(self):
        """High similarity scores should not trigger change."""
        detector = ChangeDetector(ema_alpha=0.1, change_threshold=0.3)
        # Prime the baseline
        detector.update([0.95, 0.92, 0.90])
        result = detector.update([0.93, 0.91, 0.89])
        assert result.changed is False

    def test_sudden_drop_triggers_change(self):
        """A sudden drop in similarity should trigger change detection."""
        detector = ChangeDetector(ema_alpha=0.05, change_threshold=0.3)
        # Warm up with high scores
        for _ in range(10):
            detector.update([0.95, 0.93])
        # Sudden drop
        result = detector.update([0.3, 0.2])
        assert result.changed is True
        assert result.delta > 0.3

    def test_reset_restores_baseline(self):
        """Reset should restore baseline to 1.0."""
        detector = ChangeDetector()
        detector.update([0.5])
        detector.reset()
        assert detector._baseline == 1.0


class TestRetrievalResult:
    def test_retrieval_result_fields(self):
        """RetrievalResult should properly hold all fields."""
        r = RetrievalResult(
            point_id="abc-123",
            score=0.92,
            payload=MemoryPayload(timestamp=100.0, room_id="kitchen"),
        )
        assert r.score == 0.92
        assert r.payload.room_id == "kitchen"

    def test_retrieval_result_with_all_metadata(self):
        """RetrievalResult with full metadata."""
        r = RetrievalResult(
            point_id="xyz",
            score=0.88,
            payload=MemoryPayload(
                timestamp=200.0,
                pose_x=1.0,
                pose_y=2.0,
                pose_theta=0.5,
                room_id="hallway",
                depth_mean=3.5,
            ),
        )
        assert r.payload.depth_mean == 3.5
        assert r.payload.pose_x == 1.0
