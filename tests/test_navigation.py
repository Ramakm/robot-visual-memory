"""Tests for the navigation controller."""

import pytest

from src.memory.schemas import (
    MemoryPayload,
    NavigationAction,
    NavigationDecision,
    RetrievalResult,
)
from src.navigation.controller import NavigationController


@pytest.fixture
def controller():
    return NavigationController(confident_threshold=0.85, partial_threshold=0.75)


def _make_result(score: float, room: str = "lab") -> RetrievalResult:
    return RetrievalResult(
        point_id="test-id",
        score=score,
        payload=MemoryPayload(timestamp=100.0, room_id=room),
    )


def test_no_results_explore(controller):
    """No matches should trigger EXPLORE_AND_MAP."""
    decision = controller.decide([])
    assert decision.action == NavigationAction.EXPLORE_AND_MAP
    assert decision.num_matches == 0


def test_high_score_localize(controller):
    """A high score should trigger LOCALIZE."""
    results = [_make_result(0.92)]
    decision = controller.decide(results)
    assert decision.action == NavigationAction.LOCALIZE
    assert decision.top_score == 0.92


def test_partial_score_cautious(controller):
    """A partial score should trigger CAUTIOUS_NAVIGATE."""
    results = [_make_result(0.80)]
    decision = controller.decide(results)
    assert decision.action == NavigationAction.CAUTIOUS_NAVIGATE


def test_low_score_explore(controller):
    """A low score should trigger EXPLORE_AND_MAP."""
    results = [_make_result(0.60)]
    decision = controller.decide(results)
    assert decision.action == NavigationAction.EXPLORE_AND_MAP


def test_decision_includes_room(controller):
    """Decision should include the room_id from top result."""
    results = [_make_result(0.90, room="kitchen")]
    decision = controller.decide(results)
    assert decision.room_id == "kitchen"


def test_boundary_confident(controller):
    """Score exactly at confident threshold should trigger LOCALIZE."""
    results = [_make_result(0.85)]
    decision = controller.decide(results)
    assert decision.action == NavigationAction.LOCALIZE


def test_boundary_partial(controller):
    """Score exactly at partial threshold should trigger CAUTIOUS_NAVIGATE."""
    results = [_make_result(0.75)]
    decision = controller.decide(results)
    assert decision.action == NavigationAction.CAUTIOUS_NAVIGATE
