"""Pydantic models for structured data across the system."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Pose(BaseModel):
    """Robot pose in 2D space."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class MemoryPayload(BaseModel):
    """Metadata payload stored alongside each embedding in Qdrant."""

    timestamp: float
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_theta: float = 0.0
    room_id: str = "unknown"
    depth_mean: Optional[float] = None


class NavigationAction(str, Enum):
    """Possible navigation actions based on scene recognition."""

    LOCALIZE = "LOCALIZE"
    CAUTIOUS_NAVIGATE = "CAUTIOUS_NAVIGATE"
    EXPLORE_AND_MAP = "EXPLORE_AND_MAP"


class NavigationDecision(BaseModel):
    """Structured navigation decision output."""

    action: NavigationAction
    confidence: float
    top_score: float
    num_matches: int
    room_id: Optional[str] = None


class ChangeResult(BaseModel):
    """Result from the change detection module."""

    changed: bool
    current_similarity: float
    baseline_similarity: float
    delta: float


class RetrievalResult(BaseModel):
    """A single retrieval result from memory."""

    point_id: str
    score: float
    payload: MemoryPayload
