"""Pose utilities for robot position handling."""

import math

from src.memory.schemas import Pose


def euclidean_distance(a: Pose, b: Pose) -> float:
    """Compute 2D Euclidean distance between two poses."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def angular_distance(a: Pose, b: Pose) -> float:
    """Compute smallest angular distance between two poses in radians."""
    diff = abs(a.theta - b.theta) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)


def pose_to_payload_fields(pose: Pose) -> dict[str, float]:
    """Convert a Pose to payload field dict."""
    return {"pose_x": pose.x, "pose_y": pose.y, "pose_theta": pose.theta}
