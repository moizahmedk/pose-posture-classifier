"""Rule-based posture classification from a person box and COCO keypoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .geometry import bbox_size, interior_angle
from .keypoints import BODY_SIDES, NUM_COCO_KEYPOINTS, BodySide


class Posture(str, Enum):
    """Posture labels produced by :func:`classify_posture`."""

    STANDING = "Standing"
    SITTING = "Sitting"
    LYING = "Lying"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class PostureRules:
    """Thresholds for the posture rules.

    Attributes:
        standing_min_angle: Hip angle (shoulder-hip-knee, degrees) above which
            an upright person counts as standing.
        min_keypoint_score: Minimum confidence each of the three keypoints
            must have for an angle to be trusted.
        side: ``"right"``, ``"left"`` or ``"auto"`` (use the side whose
            three keypoints are most confident).
    """

    standing_min_angle: float = 165.0
    min_keypoint_score: float = 0.3
    side: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.standing_min_angle <= 180.0:
            raise ValueError("standing_min_angle must be in (0, 180].")
        if not 0.0 <= self.min_keypoint_score <= 1.0:
            raise ValueError("min_keypoint_score must be in [0, 1].")
        if self.side not in {"auto", *BODY_SIDES}:
            raise ValueError(f"side must be 'auto', 'left' or 'right', got {self.side!r}.")


@dataclass(frozen=True)
class PostureResult:
    """Outcome of classifying one person.

    Attributes:
        posture: Predicted label.
        hip_angle: Interior shoulder-hip-knee angle in degrees, if computed.
        side: Name of the body side used for the angle, if any.
        reason: Short human-readable explanation of the decision.
    """

    posture: Posture
    hip_angle: float | None = None
    side: str | None = None
    reason: str = ""


def select_side(scores: np.ndarray, rules: PostureRules) -> BodySide | None:
    """Pick the body side to measure, or ``None`` if no side is reliable.

    Args:
        scores: Keypoint confidences with shape ``(17,)``.
        rules: Posture rules (uses ``side`` and ``min_keypoint_score``).

    Returns:
        The chosen :class:`BodySide`, or ``None`` when every candidate side
        has at least one keypoint below ``min_keypoint_score``.
    """
    candidates = list(BODY_SIDES.values()) if rules.side == "auto" else [BODY_SIDES[rules.side]]
    reliable = [
        side for side in candidates if scores[list(side.indices)].min() >= rules.min_keypoint_score
    ]
    if not reliable:
        return None
    return max(reliable, key=lambda side: float(scores[list(side.indices)].mean()))


def classify_posture(
    bbox: Sequence[float],
    keypoints: np.ndarray,
    scores: np.ndarray,
    rules: PostureRules | None = None,
) -> PostureResult:
    """Classify a person as standing, sitting, lying or unknown.

    Decision rules (in order):

    1. If the person box is at least as wide as it is tall -> ``Lying``.
    2. If no body side has confident shoulder, hip and knee -> ``Unknown``.
    3. If the interior shoulder-hip-knee angle exceeds
       ``standing_min_angle`` -> ``Standing``, otherwise ``Sitting``.

    Args:
        bbox: Person box ``(x1, y1, x2, y2[, score])`` in pixels.
        keypoints: Keypoint coordinates with shape ``(17, 2)``.
        scores: Keypoint confidences with shape ``(17,)``.
        rules: Thresholds; defaults to :class:`PostureRules()`.

    Returns:
        A :class:`PostureResult`.

    Raises:
        ValueError: If inputs have unexpected shapes or the box is degenerate.
    """
    rules = rules or PostureRules()
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if keypoints.shape != (NUM_COCO_KEYPOINTS, 2):
        raise ValueError(f"keypoints must have shape (17, 2), got {keypoints.shape}.")
    if scores.shape != (NUM_COCO_KEYPOINTS,):
        raise ValueError(f"scores must have shape (17,), got {scores.shape}.")

    width, height = bbox_size(bbox)
    if height <= width:
        return PostureResult(Posture.LYING, reason="box is at least as wide as it is tall")

    side = select_side(scores, rules)
    if side is None:
        return PostureResult(Posture.UNKNOWN, reason="shoulder/hip/knee not confidently detected")

    shoulder, hip, knee = (keypoints[i] for i in side.indices)
    try:
        angle = interior_angle(shoulder, hip, knee)
    except ValueError:
        return PostureResult(Posture.UNKNOWN, side=side.name, reason="degenerate keypoints")

    if angle > rules.standing_min_angle:
        posture = Posture.STANDING
        reason = f"hip angle {angle:.1f} > {rules.standing_min_angle:g}"
    else:
        posture = Posture.SITTING
        reason = f"hip angle {angle:.1f} <= {rules.standing_min_angle:g}"
    return PostureResult(posture, hip_angle=angle, side=side.name, reason=reason)
