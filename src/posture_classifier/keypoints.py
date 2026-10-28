"""COCO-17 keypoint definitions used by the pose model."""

from __future__ import annotations

from dataclasses import dataclass

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_COCO_KEYPOINTS = len(COCO_KEYPOINT_NAMES)
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}

# Limb connections for drawing a skeleton (pairs of keypoint indices).
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)  # fmt: skip


@dataclass(frozen=True)
class BodySide:
    """Keypoint indices of the shoulder-hip-knee chain on one side of the body.

    The hip is the angle vertex, so the angle measures hip flexion: roughly
    180 degrees when standing upright and roughly 90 degrees when seated.
    """

    name: str
    shoulder: int
    hip: int
    knee: int

    @property
    def indices(self) -> tuple[int, int, int]:
        """Return ``(shoulder, hip, knee)`` indices; the hip is the vertex."""
        return (self.shoulder, self.hip, self.knee)


RIGHT_SIDE = BodySide(
    "right",
    KEYPOINT_INDEX["right_shoulder"],
    KEYPOINT_INDEX["right_hip"],
    KEYPOINT_INDEX["right_knee"],
)
LEFT_SIDE = BodySide(
    "left",
    KEYPOINT_INDEX["left_shoulder"],
    KEYPOINT_INDEX["left_hip"],
    KEYPOINT_INDEX["left_knee"],
)
BODY_SIDES: dict[str, BodySide] = {"right": RIGHT_SIDE, "left": LEFT_SIDE}
