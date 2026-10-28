"""Draw boxes, skeletons and posture labels with OpenCV."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .config import VisualizationConfig
from .keypoints import BODY_SIDES, COCO_SKELETON
from .pipeline import PersonPose
from .posture import Posture

# BGR colours.
POSTURE_COLORS: dict[Posture, tuple[int, int, int]] = {
    Posture.STANDING: (36, 255, 12),
    Posture.SITTING: (0, 165, 255),
    Posture.LYING: (255, 128, 0),
    Posture.UNKNOWN: (160, 160, 160),
}
SKELETON_COLOR = (255, 255, 255)
KEYPOINT_COLOR = (0, 0, 255)
ANGLE_COLOR = (78, 255, 201)  # colour of the measured limb chain, as in the original notebook


def _pt(xy: np.ndarray) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def draw_person(
    image: np.ndarray, person: PersonPose, cfg: VisualizationConfig | None = None
) -> np.ndarray:
    """Draw one person's box, skeleton, measured angle chain and label in place.

    Args:
        image: BGR image to draw on (modified in place).
        person: Pipeline output for one person.
        cfg: Drawing settings.

    Returns:
        The same ``image`` for convenience.
    """
    cfg = cfg or VisualizationConfig()
    thickness = cfg.line_thickness
    visible = person.keypoint_scores >= cfg.keypoint_score_threshold
    color = POSTURE_COLORS[person.posture.posture]

    x1, y1, x2, y2 = (int(round(float(v))) for v in person.bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    for i, j in COCO_SKELETON:
        if visible[i] and visible[j]:
            cv2.line(
                image,
                _pt(person.keypoints[i]),
                _pt(person.keypoints[j]),
                SKELETON_COLOR,
                thickness,
                cv2.LINE_AA,
            )
    for idx in np.flatnonzero(visible):
        cv2.circle(image, _pt(person.keypoints[idx]), thickness + 2, KEYPOINT_COLOR, -1)

    if person.posture.side is not None:
        shoulder, hip, knee = (
            person.keypoints[i] for i in BODY_SIDES[person.posture.side].indices
        )
        cv2.line(image, _pt(shoulder), _pt(hip), ANGLE_COLOR, thickness + 1, cv2.LINE_AA)
        cv2.line(image, _pt(hip), _pt(knee), ANGLE_COLOR, thickness + 1, cv2.LINE_AA)

    label = person.posture.posture.value
    if person.posture.hip_angle is not None:
        label += f" ({person.posture.hip_angle:.0f} deg)"
    (_, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 2)
    # Keep the label inside the image even when the box touches the top edge.
    text_y = y1 - 8 if y1 - 8 - text_h >= 0 else y1 + text_h + baseline + 4
    cv2.putText(
        image,
        label,
        (max(x1, 0), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def draw_results(
    image: np.ndarray, people: Iterable[PersonPose], cfg: VisualizationConfig | None = None
) -> np.ndarray:
    """Return a copy of ``image`` annotated with every person's results."""
    canvas = image.copy()
    for person in people:
        draw_person(canvas, person, cfg)
    return canvas
