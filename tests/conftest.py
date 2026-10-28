from __future__ import annotations

import numpy as np
import pytest

from posture_classifier.keypoints import KEYPOINT_INDEX, NUM_COCO_KEYPOINTS


def make_keypoints(
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    knee: tuple[float, float],
    side: str = "right",
    score: float = 0.9,
    other_score: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (17, 2) keypoint array with the given chain on one side.

    The opposite side gets the same chain shifted 5 px, scored ``other_score``.
    """
    kpts = np.zeros((NUM_COCO_KEYPOINTS, 2))
    scores = np.full(NUM_COCO_KEYPOINTS, 0.9)
    other = "left" if side == "right" else "right"
    for name, xy in (("shoulder", shoulder), ("hip", hip), ("knee", knee)):
        kpts[KEYPOINT_INDEX[f"{side}_{name}"]] = xy
        scores[KEYPOINT_INDEX[f"{side}_{name}"]] = score
        kpts[KEYPOINT_INDEX[f"{other}_{name}"]] = (xy[0] + 5, xy[1])
        scores[KEYPOINT_INDEX[f"{other}_{name}"]] = other_score
    return kpts, scores


@pytest.fixture
def standing_person() -> tuple[list[float], np.ndarray, np.ndarray]:
    bbox = [100.0, 50.0, 160.0, 350.0, 0.99]  # tall box
    kpts, scores = make_keypoints((130, 100), (130, 200), (130, 280))
    return bbox, kpts, scores


@pytest.fixture
def sitting_person() -> tuple[list[float], np.ndarray, np.ndarray]:
    bbox = [100.0, 50.0, 220.0, 300.0, 0.95]  # still taller than wide
    kpts, scores = make_keypoints((130, 100), (130, 200), (210, 200))  # thigh horizontal
    return bbox, kpts, scores
