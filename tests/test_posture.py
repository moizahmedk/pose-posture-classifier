from __future__ import annotations

import numpy as np
import pytest

from conftest import make_keypoints
from posture_classifier.posture import Posture, PostureRules, classify_posture, select_side

TALL_BOX = [0.0, 0.0, 100.0, 400.0, 0.9]


def test_standing(standing_person):
    result = classify_posture(*standing_person)
    assert result.posture is Posture.STANDING
    assert result.hip_angle == pytest.approx(180.0)


def test_sitting(sitting_person):
    result = classify_posture(*sitting_person)
    assert result.posture is Posture.SITTING
    assert result.hip_angle == pytest.approx(90.0)


def test_wide_box_is_lying_regardless_of_keypoints(standing_person):
    _, kpts, scores = standing_person
    result = classify_posture([0, 0, 300, 100, 0.9], kpts, scores)
    assert result.posture is Posture.LYING
    assert result.hip_angle is None


def test_square_box_counts_as_lying_like_original(standing_person):
    _, kpts, scores = standing_person
    assert classify_posture([0, 0, 100, 100], kpts, scores).posture is Posture.LYING


def test_low_confidence_keypoints_give_unknown():
    kpts, scores = make_keypoints((50, 50), (50, 150), (50, 250), score=0.1, other_score=0.1)
    result = classify_posture(TALL_BOX, kpts, scores)
    assert result.posture is Posture.UNKNOWN


def test_label_is_independent_of_facing_direction():
    """A 162 deg hip angle: the original rule said Sitting facing one way, Standing the other."""
    shoulder, hip = (50.0, 50.0), (50.0, 150.0)
    theta = np.radians(18.0)  # thigh deviates 18 deg from vertical -> interior angle 162
    knee_right = (50 + 100 * np.sin(theta), 150 + 100 * np.cos(theta))
    knee_left = (50 - 100 * np.sin(theta), 150 + 100 * np.cos(theta))
    rules = PostureRules(side="right")
    for knee in (knee_right, knee_left):
        kpts, scores = make_keypoints(shoulder, hip, knee)
        result = classify_posture(TALL_BOX, kpts, scores, rules)
        assert result.posture is Posture.SITTING
        assert result.hip_angle == pytest.approx(162.0)


def test_auto_side_prefers_more_confident_side():
    kpts, scores = make_keypoints((50, 50), (50, 150), (50, 250), score=0.5, other_score=0.95)
    assert select_side(scores, PostureRules(side="auto")).name == "left"
    assert select_side(scores, PostureRules(side="right")).name == "right"


def test_auto_side_falls_back_when_one_side_is_occluded():
    kpts, scores = make_keypoints((50, 50), (50, 150), (50, 250), score=0.05, other_score=0.8)
    result = classify_posture(TALL_BOX, kpts, scores, PostureRules(side="auto"))
    assert result.side == "left"
    assert classify_posture(TALL_BOX, kpts, scores, PostureRules(side="right")).posture is (
        Posture.UNKNOWN
    )


def test_threshold_is_configurable(sitting_person):
    bbox, kpts, scores = sitting_person
    assert classify_posture(bbox, kpts, scores, PostureRules(standing_min_angle=80)).posture is (
        Posture.STANDING
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"standing_min_angle": 0},
        {"standing_min_angle": 200},
        {"min_keypoint_score": 1.5},
        {"side": "middle"},
    ],
)
def test_invalid_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        PostureRules(**kwargs)


def test_invalid_shapes_rejected(standing_person):
    bbox, kpts, scores = standing_person
    with pytest.raises(ValueError, match="keypoints"):
        classify_posture(bbox, kpts[:10], scores)
    with pytest.raises(ValueError, match="scores"):
        classify_posture(bbox, kpts, scores[:10])
