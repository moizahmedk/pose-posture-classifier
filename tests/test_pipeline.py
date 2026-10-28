from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from posture_classifier.config import load_config
from posture_classifier.pipeline import PosturePipeline, nms, select_person_boxes
from posture_classifier.posture import Posture

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class FakePipeline(PosturePipeline):
    """Pipeline with model calls replaced by fixed outputs (no MMLab needed)."""

    def __init__(self, boxes, keypoints, scores):
        super().__init__(None, None, load_config(DEFAULT_CONFIG))
        self._boxes, self._kpts, self._scores = boxes, keypoints, scores
        self.received_boxes = None

    def detect_people(self, image):
        return self._boxes

    def estimate_keypoints(self, image, boxes):
        self.received_boxes = boxes
        return self._kpts[: len(boxes)], self._scores[: len(boxes)]


def test_nms_suppresses_overlaps_and_keeps_separate_boxes():
    boxes = np.array(
        [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], dtype=float
    )
    assert nms(boxes, 0.3).tolist() == [0, 2]
    assert nms(boxes, 0.9).tolist() == [0, 1, 2]
    assert nms(np.empty((0, 5)), 0.3).size == 0


def test_select_person_boxes_filters_label_and_score():
    cfg = load_config(DEFAULT_CONFIG).detector
    bboxes = np.array([[0, 0, 10, 20], [30, 0, 40, 20], [60, 0, 70, 20]], dtype=float)
    scores = np.array([0.9, 0.2, 0.95])
    labels = np.array([0, 0, 2])  # person, low-score person, car
    kept = select_person_boxes(bboxes, scores, labels, cfg)
    assert kept.shape == (1, 5)
    assert kept[0].tolist() == [0, 0, 10, 20, 0.9]


def test_pipeline_classifies_each_person(standing_person, sitting_person):
    boxes = np.array([standing_person[0], sitting_person[0]])
    kpts = np.stack([standing_person[1], sitting_person[1]])
    scores = np.stack([standing_person[2], sitting_person[2]])
    people = FakePipeline(boxes, kpts, scores)(np.zeros((400, 400, 3), np.uint8))
    assert [p.posture.posture for p in people] == [Posture.STANDING, Posture.SITTING]


def test_manual_boxes_skip_detector_and_get_unit_score(standing_person):
    _, kpts, scores = standing_person
    pipe = FakePipeline(np.empty((0, 5)), kpts[None], scores[None])
    people = pipe(np.zeros((400, 400, 3), np.uint8), boxes=[[100, 50, 160, 350]])
    assert pipe.received_boxes.shape == (1, 5)
    assert people[0].bbox[4] == 1.0


def test_no_people_returns_empty_list():
    pipe = FakePipeline(np.empty((0, 5)), np.empty((0, 17, 2)), np.empty((0, 17)))
    assert pipe(np.zeros((10, 10, 3), np.uint8)) == []
    assert pipe(np.zeros((10, 10, 3), np.uint8), boxes=[]) == []


def test_invalid_image_rejected():
    pipe = FakePipeline(np.empty((0, 5)), None, None)
    with pytest.raises(ValueError, match="BGR"):
        pipe(np.zeros((10, 10), np.uint8))


def test_from_config_reports_missing_model_files(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    cfg = replace(cfg, detector=replace(cfg.detector, config=str(tmp_path / "missing.py")))
    with pytest.raises(FileNotFoundError, match="download_models.sh"):
        PosturePipeline.from_config(cfg)
