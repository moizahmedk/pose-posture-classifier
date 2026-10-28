from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from posture_classifier import cli
from posture_classifier.pipeline import classify_people
from posture_classifier.posture import PostureRules
from posture_classifier.visualization import draw_results


def test_draw_results_returns_annotated_copy(standing_person):
    bbox, kpts, scores = standing_person
    people = classify_people(np.array([bbox]), kpts[None], scores[None], PostureRules())
    image = np.zeros((400, 300, 3), np.uint8)
    out = draw_results(image, people)
    assert out.shape == image.shape
    assert image.sum() == 0, "input image must not be modified"
    assert out.sum() > 0


def test_label_drawn_inside_image_when_box_touches_top(standing_person):
    _, kpts, scores = standing_person
    bbox = np.array([[100.0, 0.0, 160.0, 350.0, 0.9]])
    people = classify_people(bbox, kpts[None], scores[None], PostureRules())
    out = draw_results(np.zeros((400, 300, 3), np.uint8), people)
    assert out[1:40, 100:250].sum() > 0  # text placed just inside the box


def test_collect_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in cli.collect_images(tmp_path)] == ["a.jpg"]
    with pytest.raises(FileNotFoundError):
        cli.collect_images(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No images"):
        cli.collect_images(empty)


def test_cli_missing_config_returns_error_code(tmp_path):
    img = tmp_path / "img.jpg"
    cv2.imwrite(str(img), np.zeros((20, 20, 3), np.uint8))
    assert cli.main(["-i", str(img), "-c", str(tmp_path / "none.yaml")]) == 1


def test_cli_end_to_end_with_stub_pipeline(tmp_path, monkeypatch, standing_person):
    bbox, kpts, scores = standing_person

    class StubPipeline:
        @classmethod
        def from_config(cls, config):
            return cls()

        def __call__(self, image, boxes=None):
            return classify_people(np.array([bbox]), kpts[None], scores[None], PostureRules())

    monkeypatch.setattr(cli, "PosturePipeline", StubPipeline)
    img = tmp_path / "person.png"
    cv2.imwrite(str(img), np.zeros((400, 300, 3), np.uint8))
    out_dir = tmp_path / "out"

    assert cli.main(["-i", str(img), "-o", str(out_dir)]) == 0
    assert (out_dir / "person_posture.jpg").is_file()
    results = json.loads((out_dir / "results.json").read_text())
    assert results["person.png"][0]["posture"] == "Standing"
    assert results["person.png"][0]["hip_angle_deg"] == pytest.approx(180.0)
