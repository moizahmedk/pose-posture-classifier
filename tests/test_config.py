from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from posture_classifier.config import config_from_dict, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _default_dict() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.detector.person_label == 0
    assert cfg.posture.side == "auto"
    assert cfg.pose_model.checkpoint.endswith(".pth")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_keys_rejected():
    data = _default_dict()
    data["posture"]["standing_angle"] = 170  # typo of standing_min_angle
    with pytest.raises(ValueError, match="standing_angle"):
        config_from_dict(data)
    data = _default_dict()
    data["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        config_from_dict(data)


def test_missing_required_section_rejected():
    data = _default_dict()
    del data["pose_model"]
    with pytest.raises(ValueError, match="pose_model"):
        config_from_dict(data)


def test_invalid_threshold_rejected():
    data = _default_dict()
    data["detector"]["score_threshold"] = 2
    with pytest.raises(ValueError, match="score_threshold"):
        config_from_dict(data)


def test_optional_sections_use_defaults():
    data = _default_dict()
    del data["posture"], data["visualization"], data["device"]
    cfg = config_from_dict(data)
    assert cfg.device == "auto"
    assert cfg.posture.standing_min_angle == 165.0
