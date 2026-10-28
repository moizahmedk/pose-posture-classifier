"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .posture import PostureRules


@dataclass(frozen=True)
class ModelConfig:
    """An OpenMMLab model: config file plus checkpoint (local path or URL)."""

    config: str
    checkpoint: str


@dataclass(frozen=True)
class DetectorConfig(ModelConfig):
    """Person detector settings.

    Attributes:
        person_label: Class index of "person". MMDetection 3.x uses
            0-based COCO labels, so this is ``0`` (MMPose 0.x used ``1``).
        score_threshold: Minimum detection confidence.
        nms_threshold: IoU threshold for suppressing duplicate person boxes.
    """

    person_label: int = 0
    score_threshold: float = 0.3
    nms_threshold: float = 0.3

    def __post_init__(self) -> None:
        for name in ("score_threshold", "nms_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"detector.{name} must be in [0, 1].")


@dataclass(frozen=True)
class VisualizationConfig:
    """Drawing settings for annotated output images."""

    keypoint_score_threshold: float = 0.3
    line_thickness: int = 2
    font_scale: float = 0.7


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    detector: DetectorConfig
    pose_model: ModelConfig
    device: str = "auto"
    posture: PostureRules = field(default_factory=PostureRules)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _build(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:  # missing required keys
        raise ValueError(f"Invalid '{section}' section: {exc}") from exc


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Build an :class:`AppConfig` from a plain dictionary (e.g. parsed YAML)."""
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    unknown = set(data) - {f.name for f in fields(AppConfig)}
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}")
    for section in ("detector", "pose_model"):
        if section not in data:
            raise ValueError(f"Missing required config section '{section}'.")
    return AppConfig(
        detector=_build(DetectorConfig, data.get("detector"), "detector"),
        pose_model=_build(ModelConfig, data.get("pose_model"), "pose_model"),
        device=str(data.get("device", "auto")),
        posture=_build(PostureRules, data.get("posture"), "posture"),
        visualization=_build(VisualizationConfig, data.get("visualization"), "visualization"),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file content is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data)
