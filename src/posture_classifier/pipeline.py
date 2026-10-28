"""Two-stage top-down pipeline: person detection -> pose estimation -> posture.

The OpenMMLab packages (torch, mmdet, mmpose) are imported lazily inside
:meth:`PosturePipeline.from_config`, so the pure-NumPy helpers in this module
can be imported and tested without a deep-learning stack installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig, DetectorConfig, ModelConfig
from .posture import PostureResult, PostureRules, classify_posture

logger = logging.getLogger(__name__)

DOWNLOAD_HINT = "Run `bash scripts/download_models.sh` or point the config at valid files/URLs."


@dataclass(frozen=True)
class PersonPose:
    """Pose and posture for one detected person.

    Attributes:
        bbox: ``(x1, y1, x2, y2, score)`` in pixels.
        keypoints: ``(17, 2)`` COCO keypoint coordinates in pixels.
        keypoint_scores: ``(17,)`` keypoint confidences.
        posture: Posture classification for this person.
    """

    bbox: np.ndarray
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    posture: PostureResult


def nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression.

    Args:
        boxes: Array of shape ``(N, 5)`` with ``(x1, y1, x2, y2, score)``.
        iou_threshold: Boxes overlapping a kept box by more than this IoU
            are discarded.

    Returns:
        Indices of kept boxes, ordered by descending score.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2, scores = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        inter_w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        inter_h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = inter_w * inter_h
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=int)


def select_person_boxes(
    bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, cfg: DetectorConfig
) -> np.ndarray:
    """Keep confident, non-duplicate person detections.

    Args:
        bboxes: ``(N, 4)`` boxes in ``xyxy`` format.
        scores: ``(N,)`` detection scores.
        labels: ``(N,)`` class indices.
        cfg: Detector settings (person label and thresholds).

    Returns:
        ``(M, 5)`` array of ``(x1, y1, x2, y2, score)`` sorted by score.
    """
    bboxes = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mask = (labels == cfg.person_label) & (scores > cfg.score_threshold)
    candidates = np.concatenate([bboxes[mask], scores[mask, None]], axis=1)
    return candidates[nms(candidates, cfg.nms_threshold)]


def classify_people(
    boxes: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, rules: PostureRules
) -> list[PersonPose]:
    """Combine per-person boxes and keypoints into :class:`PersonPose` records."""
    return [
        PersonPose(box, kpts, kpt_scores, classify_posture(box, kpts, kpt_scores, rules))
        for box, kpts, kpt_scores in zip(boxes, keypoints, scores)
    ]


def _check_model_files(model: ModelConfig, name: str) -> None:
    """Fail early with a helpful message if local model files are missing."""
    if not Path(model.config).is_file():
        raise FileNotFoundError(f"{name} config not found: {model.config}. {DOWNLOAD_HINT}")
    is_url = model.checkpoint.startswith(("http://", "https://"))
    if not is_url and not Path(model.checkpoint).is_file():
        raise FileNotFoundError(
            f"{name} checkpoint not found: {model.checkpoint}. {DOWNLOAD_HINT}"
        )


def resolve_device(device: str) -> str:
    """Translate ``"auto"`` into ``"cuda:0"`` or ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class PosturePipeline:
    """Person detection (MMDetection) + pose estimation (MMPose) + posture rules."""

    def __init__(self, detector: Any, pose_model: Any, config: AppConfig) -> None:
        self.detector = detector
        self.pose_model = pose_model
        self.config = config

    @classmethod
    def from_config(cls, config: AppConfig) -> PosturePipeline:
        """Load both models described by ``config``.

        Raises:
            ImportError: If the OpenMMLab stack is not installed.
            FileNotFoundError: If configured model files are missing.
        """
        _check_model_files(config.detector, "Detector")
        _check_model_files(config.pose_model, "Pose model")
        try:
            from mmdet.apis import init_detector
            from mmpose.apis import init_model
            from mmpose.utils import adapt_mmdet_pipeline
        except ImportError as exc:
            raise ImportError(
                "MMDetection/MMPose are not installed. See README 'Installation'."
            ) from exc

        device = resolve_device(config.device)
        logger.info("Loading models on %s", device)
        detector = init_detector(config.detector.config, config.detector.checkpoint, device=device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        pose_model = init_model(config.pose_model.config, config.pose_model.checkpoint, device)
        return cls(detector, pose_model, config)

    def detect_people(self, image: np.ndarray) -> np.ndarray:
        """Return ``(M, 5)`` person boxes for a BGR image."""
        from mmdet.apis import inference_detector

        instances = inference_detector(self.detector, image).pred_instances.cpu().numpy()
        return select_person_boxes(
            instances.bboxes, instances.scores, instances.labels, self.config.detector
        )

    def estimate_keypoints(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(N, 17, 2)`` keypoints and ``(N, 17)`` scores for given boxes."""
        from mmpose.apis import inference_topdown

        samples = inference_topdown(self.pose_model, image, boxes[:, :4], bbox_format="xyxy")
        keypoints = np.stack([s.pred_instances.keypoints[0] for s in samples])
        scores = np.stack([s.pred_instances.keypoint_scores[0] for s in samples])
        return keypoints, scores

    def __call__(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[PersonPose]:
        """Run the full pipeline on one BGR image.

        Args:
            image: ``(H, W, 3)`` BGR image.
            boxes: Optional ``(N, 4|5)`` person boxes; skips detection when given.

        Returns:
            One :class:`PersonPose` per person (empty if nobody was found).
        """
        if image is None or image.ndim != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3).")
        if boxes is None:
            boxes = self.detect_people(image)
        else:
            boxes = np.atleast_2d(np.asarray(boxes, dtype=float))
            if boxes.size and boxes.shape[1] not in (4, 5):
                raise ValueError("Manual boxes must have shape (N, 4) or (N, 5).")
            if boxes.size and boxes.shape[1] == 4:
                boxes = np.concatenate([boxes, np.ones((len(boxes), 1))], axis=1)
        if boxes.size == 0:
            logger.info("No person found")
            return []
        logger.info("Found %d person box(es)", len(boxes))
        keypoints, scores = self.estimate_keypoints(image, boxes)
        return classify_people(boxes, keypoints, scores, self.config.posture)
