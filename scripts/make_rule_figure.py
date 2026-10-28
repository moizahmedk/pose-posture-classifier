"""Render docs/posture_rules.png: the three posture rules on hand-made skeletons.

The skeletons are synthetic (no model involved). They exist only to illustrate
how the rules and the visualization work. Run from the repository root:

    python scripts/make_rule_figure.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from posture_classifier.pipeline import classify_people
from posture_classifier.posture import PostureRules
from posture_classifier.visualization import draw_results

# (x, y) per COCO keypoint, in a 200x300 local frame.
STANDING = [
    (100, 40), (95, 35), (105, 35), (88, 40), (112, 40),          # head
    (80, 75), (120, 75), (72, 125), (128, 125), (68, 170), (132, 170),  # arms
    (88, 160), (112, 160), (88, 220), (112, 220), (88, 280), (112, 280),  # legs
]  # fmt: skip
SITTING = [
    (90, 60), (85, 55), (95, 55), (78, 60), (102, 60),
    (75, 95), (110, 95), (78, 140), (115, 140), (100, 165), (135, 160),
    (82, 180), (105, 180), (150, 182), (172, 182), (150, 260), (172, 260),
]  # fmt: skip
LYING = [
    (30, 150), (25, 145), (25, 155), (30, 140), (30, 160),
    (65, 135), (65, 165), (105, 130), (105, 170), (140, 128), (140, 172),
    (140, 140), (140, 160), (200, 140), (200, 160), (260, 140), (260, 160),
]  # fmt: skip

PANELS = [
    ("standing", STANDING, (50, 20, 155, 295)),
    ("sitting", SITTING, (60, 40, 185, 270)),
    ("lying", LYING, (15, 115, 275, 185)),
]


def main(out_path: Path = Path("docs/posture_rules.png")) -> None:
    """Render the figure and write it to ``out_path``."""
    panels = []
    for _, points, box in PANELS:
        kpts = np.asarray(points, dtype=float)
        scores = np.full(len(kpts), 0.9)
        bbox = np.array([[*box, 1.0]])
        canvas = np.full((300, 300, 3), 32, np.uint8)
        people = classify_people(bbox, kpts[None], scores[None], PostureRules())
        panels.append(draw_results(canvas, people))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack(panels))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
