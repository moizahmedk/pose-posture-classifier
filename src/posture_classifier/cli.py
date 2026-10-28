"""Command-line interface: ``posture-classify``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from .config import load_config
from .pipeline import PersonPose, PosturePipeline
from .visualization import draw_results

logger = logging.getLogger("posture_classifier")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(path: Path) -> list[Path]:
    """Return image files at ``path`` (a single file or a directory).

    Raises:
        FileNotFoundError: If ``path`` does not exist or holds no images.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
        raise FileNotFoundError(f"No images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {path}")
    raise FileNotFoundError(f"Input not found: {path}")


def person_to_dict(person: PersonPose) -> dict[str, Any]:
    """Serialize a :class:`PersonPose` to JSON-friendly types."""
    return {
        "bbox_xyxy": [round(float(v), 2) for v in person.bbox[:4]],
        "bbox_score": round(float(person.bbox[4]), 4),
        "posture": person.posture.posture.value,
        "hip_angle_deg": None
        if person.posture.hip_angle is None
        else round(person.posture.hip_angle, 2),
        "side": person.posture.side,
        "reason": person.posture.reason,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="posture-classify",
        description="Detect people, estimate 2D pose and classify standing/sitting/lying.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs"))
    parser.add_argument("--device", help="Override config device, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Use this person box instead of the detector (repeatable).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        images = collect_images(args.input)
        pipeline = PosturePipeline.from_config(config)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict[str, Any]]] = {}
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Skipping unreadable image: %s", image_path)
            continue
        people = pipeline(image, boxes=args.bbox)
        out_path = args.output_dir / f"{image_path.stem}_posture.jpg"
        cv2.imwrite(str(out_path), draw_results(image, people, config.visualization))
        summary[image_path.name] = [person_to_dict(p) for p in people]
        labels = ", ".join(p.posture.posture.value for p in people) or "no person"
        logger.info("%s -> %s (saved %s)", image_path.name, labels, out_path)

    summary_path = args.output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
