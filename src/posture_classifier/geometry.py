"""Small, dependency-light geometry helpers for 2D keypoints and boxes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = Sequence[float]

_EPS = 1e-6


def directional_angle(a: Point, vertex: Point, c: Point) -> float:
    """Return the directional angle from ray ``vertex->a`` to ray ``vertex->c``.

    This is the angle measure used by the original notebook
    (``measureangle``), converted to degrees. The result lies in
    ``[0, 360)`` and depends on point order, so the same posture yields
    ``x`` or ``360 - x`` depending on which way the person faces. It is kept
    for reference and regression testing; classification uses
    :func:`interior_angle`.

    Args:
        a: First endpoint ``(x, y)``.
        vertex: Angle vertex ``(x, y)``.
        c: Second endpoint ``(x, y)``.

    Returns:
        Angle in degrees in ``[0, 360)``.
    """
    angle_a = math.atan2(a[1] - vertex[1], a[0] - vertex[0]) % (2 * math.pi)
    angle_c = math.atan2(c[1] - vertex[1], c[0] - vertex[0]) % (2 * math.pi)
    return math.degrees((angle_c - angle_a) % (2 * math.pi))


def interior_angle(a: Point, vertex: Point, c: Point) -> float:
    """Return the unsigned angle ``a-vertex-c`` in degrees, in ``[0, 180]``.

    Unlike :func:`directional_angle`, the result is independent of point
    order and of the person's facing direction.

    Args:
        a: First endpoint ``(x, y)``.
        vertex: Angle vertex ``(x, y)``.
        c: Second endpoint ``(x, y)``.

    Returns:
        Angle in degrees.

    Raises:
        ValueError: If either endpoint coincides with the vertex, in which
            case the angle is undefined.
    """
    v1 = np.asarray(a, dtype=float)[:2] - np.asarray(vertex, dtype=float)[:2]
    v2 = np.asarray(c, dtype=float)[:2] - np.asarray(vertex, dtype=float)[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        raise ValueError("Angle is undefined: an endpoint coincides with the vertex.")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def bbox_size(bbox: Sequence[float]) -> tuple[float, float]:
    """Return ``(width, height)`` of an ``(x1, y1, x2, y2[, score])`` box.

    Raises:
        ValueError: If the box has fewer than four values or non-positive size.
    """
    if len(bbox) < 4:
        raise ValueError(f"Expected bbox as (x1, y1, x2, y2[, score]), got {bbox!r}.")
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding box must have positive width and height, got {bbox!r}.")
    return width, height
