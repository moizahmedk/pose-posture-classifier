from __future__ import annotations

from math import atan2, pi

import numpy as np
import pytest

from posture_classifier.geometry import bbox_size, directional_angle, interior_angle


def original_measureangle(a, b, c):
    """Verbatim logic from the original notebook (radians), for regression tests."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    angle_a = atan2(ay, ax)
    angle_c = atan2(cy, cx)
    if angle_a < 0:
        angle_a += pi * 2
    if angle_c < 0:
        angle_c += pi * 2
    return (pi * 2 + angle_c - angle_a) if angle_a > angle_c else (angle_c - angle_a)


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        ((0, -1), (0, 0), (0, 1), 180.0),  # straight line
        ((0, -1), (0, 0), (1, 0), 90.0),  # right angle
        ((1, 0), (0, 0), (1, 1), 45.0),
    ],
)
def test_interior_angle_known_values(a, b, c, expected):
    assert interior_angle(a, b, c) == pytest.approx(expected)


def test_interior_angle_is_order_and_mirror_invariant():
    a, b, c = (3.0, -4.0), (0.0, 0.0), (5.0, 2.0)
    mirrored = [(-x, y) for x, y in (a, b, c)]
    assert interior_angle(a, b, c) == pytest.approx(interior_angle(c, b, a))
    assert interior_angle(a, b, c) == pytest.approx(interior_angle(*mirrored))


def test_interior_angle_rejects_degenerate_points():
    with pytest.raises(ValueError, match="undefined"):
        interior_angle((1, 1), (1, 1), (2, 3))


def test_directional_angle_matches_original_notebook():
    rng = np.random.default_rng(0)
    for a, b, c in rng.uniform(-500, 500, size=(200, 3, 2)):
        expected = np.degrees(original_measureangle(a, b, c))
        assert directional_angle(a, b, c) == pytest.approx(expected, abs=1e-9)


def test_directional_angle_depends_on_facing_direction():
    """Why the refactor switched metrics: mirroring flips x into 360 - x."""
    a, b, c = (0, -100), (0, 0), (30, 95)
    mirrored = [(-x, y) for x, y in (a, b, c)]
    left, right = directional_angle(a, b, c), directional_angle(*mirrored)
    assert left + right == pytest.approx(360.0)
    assert interior_angle(a, b, c) == pytest.approx(min(left, right))


def test_bbox_size_and_validation():
    assert bbox_size([10, 20, 40, 100, 0.9]) == (30.0, 80.0)
    with pytest.raises(ValueError, match="positive"):
        bbox_size([10, 20, 5, 100])
    with pytest.raises(ValueError, match="Expected bbox"):
        bbox_size([1, 2, 3])
