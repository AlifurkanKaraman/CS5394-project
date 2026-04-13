"""Focused tests for ai_car_sim.core.vector_utils.

Covers mathematical correctness, edge cases, and consistency
between related functions.
"""

import math
import pytest
from ai_car_sim.core.vector_utils import (
    distance,
    rotate_point,
    clamp,
    heading_to_vector,
    normalize_angle,
    angle_between,
)


# ------------------------------------------------------------------
# distance
# ------------------------------------------------------------------

def test_distance_zero():
    assert distance(0, 0, 0, 0) == pytest.approx(0.0)


def test_distance_3_4_5():
    assert distance(0, 0, 3, 4) == pytest.approx(5.0)


def test_distance_negative_coords():
    assert distance(-3, -4, 0, 0) == pytest.approx(5.0)


def test_distance_symmetric():
    assert distance(1, 2, 5, 6) == pytest.approx(distance(5, 6, 1, 2))


def test_distance_horizontal():
    assert distance(0, 0, 10, 0) == pytest.approx(10.0)


def test_distance_vertical():
    assert distance(0, 0, 0, 7) == pytest.approx(7.0)


def test_distance_float_precision():
    assert distance(0.0, 0.0, 1.0, 1.0) == pytest.approx(math.sqrt(2))


# ------------------------------------------------------------------
# rotate_point
# ------------------------------------------------------------------

def test_rotate_point_zero_length():
    x, y = rotate_point(50.0, 50.0, 0.0, 0.0)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(50.0)


def test_rotate_point_180_degrees():
    # angle=180 → 360-180=180 → cos(180°)=-1, sin(180°)=0
    x, y = rotate_point(0.0, 0.0, 180.0, 10.0)
    assert x == pytest.approx(-10.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_rotate_point_270_degrees():
    # angle=270 → 360-270=90 → cos(90°)=0, sin(90°)=1
    x, y = rotate_point(0.0, 0.0, 270.0, 10.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(10.0, abs=1e-9)


def test_rotate_point_result_distance_equals_length():
    cx, cy, length = 100.0, 200.0, 50.0
    for angle in [0, 30, 45, 90, 135, 180, 270]:
        x, y = rotate_point(cx, cy, float(angle), length)
        assert distance(cx, cy, x, y) == pytest.approx(length, abs=1e-6)


def test_rotate_point_with_nonzero_origin():
    x, y = rotate_point(10.0, 10.0, 0.0, 5.0)
    assert x == pytest.approx(15.0, abs=1e-9)
    assert y == pytest.approx(10.0, abs=1e-9)


# ------------------------------------------------------------------
# clamp
# ------------------------------------------------------------------

def test_clamp_inside():
    assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)


def test_clamp_at_min():
    assert clamp(0.0, 0.0, 10.0) == pytest.approx(0.0)


def test_clamp_at_max():
    assert clamp(10.0, 0.0, 10.0) == pytest.approx(10.0)


def test_clamp_below_min():
    assert clamp(-1.0, 0.0, 10.0) == pytest.approx(0.0)


def test_clamp_above_max():
    assert clamp(11.0, 0.0, 10.0) == pytest.approx(10.0)


def test_clamp_equal_bounds():
    assert clamp(99.0, 5.0, 5.0) == pytest.approx(5.0)


def test_clamp_negative_range():
    assert clamp(-3.0, -10.0, -1.0) == pytest.approx(-3.0)


def test_clamp_invalid_bounds_raises():
    with pytest.raises(ValueError):
        clamp(5.0, 10.0, 0.0)


# ------------------------------------------------------------------
# heading_to_vector
# ------------------------------------------------------------------

def test_heading_to_vector_unit_length():
    for angle in range(0, 360, 15):
        dx, dy = heading_to_vector(float(angle))
        assert math.hypot(dx, dy) == pytest.approx(1.0, abs=1e-9)


def test_heading_to_vector_zero_points_right():
    dx, dy = heading_to_vector(0.0)
    assert dx == pytest.approx(1.0, abs=1e-9)
    assert dy == pytest.approx(0.0, abs=1e-9)


def test_heading_to_vector_matches_rotate_point():
    for angle in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
        dx, dy = heading_to_vector(angle)
        rx, ry = rotate_point(0.0, 0.0, angle, 1.0)
        assert dx == pytest.approx(rx, abs=1e-9)
        assert dy == pytest.approx(ry, abs=1e-9)


# ------------------------------------------------------------------
# normalize_angle
# ------------------------------------------------------------------

def test_normalize_angle_zero():
    assert normalize_angle(0.0) == pytest.approx(0.0)


def test_normalize_angle_360_wraps_to_0():
    assert normalize_angle(360.0) == pytest.approx(0.0)


def test_normalize_angle_negative():
    assert normalize_angle(-90.0) == pytest.approx(270.0)


def test_normalize_angle_large_positive():
    assert normalize_angle(720.0) == pytest.approx(0.0)


def test_normalize_angle_result_in_range():
    for a in [-720, -360, -1, 0, 1, 180, 359, 360, 721]:
        result = normalize_angle(float(a))
        assert 0.0 <= result < 360.0


# ------------------------------------------------------------------
# angle_between
# ------------------------------------------------------------------

def test_angle_between_east():
    assert angle_between(0, 0, 1, 0) == pytest.approx(0.0)


def test_angle_between_south():
    assert angle_between(0, 0, 0, 1) == pytest.approx(90.0)


def test_angle_between_result_in_range():
    for x2, y2 in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        result = angle_between(0.0, 0.0, float(x2), float(y2))
        assert 0.0 <= result < 360.0
