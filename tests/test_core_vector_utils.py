"""Tests for ai_car_sim.core.vector_utils."""

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

def test_distance_same_point():
    assert distance(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)


def test_distance_horizontal():
    assert distance(0.0, 0.0, 3.0, 0.0) == pytest.approx(3.0)


def test_distance_vertical():
    assert distance(0.0, 0.0, 0.0, 4.0) == pytest.approx(4.0)


def test_distance_pythagorean():
    assert distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_distance_negative_coords():
    assert distance(-1.0, -1.0, 2.0, 3.0) == pytest.approx(5.0)


def test_distance_is_symmetric():
    assert distance(1.0, 2.0, 5.0, 6.0) == pytest.approx(distance(5.0, 6.0, 1.0, 2.0))


# ------------------------------------------------------------------
# rotate_point
# ------------------------------------------------------------------

def test_rotate_point_zero_angle_moves_right():
    # angle=0 → 360-0=360 → cos(360°)=1, sin(360°)=0
    x, y = rotate_point(0.0, 0.0, 0.0, 10.0)
    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_rotate_point_90_degrees():
    # angle=90 → 360-90=270 → cos(270°)=0, sin(270°)=-1
    x, y = rotate_point(0.0, 0.0, 90.0, 10.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-10.0, abs=1e-9)


def test_rotate_point_with_offset_origin():
    x, y = rotate_point(100.0, 200.0, 0.0, 0.0)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(200.0)


def test_rotate_point_length_scales_result():
    x1, y1 = rotate_point(0.0, 0.0, 0.0, 5.0)
    x2, y2 = rotate_point(0.0, 0.0, 0.0, 10.0)
    assert x2 == pytest.approx(x1 * 2)
    assert y2 == pytest.approx(y1 * 2)


# ------------------------------------------------------------------
# clamp
# ------------------------------------------------------------------

def test_clamp_within_bounds():
    assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)


def test_clamp_at_lower_bound():
    assert clamp(0.0, 0.0, 10.0) == pytest.approx(0.0)


def test_clamp_at_upper_bound():
    assert clamp(10.0, 0.0, 10.0) == pytest.approx(10.0)


def test_clamp_below_lower_bound():
    assert clamp(-5.0, 0.0, 10.0) == pytest.approx(0.0)


def test_clamp_above_upper_bound():
    assert clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)


def test_clamp_equal_bounds():
    assert clamp(7.0, 5.0, 5.0) == pytest.approx(5.0)


def test_clamp_invalid_bounds_raises():
    with pytest.raises(ValueError):
        clamp(5.0, 10.0, 0.0)


# ------------------------------------------------------------------
# heading_to_vector
# ------------------------------------------------------------------

def test_heading_to_vector_is_unit_length():
    for angle in [0, 45, 90, 135, 180, 270, 315]:
        dx, dy = heading_to_vector(float(angle))
        length = math.hypot(dx, dy)
        assert length == pytest.approx(1.0, abs=1e-9), f"angle={angle}"


def test_heading_to_vector_zero_angle():
    dx, dy = heading_to_vector(0.0)
    assert dx == pytest.approx(1.0, abs=1e-9)
    assert dy == pytest.approx(0.0, abs=1e-9)


def test_heading_to_vector_consistent_with_rotate_point():
    # heading_to_vector should match rotate_point with length=1
    for angle in [0.0, 45.0, 90.0, 180.0, 270.0]:
        dx, dy = heading_to_vector(angle)
        rx, ry = rotate_point(0.0, 0.0, angle, 1.0)
        assert dx == pytest.approx(rx, abs=1e-9), f"angle={angle}"
        assert dy == pytest.approx(ry, abs=1e-9), f"angle={angle}"


# ------------------------------------------------------------------
# normalize_angle
# ------------------------------------------------------------------

def test_normalize_angle_already_in_range():
    assert normalize_angle(90.0) == pytest.approx(90.0)


def test_normalize_angle_360_becomes_0():
    assert normalize_angle(360.0) == pytest.approx(0.0)


def test_normalize_angle_negative():
    assert normalize_angle(-90.0) == pytest.approx(270.0)


def test_normalize_angle_large():
    assert normalize_angle(450.0) == pytest.approx(90.0)


# ------------------------------------------------------------------
# angle_between
# ------------------------------------------------------------------

def test_angle_between_east():
    assert angle_between(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)


def test_angle_between_south():
    # atan2(1, 0) = 90° in standard math coords
    assert angle_between(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0)


def test_angle_between_result_in_range():
    result = angle_between(0.0, 0.0, -1.0, -1.0)
    assert 0.0 <= result < 360.0
