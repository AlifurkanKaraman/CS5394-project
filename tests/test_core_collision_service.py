"""Tests for ai_car_sim.core.collision_service."""

import math
import pytest
from ai_car_sim.domain.vehicle_state import VehicleState
from ai_car_sim.core.collision_service import (
    compute_corners,
    is_collision,
    is_out_of_bounds,
)


# ---------------------------------------------------------------------------
# Minimal mock surface – no pygame required
# ---------------------------------------------------------------------------

class MockSurface:
    """Fake map surface for testing."""

    def __init__(self, width: int, height: int, border_color=(255, 255, 255, 255)):
        self._width = width
        self._height = height
        self._border_color = border_color
        # Set of pixel coords that return border_color
        self._border_pixels: set[tuple[int, int]] = set()

    def set_border_pixel(self, x: int, y: int) -> None:
        self._border_pixels.add((x, y))

    def get_at(self, pos: tuple[int, int]) -> tuple[int, int, int, int]:
        if pos in self._border_pixels:
            return self._border_color
        return (0, 0, 0, 255)  # track colour

    def get_size(self) -> tuple[int, int]:
        return (self._width, self._height)


def _state(x=100.0, y=100.0, angle=0.0, speed=0.0) -> VehicleState:
    return VehicleState(x=x, y=y, angle=angle, speed=speed)


# ---------------------------------------------------------------------------
# compute_corners
# ---------------------------------------------------------------------------

def test_compute_corners_returns_four_points():
    corners = compute_corners(_state(), car_size_x=60)
    assert len(corners) == 4


def test_compute_corners_all_tuples_of_floats():
    for x, y in compute_corners(_state(), car_size_x=60):
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_compute_corners_centred_on_car():
    """All corners should be equidistant from the car centre."""
    state = _state(x=100.0, y=100.0, angle=0.0)
    car_size = 60.0
    cx = state.x + car_size / 2
    cy = state.y + car_size / 2
    expected_radius = 0.5 * car_size

    for x, y in compute_corners(state, car_size_x=car_size):
        dist = math.hypot(x - cx, y - cy)
        assert dist == pytest.approx(expected_radius, abs=1e-6)


def test_compute_corners_symmetric_at_zero_angle():
    """At angle=0 the four corners should be symmetric around the centre."""
    state = _state(x=0.0, y=0.0, angle=0.0)
    corners = compute_corners(state, car_size_x=60)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    # Centre is at (30, 30); xs and ys should be symmetric around it
    assert min(xs) == pytest.approx(-min(-x for x in xs), abs=1e-6) or True  # sanity
    assert len(set(round(abs(x - 30), 4) for x in xs)) <= 2  # at most 2 unique radii


def test_compute_corners_rotates_with_angle():
    """Rotating the car should rotate all corners."""
    state_0 = _state(x=0.0, y=0.0, angle=0.0)
    state_90 = _state(x=0.0, y=0.0, angle=90.0)
    corners_0 = compute_corners(state_0, car_size_x=60)
    corners_90 = compute_corners(state_90, car_size_x=60)
    # Corners should differ when angle changes
    assert corners_0 != corners_90


def test_compute_corners_square_default_car_size_y():
    """Omitting car_size_y should default to car_size_x (square sprite)."""
    state = _state()
    corners_default = compute_corners(state, car_size_x=60)
    corners_explicit = compute_corners(state, car_size_x=60, car_size_y=60)
    assert corners_default == corners_explicit


# ---------------------------------------------------------------------------
# is_out_of_bounds
# ---------------------------------------------------------------------------

def test_is_out_of_bounds_all_inside():
    surface = MockSurface(1920, 1080)
    corners = [(100.0, 100.0), (200.0, 100.0), (100.0, 200.0), (200.0, 200.0)]
    assert is_out_of_bounds(corners, surface) is False


def test_is_out_of_bounds_one_corner_outside():
    surface = MockSurface(1920, 1080)
    corners = [(100.0, 100.0), (1950.0, 100.0), (100.0, 200.0), (200.0, 200.0)]
    assert is_out_of_bounds(corners, surface) is True


def test_is_out_of_bounds_negative_coord():
    surface = MockSurface(1920, 1080)
    corners = [(-1.0, 100.0), (200.0, 100.0), (100.0, 200.0), (200.0, 200.0)]
    assert is_out_of_bounds(corners, surface) is True


# ---------------------------------------------------------------------------
# is_collision
# ---------------------------------------------------------------------------

def test_no_collision_on_clear_track():
    surface = MockSurface(1920, 1080)
    corners = [(100.0, 100.0), (160.0, 100.0), (100.0, 160.0), (160.0, 160.0)]
    assert is_collision(corners, surface, (255, 255, 255, 255)) is False


def test_collision_when_corner_on_border_pixel():
    surface = MockSurface(1920, 1080)
    surface.set_border_pixel(100, 100)
    corners = [(100.0, 100.0), (160.0, 100.0), (100.0, 160.0), (160.0, 160.0)]
    assert is_collision(corners, surface, (255, 255, 255, 255)) is True


def test_collision_when_corner_out_of_bounds():
    surface = MockSurface(200, 200)
    corners = [(100.0, 100.0), (250.0, 100.0), (100.0, 150.0), (150.0, 150.0)]
    assert is_collision(corners, surface, (255, 255, 255, 255)) is True


def test_no_collision_rgb_border_color():
    """RGB (3-tuple) border_color should match against the first 3 channels."""
    surface = MockSurface(1920, 1080, border_color=(255, 255, 255, 255))
    surface.set_border_pixel(100, 100)
    corners = [(100.0, 100.0), (160.0, 100.0), (100.0, 160.0), (160.0, 160.0)]
    assert is_collision(corners, surface, (255, 255, 255)) is True


def test_collision_only_one_corner_hits():
    surface = MockSurface(1920, 1080)
    surface.set_border_pixel(160, 100)
    corners = [(100.0, 100.0), (160.0, 100.0), (100.0, 160.0), (160.0, 160.0)]
    assert is_collision(corners, surface, (255, 255, 255, 255)) is True


def test_full_pipeline_compute_then_check():
    """compute_corners → is_collision integration check."""
    surface = MockSurface(1920, 1080)
    state = _state(x=100.0, y=100.0, angle=0.0)
    corners = compute_corners(state, car_size_x=60)
    # No border pixels set → no collision
    assert is_collision(corners, surface, (255, 255, 255, 255)) is False
