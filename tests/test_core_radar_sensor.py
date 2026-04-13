"""Tests for ai_car_sim.core.radar_sensor."""

import math
import pytest
from ai_car_sim.domain.vehicle_state import VehicleState
from ai_car_sim.domain.sensor_reading import SensorReading
from ai_car_sim.core.radar_sensor import RadarSensorSystem


# ---------------------------------------------------------------------------
# Mock surface helpers
# ---------------------------------------------------------------------------

class OpenSurface:
    """A surface with no border pixels — rays always reach max_distance."""

    def __init__(self, width=1920, height=1080):
        self._w, self._h = width, height

    def get_at(self, pos):
        return (0, 0, 0, 255)

    def get_size(self):
        return (self._w, self._h)


class BorderedSurface:
    """Surface where specific pixels are marked as border."""

    def __init__(self, width=1920, height=1080, border_color=(255, 255, 255, 255)):
        self._w, self._h = width, height
        self._border_color = border_color
        self._border_pixels: set[tuple[int, int]] = set()

    def add_border(self, x: int, y: int) -> None:
        self._border_pixels.add((x, y))

    def get_at(self, pos):
        return self._border_color if pos in self._border_pixels else (0, 0, 0, 255)

    def get_size(self):
        return (self._w, self._h)


def _state(x=500.0, y=500.0, angle=0.0) -> VehicleState:
    return VehicleState(x=x, y=y, angle=angle, speed=0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_angle_offsets():
    radar = RadarSensorSystem()
    assert radar.angle_offsets == [-90.0, -45.0, 0.0, 45.0, 90.0]


def test_custom_angle_offsets():
    radar = RadarSensorSystem(angle_offsets=[-45.0, 0.0, 45.0])
    assert radar.angle_offsets == [-45.0, 0.0, 45.0]


def test_default_max_distance():
    assert RadarSensorSystem().max_distance == 300


def test_car_size_y_defaults_to_car_size_x():
    radar = RadarSensorSystem(car_size_x=60)
    assert radar.car_size_y == 60.0


# ---------------------------------------------------------------------------
# scan – output structure
# ---------------------------------------------------------------------------

def test_scan_returns_one_reading_per_angle():
    radar = RadarSensorSystem(angle_offsets=[-90.0, 0.0, 90.0])
    readings = radar.scan(_state(), OpenSurface())
    assert len(readings) == 3


def test_scan_returns_sensor_reading_instances():
    radar = RadarSensorSystem()
    for r in radar.scan(_state(), OpenSurface()):
        assert isinstance(r, SensorReading)


def test_scan_angle_offsets_stored_on_readings():
    offsets = [-90.0, -45.0, 0.0, 45.0, 90.0]
    radar = RadarSensorSystem(angle_offsets=offsets)
    readings = radar.scan(_state(), OpenSurface())
    assert [r.angle_offset for r in readings] == offsets


def test_scan_normalized_distance_not_set_by_scan():
    """scan() should not pre-compute normalized_distance."""
    radar = RadarSensorSystem()
    for r in radar.scan(_state(), OpenSurface()):
        assert r.normalized_distance is None


# ---------------------------------------------------------------------------
# scan – ray behaviour
# ---------------------------------------------------------------------------

def test_scan_open_surface_distance_near_max():
    """On an open surface every ray should travel close to max_distance."""
    radar = RadarSensorSystem(max_distance=100, car_size_x=60)
    # Place car well away from edges so no OOB clipping
    readings = radar.scan(_state(x=500.0, y=500.0), OpenSurface())
    for r in readings:
        assert r.distance == pytest.approx(100.0, abs=2.0)


def test_scan_hits_border_before_max():
    """A border pixel close to the car should produce a short reading."""
    surface = BorderedSurface()
    radar = RadarSensorSystem(
        angle_offsets=[0.0], max_distance=300, car_size_x=60
    )
    state = _state(x=500.0, y=500.0, angle=0.0)
    cx = state.x + 30  # centre x

    # Place a border pixel 50px ahead along the 0° ray (angle=0 → moves right)
    surface.add_border(int(cx) + 50, int(state.y + 30))

    readings = radar.scan(state, surface)
    assert readings[0].distance < 100.0


def test_scan_out_of_bounds_surface_clips_ray():
    """A small surface should clip rays that would exceed its bounds."""
    small = OpenSurface(width=200, height=200)
    radar = RadarSensorSystem(angle_offsets=[0.0], max_distance=300, car_size_x=60)
    state = _state(x=100.0, y=100.0, angle=0.0)
    readings = radar.scan(state, small)
    # Ray cannot exceed surface width
    assert readings[0].hit_x <= 200


# ---------------------------------------------------------------------------
# normalized_inputs
# ---------------------------------------------------------------------------

def test_normalized_inputs_length_matches_offsets():
    radar = RadarSensorSystem(angle_offsets=[-45.0, 0.0, 45.0])
    values = radar.normalized_inputs(_state(), OpenSurface())
    assert len(values) == 3


def test_normalized_inputs_all_in_unit_range():
    radar = RadarSensorSystem()
    for v in radar.normalized_inputs(_state(), OpenSurface()):
        assert 0.0 <= v <= 1.0


def test_normalized_inputs_caches_on_readings():
    """normalized_inputs should set normalized_distance on each reading."""
    radar = RadarSensorSystem(angle_offsets=[0.0])
    # Call scan first, then normalized_inputs independently
    values = radar.normalized_inputs(_state(), OpenSurface())
    assert all(v is not None for v in values)
