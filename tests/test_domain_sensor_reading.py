"""Tests for ai_car_sim.domain.sensor_reading."""

import pytest
from ai_car_sim.domain.sensor_reading import SensorReading


def _make_reading(**kwargs) -> SensorReading:
    defaults = dict(angle_offset=0.0, hit_x=150, hit_y=200, distance=120.0)
    defaults.update(kwargs)
    return SensorReading(**defaults)


# ------------------------------------------------------------------
# Construction & defaults
# ------------------------------------------------------------------

def test_default_normalized_distance_is_none():
    r = _make_reading()
    assert r.normalized_distance is None


def test_explicit_fields():
    r = _make_reading(angle_offset=-45.0, hit_x=10, hit_y=20, distance=50.0)
    assert r.angle_offset == -45.0
    assert r.hit_x == 10
    assert r.hit_y == 20
    assert r.distance == 50.0


# ------------------------------------------------------------------
# hit_point
# ------------------------------------------------------------------

def test_hit_point_returns_tuple():
    r = _make_reading(hit_x=300, hit_y=400)
    assert r.hit_point() == (300, 400)


# ------------------------------------------------------------------
# as_input_value – normalisation
# ------------------------------------------------------------------

def test_as_input_value_exact_max():
    r = _make_reading(distance=300.0)
    assert r.as_input_value(300.0) == pytest.approx(1.0)


def test_as_input_value_half_max():
    r = _make_reading(distance=150.0)
    assert r.as_input_value(300.0) == pytest.approx(0.5)


def test_as_input_value_clamped_above_max():
    # distance > max_distance should clamp to 1.0
    r = _make_reading(distance=400.0)
    assert r.as_input_value(300.0) == pytest.approx(1.0)


def test_as_input_value_zero_distance():
    r = _make_reading(distance=0.0)
    assert r.as_input_value(300.0) == pytest.approx(0.0)


def test_as_input_value_caches_normalized_distance():
    r = _make_reading(distance=60.0)
    result = r.as_input_value(300.0)
    assert r.normalized_distance == pytest.approx(result)
    assert r.normalized_distance == pytest.approx(0.2)


def test_as_input_value_invalid_max_distance_raises():
    r = _make_reading()
    with pytest.raises(ValueError):
        r.as_input_value(0.0)


def test_as_input_value_negative_max_distance_raises():
    r = _make_reading()
    with pytest.raises(ValueError):
        r.as_input_value(-10.0)


# ------------------------------------------------------------------
# Serialisation round-trip
# ------------------------------------------------------------------

def test_to_dict_fields():
    r = _make_reading(angle_offset=45.0, normalized_distance=0.4)
    d = r.to_dict()
    assert d["angle_offset"] == 45.0
    assert d["hit_x"] == 150
    assert d["hit_y"] == 200
    assert d["distance"] == 120.0
    assert d["normalized_distance"] == pytest.approx(0.4)


def test_from_dict_roundtrip():
    original = _make_reading(angle_offset=-90.0, distance=75.0)
    original.as_input_value(300.0)
    restored = SensorReading.from_dict(original.to_dict())
    assert restored.angle_offset == original.angle_offset
    assert restored.hit_x == original.hit_x
    assert restored.hit_y == original.hit_y
    assert restored.distance == pytest.approx(original.distance)
    assert restored.normalized_distance == pytest.approx(original.normalized_distance)


def test_from_dict_none_normalized():
    r = _make_reading()
    restored = SensorReading.from_dict(r.to_dict())
    assert restored.normalized_distance is None
