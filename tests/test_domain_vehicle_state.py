"""Tests for ai_car_sim.domain.vehicle_state."""

import pytest
from ai_car_sim.domain.vehicle_state import VehicleState


def _make_state(**kwargs) -> VehicleState:
    defaults = dict(x=100.0, y=200.0, angle=45.0, speed=20.0)
    defaults.update(kwargs)
    return VehicleState(**defaults)


# ------------------------------------------------------------------
# Construction & defaults
# ------------------------------------------------------------------

def test_default_fields():
    s = _make_state()
    assert s.alive is True
    assert s.distance_travelled == 0.0
    assert s.time_steps == 0
    assert s.lap_progress is None
    assert s.checkpoint_index == 0
    assert s.fitness == 0.0


def test_explicit_fields():
    s = _make_state(alive=False, distance_travelled=500.0, time_steps=10, fitness=3.5)
    assert s.alive is False
    assert s.distance_travelled == 500.0
    assert s.time_steps == 10
    assert s.fitness == 3.5


# ------------------------------------------------------------------
# position_tuple
# ------------------------------------------------------------------

def test_position_tuple():
    s = _make_state(x=10.5, y=20.5)
    assert s.position_tuple() == (10.5, 20.5)


# ------------------------------------------------------------------
# mark_crashed
# ------------------------------------------------------------------

def test_mark_crashed_sets_alive_false():
    s = _make_state()
    assert s.alive is True
    s.mark_crashed()
    assert s.alive is False


def test_mark_crashed_zeroes_speed():
    s = _make_state(speed=30.0)
    s.mark_crashed()
    assert s.speed == 0.0


def test_mark_crashed_idempotent():
    s = _make_state()
    s.mark_crashed()
    s.mark_crashed()
    assert s.alive is False
    assert s.speed == 0.0


# ------------------------------------------------------------------
# advance_time
# ------------------------------------------------------------------

def test_advance_time_increments_steps():
    s = _make_state()
    s.advance_time()
    assert s.time_steps == 1
    s.advance_time()
    assert s.time_steps == 2


def test_advance_time_accumulates_distance():
    s = _make_state()
    s.advance_time(delta_distance=20.0)
    s.advance_time(delta_distance=15.0)
    assert s.distance_travelled == pytest.approx(35.0)


def test_advance_time_accumulates_fitness():
    s = _make_state()
    s.advance_time(delta_fitness=1.5)
    s.advance_time(delta_fitness=2.5)
    assert s.fitness == pytest.approx(4.0)


def test_advance_time_no_args_leaves_distance_and_fitness():
    s = _make_state(distance_travelled=10.0, fitness=5.0)
    s.advance_time()
    assert s.distance_travelled == pytest.approx(10.0)
    assert s.fitness == pytest.approx(5.0)


# ------------------------------------------------------------------
# Serialisation round-trip
# ------------------------------------------------------------------

def test_to_dict_contains_all_fields():
    s = _make_state(lap_progress=0.5, checkpoint_index=3)
    d = s.to_dict()
    assert d["x"] == 100.0
    assert d["y"] == 200.0
    assert d["angle"] == 45.0
    assert d["speed"] == 20.0
    assert d["alive"] is True
    assert d["lap_progress"] == 0.5
    assert d["checkpoint_index"] == 3


def test_from_dict_roundtrip():
    original = _make_state(
        distance_travelled=123.4,
        time_steps=7,
        fitness=9.9,
        lap_progress=0.75,
        checkpoint_index=2,
    )
    restored = VehicleState.from_dict(original.to_dict())
    assert restored.x == original.x
    assert restored.y == original.y
    assert restored.angle == original.angle
    assert restored.speed == original.speed
    assert restored.alive == original.alive
    assert restored.distance_travelled == pytest.approx(original.distance_travelled)
    assert restored.time_steps == original.time_steps
    assert restored.fitness == pytest.approx(original.fitness)
    assert restored.lap_progress == pytest.approx(original.lap_progress)
    assert restored.checkpoint_index == original.checkpoint_index


def test_from_dict_crashed_state():
    s = _make_state()
    s.mark_crashed()
    restored = VehicleState.from_dict(s.to_dict())
    assert restored.alive is False
    assert restored.speed == 0.0
