"""Focused tests for ai_car_sim.domain.vehicle_state.VehicleState.

Covers state transitions, boundary conditions, and serialisation
edge cases beyond the basic suite in test_domain_vehicle_state.py.
"""

import pytest
from ai_car_sim.domain.vehicle_state import VehicleState


def _state(**kw) -> VehicleState:
    defaults = dict(x=100.0, y=200.0, angle=0.0, speed=20.0)
    defaults.update(kw)
    return VehicleState(**defaults)


# ------------------------------------------------------------------
# Initial state
# ------------------------------------------------------------------

def test_alive_by_default():
    assert _state().alive is True


def test_distance_zero_by_default():
    assert _state().distance_travelled == 0.0


def test_time_steps_zero_by_default():
    assert _state().time_steps == 0


def test_fitness_zero_by_default():
    assert _state().fitness == 0.0


def test_checkpoint_index_zero_by_default():
    assert _state().checkpoint_index == 0


def test_lap_progress_none_by_default():
    assert _state().lap_progress is None


# ------------------------------------------------------------------
# position_tuple
# ------------------------------------------------------------------

def test_position_tuple_values():
    s = _state(x=55.5, y=77.7)
    assert s.position_tuple() == (55.5, 77.7)


def test_position_tuple_type():
    assert isinstance(_state().position_tuple(), tuple)


# ------------------------------------------------------------------
# mark_crashed
# ------------------------------------------------------------------

def test_mark_crashed_sets_alive_false():
    s = _state()
    s.mark_crashed()
    assert s.alive is False


def test_mark_crashed_zeroes_speed():
    s = _state(speed=35.0)
    s.mark_crashed()
    assert s.speed == 0.0


def test_mark_crashed_does_not_change_position():
    s = _state(x=10.0, y=20.0)
    s.mark_crashed()
    assert s.x == 10.0
    assert s.y == 20.0


def test_mark_crashed_idempotent():
    s = _state()
    s.mark_crashed()
    s.mark_crashed()
    assert s.alive is False
    assert s.speed == 0.0


# ------------------------------------------------------------------
# advance_time
# ------------------------------------------------------------------

def test_advance_time_increments_steps_by_one():
    s = _state()
    s.advance_time()
    assert s.time_steps == 1


def test_advance_time_multiple_calls():
    s = _state()
    for _ in range(5):
        s.advance_time()
    assert s.time_steps == 5


def test_advance_time_accumulates_distance():
    s = _state()
    s.advance_time(delta_distance=10.0)
    s.advance_time(delta_distance=15.0)
    assert s.distance_travelled == pytest.approx(25.0)


def test_advance_time_accumulates_fitness():
    s = _state()
    s.advance_time(delta_fitness=2.5)
    s.advance_time(delta_fitness=3.5)
    assert s.fitness == pytest.approx(6.0)


def test_advance_time_zero_deltas_leaves_accumulators():
    s = _state()
    s.advance_time()
    assert s.distance_travelled == 0.0
    assert s.fitness == 0.0


def test_advance_time_does_not_change_position():
    s = _state(x=50.0, y=60.0)
    s.advance_time(delta_distance=100.0)
    assert s.x == 50.0
    assert s.y == 60.0


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------

def test_to_dict_round_trip_all_fields():
    s = _state(
        x=1.1, y=2.2, angle=45.0, speed=25.0,
        alive=False, distance_travelled=300.0,
        time_steps=15, lap_progress=0.5,
        checkpoint_index=3, fitness=12.5,
    )
    restored = VehicleState.from_dict(s.to_dict())
    assert restored.x == pytest.approx(s.x)
    assert restored.y == pytest.approx(s.y)
    assert restored.angle == pytest.approx(s.angle)
    assert restored.speed == pytest.approx(s.speed)
    assert restored.alive == s.alive
    assert restored.distance_travelled == pytest.approx(s.distance_travelled)
    assert restored.time_steps == s.time_steps
    assert restored.lap_progress == pytest.approx(s.lap_progress)
    assert restored.checkpoint_index == s.checkpoint_index
    assert restored.fitness == pytest.approx(s.fitness)


def test_to_dict_contains_all_keys():
    d = _state().to_dict()
    for key in ("x", "y", "angle", "speed", "alive",
                "distance_travelled", "time_steps",
                "lap_progress", "checkpoint_index", "fitness"):
        assert key in d


def test_from_dict_crashed_state():
    s = _state()
    s.mark_crashed()
    r = VehicleState.from_dict(s.to_dict())
    assert r.alive is False
    assert r.speed == 0.0


def test_from_dict_none_lap_progress():
    s = _state(lap_progress=None)
    r = VehicleState.from_dict(s.to_dict())
    assert r.lap_progress is None
