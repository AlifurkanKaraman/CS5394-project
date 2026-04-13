"""Tests for ai_car_sim.core.car."""

import pytest
from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.core.car import Car, Action


# ---------------------------------------------------------------------------
# Minimal mock surface (no pygame)
# ---------------------------------------------------------------------------

class OpenSurface:
    """All pixels are track colour — no collisions, no OOB."""

    def __init__(self, width=1920, height=1080):
        self._w, self._h = width, height

    def get_at(self, pos):
        return (0, 0, 0, 255)

    def get_size(self):
        return (self._w, self._h)


class BorderSurface(OpenSurface):
    """Surface where every pixel is border colour — instant crash."""

    def get_at(self, pos):
        return (255, 255, 255, 255)


def _cfg(**kwargs) -> SimulationConfig:
    defaults = dict(
        screen_width=1920, screen_height=1080,
        car_size_x=60, car_size_y=60,
        default_speed=20.0, min_speed=12.0, max_speed=40.0,
        border_color=(255, 255, 255, 255),
        radar_angles=[-90, -45, 0, 45, 90],
        max_radar_distance=300,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def _car(**kwargs) -> Car:
    return Car(config=_cfg(**kwargs), sprite_surface=None)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_car_starts_alive():
    assert _car().is_alive() is True


def test_car_initial_speed_from_config():
    car = _car(default_speed=25.0)
    assert car.state.speed == 25.0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_restores_position():
    car = _car()
    car.reset(spawn_x=100.0, spawn_y=200.0, spawn_angle=45.0)
    assert car.state.x == 100.0
    assert car.state.y == 200.0
    assert car.state.angle == 45.0


def test_reset_restores_alive():
    car = _car()
    car.state.mark_crashed()
    car.reset(spawn_x=100.0, spawn_y=200.0)
    assert car.is_alive() is True


def test_reset_zeroes_distance_and_time():
    car = _car()
    car.state.advance_time(delta_distance=500.0)
    car.reset(spawn_x=100.0, spawn_y=200.0)
    assert car.state.distance_travelled == 0.0
    assert car.state.time_steps == 0


# ---------------------------------------------------------------------------
# apply_action
# ---------------------------------------------------------------------------

def test_turn_left_increases_angle():
    car = _car()
    initial = car.state.angle
    car.apply_action(Action.TURN_LEFT)
    assert car.state.angle == initial + 10.0


def test_turn_right_decreases_angle():
    car = _car()
    initial = car.state.angle
    car.apply_action(Action.TURN_RIGHT)
    assert car.state.angle == initial - 10.0


def test_speed_up_increases_speed():
    car = _car(default_speed=20.0, max_speed=40.0)
    car.apply_action(Action.SPEED_UP)
    assert car.state.speed == pytest.approx(22.0)


def test_slow_down_decreases_speed():
    car = _car(default_speed=20.0, min_speed=12.0)
    car.apply_action(Action.SLOW_DOWN)
    assert car.state.speed == pytest.approx(18.0)


def test_speed_clamped_at_max():
    car = _car(default_speed=40.0, max_speed=40.0)
    car.apply_action(Action.SPEED_UP)
    assert car.state.speed == pytest.approx(40.0)


def test_speed_clamped_at_min():
    car = _car(default_speed=12.0, min_speed=12.0)
    car.apply_action(Action.SLOW_DOWN)
    assert car.state.speed == pytest.approx(12.0)


def test_apply_action_accepts_int():
    car = _car()
    car.apply_action(0)  # TURN_LEFT as raw int
    assert car.state.angle == 10.0


# ---------------------------------------------------------------------------
# update – movement
# ---------------------------------------------------------------------------

def test_update_advances_position():
    car = _car(default_speed=20.0)
    car.reset(spawn_x=500.0, spawn_y=500.0, spawn_angle=0.0)
    x_before = car.state.x
    car.update(OpenSurface())
    assert car.state.x != x_before


def test_update_increments_time_steps():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.update(OpenSurface())
    assert car.state.time_steps == 1


def test_update_accumulates_distance():
    car = _car(default_speed=20.0)
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.update(OpenSurface())
    assert car.state.distance_travelled == pytest.approx(20.0)


def test_update_does_nothing_when_crashed():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.state.mark_crashed()
    car.update(OpenSurface())
    assert car.state.time_steps == 0  # no advance


# ---------------------------------------------------------------------------
# update – collision
# ---------------------------------------------------------------------------

def test_update_crashes_on_border_surface():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.update(BorderSurface())
    assert car.is_alive() is False


def test_update_stays_alive_on_open_surface():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.update(OpenSurface())
    assert car.is_alive() is True


# ---------------------------------------------------------------------------
# get_reward
# ---------------------------------------------------------------------------

def test_reward_zero_at_start():
    car = _car()
    assert car.get_reward() == pytest.approx(0.0)


def test_reward_grows_with_distance():
    car = _car(car_size_x=60)
    car.reset(spawn_x=500.0, spawn_y=500.0)
    car.update(OpenSurface())
    car.update(OpenSurface())
    # reward = distance / (car_size_x / 2) = (2 * speed) / 30
    expected = car.state.distance_travelled / 30.0
    assert car.get_reward() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_sensor_inputs
# ---------------------------------------------------------------------------

def test_sensor_inputs_length_matches_radar_angles():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    inputs = car.get_sensor_inputs(OpenSurface())
    assert len(inputs) == len(car.config.radar_angles)


def test_sensor_inputs_all_in_unit_range():
    car = _car()
    car.reset(spawn_x=500.0, spawn_y=500.0)
    for v in car.get_sensor_inputs(OpenSurface()):
        assert 0.0 <= v <= 1.0
