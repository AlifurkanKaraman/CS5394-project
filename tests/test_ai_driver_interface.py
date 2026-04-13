"""Tests for ai_car_sim.ai.driver_interface."""

import pytest
from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.domain.vehicle_state import VehicleState


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

def test_action_values_are_unique():
    values = [a.value for a in Action]
    assert len(values) == len(set(values))


def test_action_has_four_members():
    assert len(Action) == 4


def test_action_int_mapping():
    assert Action(0) is Action.TURN_LEFT
    assert Action(1) is Action.TURN_RIGHT
    assert Action(2) is Action.SLOW_DOWN
    assert Action(3) is Action.SPEED_UP


def test_action_is_int_comparable():
    assert Action.TURN_LEFT == 0
    assert Action.SPEED_UP == 3


# ---------------------------------------------------------------------------
# DriverInterface – abstract enforcement
# ---------------------------------------------------------------------------

def test_cannot_instantiate_driver_interface_directly():
    with pytest.raises(TypeError):
        DriverInterface()  # type: ignore[abstract]


def test_concrete_driver_must_implement_decide_action():
    """A subclass that omits decide_action should still raise TypeError."""
    class IncompleteDriver(DriverInterface):
        pass

    with pytest.raises(TypeError):
        IncompleteDriver()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete minimal driver
# ---------------------------------------------------------------------------

class ConstantDriver(DriverInterface):
    """Always returns the same action — useful for testing."""

    def __init__(self, action: Action) -> None:
        self._action = action

    def decide_action(self, sensor_inputs: list[float], state: VehicleState) -> Action:
        return self._action


def _state() -> VehicleState:
    return VehicleState(x=0.0, y=0.0, angle=0.0, speed=20.0)


def test_concrete_driver_returns_correct_action():
    driver = ConstantDriver(Action.SPEED_UP)
    result = driver.decide_action([0.5, 0.5, 1.0, 0.5, 0.5], _state())
    assert result is Action.SPEED_UP


def test_decide_action_receives_sensor_inputs():
    received: list[list[float]] = []

    class RecordingDriver(DriverInterface):
        def decide_action(self, sensor_inputs, state):
            received.append(sensor_inputs)
            return Action.TURN_LEFT

    driver = RecordingDriver()
    inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    driver.decide_action(inputs, _state())
    assert received[0] == inputs


def test_reset_is_noop_by_default():
    """Default reset() should not raise."""
    driver = ConstantDriver(Action.TURN_LEFT)
    driver.reset()  # should not raise


def test_reset_can_be_overridden():
    class StatefulDriver(DriverInterface):
        def __init__(self):
            self.reset_count = 0

        def decide_action(self, sensor_inputs, state):
            return Action.TURN_LEFT

        def reset(self):
            self.reset_count += 1

    driver = StatefulDriver()
    driver.reset()
    driver.reset()
    assert driver.reset_count == 2


# ---------------------------------------------------------------------------
# Action re-exported from car module (backward compat)
# ---------------------------------------------------------------------------

def test_action_importable_from_car_module():
    from ai_car_sim.core.car import Action as CarAction
    assert CarAction is Action
