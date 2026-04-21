"""Focused tests for ai_car_sim.ai.neat_driver.NeatDriver.

Covers action mapping, input forwarding, output validation,
and integration with the DriverInterface contract.
"""

import pytest
from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.ai.neat_driver import NeatDriver
from ai_car_sim.domain.vehicle_state import VehicleState


# ------------------------------------------------------------------
# Stubs
# ------------------------------------------------------------------

class FixedNet:
    def __init__(self, outputs): self._out = outputs
    def activate(self, inputs): return list(self._out)


class RecordingNet:
    def __init__(self, outputs):
        self._out = outputs
        self.calls: list[list[float]] = []
    def activate(self, inputs):
        self.calls.append(list(inputs))
        return list(self._out)


def _state() -> VehicleState:
    return VehicleState(x=0.0, y=0.0, angle=0.0, speed=20.0)


def _inputs() -> list[float]:
    return [0.1, 0.2, 0.8, 0.4, 0.5]


# ------------------------------------------------------------------
# Interface compliance
# ------------------------------------------------------------------

def test_neat_driver_is_driver_interface():
    assert isinstance(NeatDriver(FixedNet([0, 0, 0, 1])), DriverInterface)


def test_neat_driver_cannot_be_instantiated_without_network():
    with pytest.raises(TypeError):
        NeatDriver()  # type: ignore


# ------------------------------------------------------------------
# Action selection (argmax)
# ------------------------------------------------------------------

@pytest.mark.parametrize("outputs,expected", [
    ([1.0, 0.0, 0.0, 0.0], Action.TURN_LEFT),
    ([0.0, 1.0, 0.0, 0.0], Action.TURN_RIGHT),
    ([0.0, 0.0, 1.0, 0.0], Action.SLOW_DOWN),
    ([0.0, 0.0, 0.0, 1.0], Action.SPEED_UP),
])
def test_argmax_selects_correct_action(outputs, expected):
    driver = NeatDriver(FixedNet(outputs))
    assert driver.decide_action(_inputs(), _state()) is expected


def test_all_equal_outputs_picks_first():
    driver = NeatDriver(FixedNet([0.5, 0.5, 0.5, 0.5]))
    assert driver.decide_action(_inputs(), _state()) is Action.TURN_LEFT


def test_negative_outputs_argmax():
    driver = NeatDriver(FixedNet([-0.5, -0.1, -0.9, -0.2]))
    assert driver.decide_action(_inputs(), _state()) is Action.TURN_RIGHT


def test_large_output_values():
    driver = NeatDriver(FixedNet([100.0, 200.0, 50.0, 150.0]))
    assert driver.decide_action(_inputs(), _state()) is Action.TURN_RIGHT


# ------------------------------------------------------------------
# Input forwarding
# ------------------------------------------------------------------

def test_sensor_inputs_passed_to_network():
    net = RecordingNet([0, 0, 0, 1])
    driver = NeatDriver(net)
    inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    driver.decide_action(inputs, _state())
    assert net.calls[0] == inputs


def test_empty_inputs_forwarded():
    net = RecordingNet([0, 0, 0, 1])
    driver = NeatDriver(net)
    driver.decide_action([], _state())
    assert net.calls[0] == []


# ------------------------------------------------------------------
# Output validation
# ------------------------------------------------------------------

def test_empty_output_raises():
    class EmptyNet:
        def activate(self, inputs): return []
    with pytest.raises(ValueError, match="empty"):
        NeatDriver(EmptyNet()).decide_action(_inputs(), _state())


def test_wrong_output_count_raises_when_expected_set():
    driver = NeatDriver(FixedNet([0.1, 0.2, 0.3]), expected_outputs=4)
    with pytest.raises(ValueError, match="Expected 4"):
        driver.decide_action(_inputs(), _state())


def test_correct_output_count_no_raise():
    driver = NeatDriver(FixedNet([0.1, 0.2, 0.3, 0.4]), expected_outputs=4)
    driver.decide_action(_inputs(), _state())  # should not raise


def test_no_expected_outputs_skips_count_check():
    driver = NeatDriver(FixedNet([0.9, 0.1, 0.2]), expected_outputs=None)
    result = driver.decide_action(_inputs(), _state())
    assert result is Action.TURN_LEFT


# ------------------------------------------------------------------
# reset (inherited no-op)
# ------------------------------------------------------------------

def test_reset_does_not_raise():
    NeatDriver(FixedNet([0, 0, 0, 1])).reset()


# ------------------------------------------------------------------
# State is not mutated by decide_action
# ------------------------------------------------------------------

def test_decide_action_does_not_mutate_state():
    state = _state()
    original_x = state.x
    NeatDriver(FixedNet([0, 0, 0, 1])).decide_action(_inputs(), state)
    assert state.x == original_x
