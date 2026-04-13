"""Tests for ai_car_sim.ai.neat_driver."""

import pytest
from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.ai.neat_driver import NeatDriver
from ai_car_sim.domain.vehicle_state import VehicleState


# ---------------------------------------------------------------------------
# Fake network stubs
# ---------------------------------------------------------------------------

class FixedOutputNetwork:
    """Returns a pre-set output vector regardless of inputs."""

    def __init__(self, outputs: list[float]) -> None:
        self._outputs = outputs

    def activate(self, inputs: list[float]) -> list[float]:
        return list(self._outputs)


class EchoNetwork:
    """Returns the inputs unchanged (useful for checking pass-through)."""

    def activate(self, inputs: list[float]) -> list[float]:
        return list(inputs)


class EmptyOutputNetwork:
    def activate(self, inputs):
        return []


def _state() -> VehicleState:
    return VehicleState(x=0.0, y=0.0, angle=0.0, speed=20.0)


def _inputs() -> list[float]:
    return [0.5, 0.5, 1.0, 0.5, 0.5]


# ---------------------------------------------------------------------------
# NeatDriver is a DriverInterface
# ---------------------------------------------------------------------------

def test_neat_driver_is_driver_interface():
    driver = NeatDriver(FixedOutputNetwork([0.1, 0.2, 0.3, 0.4]))
    assert isinstance(driver, DriverInterface)


# ---------------------------------------------------------------------------
# decide_action – argmax mapping
# ---------------------------------------------------------------------------

def test_turn_left_selected_when_output_0_highest():
    net = FixedOutputNetwork([1.0, 0.0, 0.0, 0.0])
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.TURN_LEFT


def test_turn_right_selected_when_output_1_highest():
    net = FixedOutputNetwork([0.0, 1.0, 0.0, 0.0])
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.TURN_RIGHT


def test_slow_down_selected_when_output_2_highest():
    net = FixedOutputNetwork([0.0, 0.0, 1.0, 0.0])
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.SLOW_DOWN


def test_speed_up_selected_when_output_3_highest():
    net = FixedOutputNetwork([0.0, 0.0, 0.0, 1.0])
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.SPEED_UP


def test_argmax_with_all_equal_picks_first():
    net = FixedOutputNetwork([0.5, 0.5, 0.5, 0.5])
    # list.index returns the first maximum
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.TURN_LEFT


def test_argmax_with_negative_values():
    net = FixedOutputNetwork([-0.1, -0.5, -0.2, -0.05])
    assert NeatDriver(net).decide_action(_inputs(), _state()) is Action.SPEED_UP


def test_sensor_inputs_forwarded_to_network():
    received: list[list[float]] = []

    class RecordingNetwork:
        def activate(self, inputs):
            received.append(list(inputs))
            return [0.0, 0.0, 0.0, 1.0]

    driver = NeatDriver(RecordingNetwork())
    inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    driver.decide_action(inputs, _state())
    assert received[0] == inputs


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def test_empty_output_raises_value_error():
    driver = NeatDriver(EmptyOutputNetwork())
    with pytest.raises(ValueError, match="empty"):
        driver.decide_action(_inputs(), _state())


def test_wrong_output_count_raises_when_expected_set():
    net = FixedOutputNetwork([0.1, 0.2, 0.3])  # 3 outputs, expected 4
    driver = NeatDriver(net, expected_outputs=4)
    with pytest.raises(ValueError, match="Expected 4"):
        driver.decide_action(_inputs(), _state())


def test_correct_output_count_does_not_raise():
    net = FixedOutputNetwork([0.1, 0.2, 0.3, 0.4])
    driver = NeatDriver(net, expected_outputs=4)
    driver.decide_action(_inputs(), _state())  # should not raise


def test_no_expected_outputs_skips_validation():
    net = FixedOutputNetwork([0.1, 0.9, 0.3])  # 3 outputs, no expectation set
    driver = NeatDriver(net, expected_outputs=None)
    result = driver.decide_action(_inputs(), _state())
    assert result is Action.TURN_RIGHT


# ---------------------------------------------------------------------------
# reset (inherited no-op)
# ---------------------------------------------------------------------------

def test_reset_does_not_raise():
    driver = NeatDriver(FixedOutputNetwork([0.0, 0.0, 0.0, 1.0]))
    driver.reset()  # should not raise
