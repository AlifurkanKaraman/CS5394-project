"""AI driver interface for the car simulation.

Defines the canonical :class:`Action` enum and the :class:`DriverInterface`
abstract base class that all driver implementations must satisfy.

Keeping this module framework-agnostic means NEAT, manual, scripted, or
any future ML driver can be swapped in without touching the car entity or
simulation engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from ai_car_sim.domain.vehicle_state import VehicleState


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Discrete driving actions produced by any :class:`DriverInterface`.

    Values are intentionally small integers so they map directly onto
    NEAT network output indices (``output.index(max(output))``).
    """

    TURN_LEFT  = 0  # increase heading angle by the configured step
    TURN_RIGHT = 1  # decrease heading angle by the configured step
    SLOW_DOWN  = 2  # reduce speed (clamped to min_speed)
    SPEED_UP   = 3  # increase speed (clamped to max_speed)


# ---------------------------------------------------------------------------
# DriverInterface
# ---------------------------------------------------------------------------

class DriverInterface(ABC):
    """Abstract base class for all car-driving agents.

    Subclasses must implement :meth:`decide_action`.  The interface is
    intentionally minimal so it is easy to wrap any algorithm — NEAT
    networks, rule-based scripts, human keyboard input, or replay
    playback — without coupling to simulation internals.

    Example::

        class MyDriver(DriverInterface):
            def decide_action(
                self,
                sensor_inputs: list[float],
                state: VehicleState,
            ) -> Action:
                # always go straight
                return Action.SPEED_UP
    """

    @abstractmethod
    def decide_action(
        self,
        sensor_inputs: list[float],
        state: VehicleState,
    ) -> Action:
        """Choose the next driving action given current observations.

        Args:
            sensor_inputs: Normalised radar distances in ``[0.0, 1.0]``,
                one value per configured radar angle.  Produced by
                :meth:`~ai_car_sim.core.car.Car.get_sensor_inputs`.
            state: Current :class:`~ai_car_sim.domain.vehicle_state.VehicleState`
                of the car (read-only; drivers must not mutate it).

        Returns:
            The :class:`Action` to apply this tick.
        """

    def reset(self) -> None:
        """Optional hook called when the car is reset between episodes.

        Override to clear any per-episode internal state (e.g. hidden
        RNN state).  The default implementation is a no-op.
        """
