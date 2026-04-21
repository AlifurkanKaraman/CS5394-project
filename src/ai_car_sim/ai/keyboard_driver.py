"""Keyboard driver for manual car control.

Maps pygame arrow-key state to the discrete :class:`~ai_car_sim.ai.driver_interface.Action`
enum so a human player can drive using the same physics and collision
system as the AI cars.

Key bindings
------------
LEFT   — steer left  (TURN_LEFT)
RIGHT  — steer right (TURN_RIGHT)
UP     — accelerate  (SPEED_UP)
DOWN   — brake       (SLOW_DOWN)

When no key is held the driver returns SPEED_UP by default so the car
keeps moving (mirrors the original simulation behaviour where cars always
have a minimum speed).

pygame is imported lazily so the module loads cleanly in headless
environments.
"""

from __future__ import annotations

from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.domain.vehicle_state import VehicleState


class KeyboardDriver(DriverInterface):
    """Human-controlled driver that reads live pygame key state each tick.

    Because ``pygame.key.get_pressed()`` is polled every physics tick,
    the driver is always up-to-date without needing to buffer events.

    Args:
        default_action: Action returned when no directional key is held.
            Defaults to :attr:`~Action.SPEED_UP` so the car keeps moving.
    """

    def __init__(self, default_action: Action = Action.SPEED_UP) -> None:
        self._default = default_action

    def decide_action(
        self,
        sensor_inputs: list[float],
        state: VehicleState,
    ) -> Action:
        """Return the action corresponding to the currently held arrow key.

        Priority order when multiple keys are held simultaneously:
        LEFT > RIGHT > UP > DOWN.

        Args:
            sensor_inputs: Ignored — human drivers use visual feedback.
            state: Current vehicle state (unused but required by interface).

        Returns:
            The :class:`Action` matching the pressed key, or
            :attr:`default_action` when no directional key is held.
        """
        import pygame

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            return Action.TURN_LEFT
        if keys[pygame.K_RIGHT]:
            return Action.TURN_RIGHT
        if keys[pygame.K_UP]:
            return Action.SPEED_UP
        if keys[pygame.K_DOWN]:
            return Action.SLOW_DOWN

        return self._default
