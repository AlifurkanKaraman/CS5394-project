"""Car entity for the AI car simulation.

Composes VehicleState, RadarSensorSystem, and collision service into a
single entity.  Sprite loading and drawing remain here since they are
tightly coupled to the car's position and angle, but all physics,
sensing, and collision logic is delegated to focused modules.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ai_car_sim.domain.vehicle_state import VehicleState
from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.core.radar_sensor import RadarSensorSystem, MapSurface
from ai_car_sim.core.collision_service import compute_corners, is_collision
from ai_car_sim.core.vector_utils import clamp
from ai_car_sim.ai.driver_interface import Action

# Re-export so existing imports from this module keep working
__all__ = ["Car", "Action"]


# ---------------------------------------------------------------------------
# Minimal pygame-surface protocol for draw() – avoids hard import at module level
# ---------------------------------------------------------------------------

@runtime_checkable
class DrawSurface(Protocol):
    """Subset of pygame.Surface needed for rendering."""

    def blit(self, source: object, dest: object) -> object: ...


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

class Car:
    """Main car entity that owns state, sensors, and sprite rendering.

    Designed to be constructed once per generation and reset between
    episodes via :meth:`reset`.

    Args:
        config: Simulation configuration (sprite size, speeds, radar, …).
        sprite_surface: A pygame-like surface for the car sprite.  When
            ``None`` the car operates in headless / test mode and
            :meth:`draw` is a no-op.
        manual_mode: When ``True`` the car uses the slower manual-mode
            speed profile (``manual_default_speed``, ``manual_max_speed``,
            ``manual_min_speed``, ``manual_turn_step``) instead of the
            AI training profile.
    """

    def __init__(
        self,
        config: SimulationConfig,
        sprite_surface: object | None = None,
        manual_mode: bool = False,
    ) -> None:
        self.config = config
        self.manual_mode = manual_mode

        # Sprite (may be None in headless/test mode)
        self._sprite = sprite_surface
        self._rotated_sprite = sprite_surface

        # Pick the correct speed profile up-front
        self._default_speed = (
            config.manual_default_speed if manual_mode else config.default_speed
        )
        self._max_speed = (
            config.manual_max_speed if manual_mode else config.max_speed
        )
        self._min_speed = (
            config.manual_min_speed if manual_mode else config.min_speed
        )
        self._turn_step = (
            config.manual_turn_step if manual_mode else 10.0
        )
        # Speed delta per SPEED_UP / SLOW_DOWN action — proportional to profile
        self._speed_delta = 1.0 if manual_mode else 2.0

        # State – initialised properly by reset()
        self.state = VehicleState(
            x=float(config.screen_width // 2),
            y=float(config.screen_height // 2),
            angle=0.0,
            speed=self._default_speed,
        )

        # Radar system wired from config
        self.radar = RadarSensorSystem(
            angle_offsets=[float(a) for a in config.radar_angles],
            max_distance=config.max_radar_distance,
            border_color=config.border_color,
            car_size_x=float(config.car_size_x),
            car_size_y=float(config.car_size_y),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        spawn_x: float,
        spawn_y: float,
        spawn_angle: float = 0.0,
    ) -> None:
        """Restore the car to a spawn position for a new episode.

        Uses the mode-appropriate default speed (manual vs AI).

        Args:
            spawn_x: Starting X position (top-left of sprite).
            spawn_y: Starting Y position (top-left of sprite).
            spawn_angle: Starting heading angle in degrees.
        """
        self.state = VehicleState(
            x=spawn_x,
            y=spawn_y,
            angle=spawn_angle,
            speed=self._default_speed,
        )
        self._rotated_sprite = self._sprite

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(self, action: Action | int) -> None:
        """Update speed and angle based on a discrete action.

        Uses mode-specific turn step and speed delta so manual mode feels
        slower and more controllable than AI training mode.

        - TURN_LEFT  → angle += turn_step°
        - TURN_RIGHT → angle -= turn_step°
        - SLOW_DOWN  → speed -= speed_delta (clamped to min_speed)
        - SPEED_UP   → speed += speed_delta (clamped to max_speed)

        Args:
            action: One of the :class:`Action` values (or its int equivalent).
        """
        action = Action(action)
        s = self.state
        if action == Action.TURN_LEFT:
            s.angle += self._turn_step
        elif action == Action.TURN_RIGHT:
            s.angle -= self._turn_step
        elif action == Action.SLOW_DOWN:
            s.speed = clamp(s.speed - self._speed_delta, self._min_speed, self._max_speed)
        elif action == Action.SPEED_UP:
            s.speed = clamp(s.speed + self._speed_delta, self._min_speed, self._max_speed)

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------

    def update(self, game_map: MapSurface) -> None:
        """Advance physics, check collision, and update sensor readings.

        Mirrors the ``update`` method from ``newcar.py``:
        - Move position by speed in the heading direction
        - Clamp position to screen bounds
        - Compute rotated sprite (when sprite is available)
        - Check corner collision against the map
        - Accumulate distance and time

        Args:
            game_map: Map surface used for collision and radar pixel lookup.
        """
        if not self.state.alive:
            return

        s = self.state
        cfg = self.config

        # Move in heading direction (pygame 360-angle convention)
        rad = math.radians(360.0 - s.angle)
        dx = math.cos(rad) * s.speed
        dy = math.sin(rad) * s.speed

        s.x = clamp(s.x + dx, 20.0, cfg.screen_width - cfg.car_size_x - 20.0)
        s.y = clamp(s.y + dy, 20.0, cfg.screen_height - cfg.car_size_y - 20.0)

        # Rotate sprite if available
        if self._sprite is not None:
            self._rotated_sprite = self._rotate_sprite(self._sprite, s.angle)

        # Collision check
        corners = compute_corners(s, cfg.car_size_x, cfg.car_size_y)
        if is_collision(corners, game_map, cfg.border_color):
            s.mark_crashed()
            return

        # Accumulate metrics
        s.advance_time(delta_distance=s.speed)

    # ------------------------------------------------------------------
    # Sensor / AI interface
    # ------------------------------------------------------------------

    def get_sensor_inputs(self, game_map: MapSurface) -> list[float]:
        """Return normalised radar distances ready for neural-network input.

        Args:
            game_map: Map surface for radar pixel lookup.

        Returns:
            List of floats in ``[0.0, 1.0]``, one per radar angle.
        """
        return self.radar.normalized_inputs(self.state, game_map)

    def get_reward(self) -> float:
        """Return the accumulated fitness/reward for this episode.

        Fitness is composed of:
        - Base score: distance_travelled normalised by half the car width
          (same as the original formula, keeps values in a familiar range).
        - Survival bonus: small bonus per time-step alive, rewarding
          staying on track longer.
        - Stagnation penalty: if the car has been alive a long time but
          hasn't moved much, the raw distance score already captures this
          naturally — no extra penalty needed.

        The result is always non-negative.

        Returns:
            Non-negative float fitness value.
        """
        half = self.config.car_half_size()
        distance_score = self.state.distance_travelled / half

        # Small survival bonus: 0.1 per time-step alive (encourages staying
        # on track, but distance dominates so spinning in place is not rewarded)
        survival_bonus = self.state.time_steps * 0.1

        # Efficiency ratio: reward cars that cover distance quickly.
        # Avoids inflating score for cars that survive by barely moving.
        if self.state.time_steps > 0:
            efficiency = self.state.distance_travelled / self.state.time_steps
            # Normalise: AI default speed is ~20 px/tick, so efficiency near 1.0
            # means the car is moving at roughly full speed.
            efficiency_bonus = efficiency * 0.5
        else:
            efficiency_bonus = 0.0

        return max(0.0, distance_score + survival_bonus + efficiency_bonus)

    def is_alive(self) -> bool:
        """Return whether the car has not yet crashed."""
        return self.state.alive

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, screen: DrawSurface) -> None:
        """Blit the rotated car sprite onto *screen*.

        No-op when the car was constructed without a sprite (headless mode).

        Args:
            screen: Pygame-like surface to draw onto.
        """
        if self._rotated_sprite is None:
            return
        screen.blit(self._rotated_sprite, (int(self.state.x), int(self.state.y)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_sprite(sprite: object, angle: float) -> object:
        """Rotate *sprite* by *angle* degrees using pygame.

        Imported lazily so the module can be loaded without a display.

        Args:
            sprite: A ``pygame.Surface`` instance.
            angle: Rotation angle in degrees.

        Returns:
            A new rotated ``pygame.Surface``.
        """
        import pygame  # lazy import – only needed when rendering
        rectangle = sprite.get_rect()
        rotated = pygame.transform.rotate(sprite, angle)
        rotated_rect = rectangle.copy()
        rotated_rect.center = rotated.get_rect().center
        return rotated.subsurface(rotated_rect).copy()

    def __repr__(self) -> str:  # pragma: no cover
        return f"Car(state={self.state!r})"
