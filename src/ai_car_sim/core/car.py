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
from ai_car_sim.domain.track import Track, Checkpoint

# Re-export so existing imports from this module keep working
__all__ = ["Car", "Action"]

# ---------------------------------------------------------------------------
# Stuck-detection constants
# ---------------------------------------------------------------------------
# How many ticks between progress snapshots
_STUCK_CHECK_INTERVAL: int = 60
# Minimum distance (px) the car must travel per interval to not be stuck
_STUCK_MIN_DISTANCE: float = 30.0
# Checkpoint reward bonus per new checkpoint reached
_CHECKPOINT_BONUS: float = 200.0


# ---------------------------------------------------------------------------
# Minimal pygame-surface protocol for draw() – avoids hard import at module level
# ---------------------------------------------------------------------------

@runtime_checkable
class DrawSurface(Protocol):
    """Subset of pygame.Surface needed for rendering."""

    def blit(self, source: object, dest: object) -> object: ...


# ---------------------------------------------------------------------------
# Segment-crossing helper for checkpoint detection
# ---------------------------------------------------------------------------

def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Return True if line segment p1-p2 intersects segment p3-p4.

    Uses the standard cross-product orientation test.
    """
    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


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
        track: Optional :class:`~ai_car_sim.domain.track.Track` used for
            checkpoint-based reward.  When ``None`` checkpoints are skipped.
    """

    def __init__(
        self,
        config: SimulationConfig,
        sprite_surface: object | None = None,
        manual_mode: bool = False,
        track: Track | None = None,
    ) -> None:
        self.config = config
        self.manual_mode = manual_mode
        self._track = track

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

        # Stuck detection: snapshot distance every N ticks
        self._stuck_snapshot_dist: float = 0.0
        self._stuck_snapshot_tick: int = 0

        # Checkpoint tracking: accumulated bonus from checkpoints
        self._checkpoint_bonus: float = 0.0

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

        Clears all per-episode state including stuck detection and
        checkpoint progress.

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
        self._stuck_snapshot_dist = 0.0
        self._stuck_snapshot_tick = 0
        self._checkpoint_bonus = 0.0

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
        """Advance physics, check collision, run stuck detection, and
        award checkpoint bonuses.

        Args:
            game_map: Map surface used for collision and radar pixel lookup.
        """
        if not self.state.alive:
            return

        s = self.state
        cfg = self.config

        prev_x, prev_y = s.x, s.y

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

        # Accumulate distance and time
        s.advance_time(delta_distance=s.speed)

        # Checkpoint detection (only in AI/training mode with a track)
        if not self.manual_mode and self._track is not None:
            self._check_checkpoints(prev_x, prev_y, s.x, s.y)

        # Stuck detection (only in AI/training mode)
        if not self.manual_mode:
            self._check_stuck()

    def _check_checkpoints(
        self,
        prev_x: float,
        prev_y: float,
        curr_x: float,
        curr_y: float,
    ) -> None:
        """Award a bonus if the car's movement crosses the next checkpoint.

        Checkpoints must be reached in order; already-passed checkpoints
        are ignored to prevent farming.

        Args:
            prev_x: Car X position before this tick's movement.
            prev_y: Car Y position before this tick's movement.
            curr_x: Car X position after this tick's movement.
            curr_y: Car Y position after this tick's movement.
        """
        checkpoints = self._track.checkpoints  # type: ignore[union-attr]
        if not checkpoints:
            return

        s = self.state
        next_idx = s.checkpoint_index  # next checkpoint to reach

        if next_idx >= len(checkpoints):
            return  # all checkpoints already passed

        cp = checkpoints[next_idx]
        cp_a = (float(cp[0][0]), float(cp[0][1]))
        cp_b = (float(cp[1][0]), float(cp[1][1]))

        # Use car centre for the movement segment
        cx_prev = prev_x + self.config.car_size_x / 2.0
        cy_prev = prev_y + self.config.car_size_y / 2.0
        cx_curr = curr_x + self.config.car_size_x / 2.0
        cy_curr = curr_y + self.config.car_size_y / 2.0

        if _segments_intersect(
            (cx_prev, cy_prev), (cx_curr, cy_curr), cp_a, cp_b
        ):
            s.checkpoint_index += 1
            self._checkpoint_bonus += _CHECKPOINT_BONUS
            # Update lap_progress as fraction of checkpoints completed
            s.lap_progress = s.checkpoint_index / len(checkpoints)

    def _check_stuck(self) -> None:
        """Kill the car if it hasn't made meaningful forward progress.

        Every ``_STUCK_CHECK_INTERVAL`` ticks, compare current
        ``distance_travelled`` against the snapshot taken at the last
        check.  If the delta is below ``_STUCK_MIN_DISTANCE`` the car is
        marked as crashed (inactive) so it stops consuming the generation
        budget without contributing useful fitness signal.
        """
        s = self.state
        if s.time_steps - self._stuck_snapshot_tick < _STUCK_CHECK_INTERVAL:
            return

        delta = s.distance_travelled - self._stuck_snapshot_dist
        self._stuck_snapshot_dist = s.distance_travelled
        self._stuck_snapshot_tick = s.time_steps

        if delta < _STUCK_MIN_DISTANCE:
            s.mark_crashed()  # treat stuck as crashed — stops reward accrual

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

        Fitness formula (all terms non-negative):

        1. **Distance score** — ``distance_travelled / (car_size_x / 2)``.
           This is the primary signal: cars that drive further score higher.
           Normalised so the value is in a human-readable range.

        2. **Checkpoint bonus** — flat bonus per checkpoint crossed
           (``_CHECKPOINT_BONUS`` per checkpoint).  Only active when the
           track has checkpoints defined.  Dominates distance score so
           reaching checkpoints is strongly preferred.

        Intentionally excluded:
        - Survival bonus (time_steps * k): causes spinning-in-place exploit
          where a car that does nothing scores higher than one that crashes
          while driving.
        - Efficiency bonus: distance already captures this; adding a ratio
          term creates non-linear interactions that confuse NEAT early on.

        Returns:
            Non-negative float fitness value.
        """
        distance_score = self.state.distance_travelled / self.config.car_half_size()
        return max(0.0, distance_score + self._checkpoint_bonus)

    def is_alive(self) -> bool:
        """Return whether the car has not yet crashed or been stuck-killed."""
        return self.state.alive

    # ------------------------------------------------------------------
    # Accessors for debug / HUD
    # ------------------------------------------------------------------

    @property
    def checkpoints_reached(self) -> int:
        """Number of checkpoints the car has passed this episode."""
        return self.state.checkpoint_index

    @property
    def distance_travelled(self) -> float:
        """Raw distance travelled this episode in pixels."""
        return self.state.distance_travelled

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
