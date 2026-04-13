"""
Mutable car state domain model.

Holds the current position (x, y), heading angle, speed, cumulative
distance travelled, elapsed time ticks, alive flag, and optional
fitness/checkpoint tracking fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class VehicleState:
    """Mutable physical and gameplay state of a single car.

    This is a lightweight domain object with no UI or training logic.
    All fields are plain Python types so the state is trivially
    serialisable and testable.

    Args:
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
        angle: Heading angle in degrees (0 = right, increases counter-clockwise).
        speed: Current speed in pixels per tick.
        alive: Whether the car is still active (not crashed).
        distance_travelled: Cumulative distance driven this episode.
        time_steps: Number of simulation ticks elapsed.
        lap_progress: Optional normalised lap completion [0.0, 1.0].
        checkpoint_index: Index of the last checkpoint reached.
        fitness: Accumulated fitness / reward score.
    """

    x: float
    y: float
    angle: float
    speed: float
    alive: bool = True
    distance_travelled: float = 0.0
    time_steps: int = 0
    lap_progress: float | None = None
    checkpoint_index: int = 0
    fitness: float = 0.0

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def position_tuple(self) -> tuple[float, float]:
        """Return current position as an (x, y) tuple."""
        return (self.x, self.y)

    # ------------------------------------------------------------------
    # State mutation helpers
    # ------------------------------------------------------------------

    def mark_crashed(self) -> None:
        """Mark the car as crashed; sets alive to False and speed to 0."""
        self.alive = False
        self.speed = 0.0

    def advance_time(self, delta_distance: float = 0.0, delta_fitness: float = 0.0) -> None:
        """Increment the time counter and accumulate distance / fitness.

        Args:
            delta_distance: Distance covered this tick (added to distance_travelled).
            delta_fitness: Fitness gained this tick (added to fitness).
        """
        self.time_steps += 1
        self.distance_travelled += delta_distance
        self.fitness += delta_fitness

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise state to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VehicleState:
        """Deserialise state from a plain dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A new :class:`VehicleState` instance.
        """
        return cls(**data)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        status = "alive" if self.alive else "crashed"
        return (
            f"VehicleState(pos=({self.x:.1f}, {self.y:.1f}), "
            f"angle={self.angle:.1f}°, speed={self.speed:.1f}, "
            f"dist={self.distance_travelled:.1f}, t={self.time_steps}, {status})"
        )
