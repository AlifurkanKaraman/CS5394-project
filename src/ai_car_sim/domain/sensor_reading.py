"""
Single radar sensor reading domain model.

Holds the endpoint coordinates where a radar ray terminated, the
distance from the car's centre to that endpoint, the angle offset the
ray was cast at, and an optional pre-computed normalised distance.

No pygame or UI logic lives here; this is a pure data object consumed
by AI drivers and the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensorReading:
    """One radar/sensor reading from the car to a surrounding border.

    Args:
        angle_offset: Angle (degrees) at which the ray was cast relative
            to the car's heading.  Negative = left, positive = right.
        hit_x: X pixel coordinate where the ray hit the border.
        hit_y: Y pixel coordinate where the ray hit the border.
        distance: Euclidean distance in pixels from the car centre to
            the hit point.
        normalized_distance: Pre-computed distance normalised to [0, 1]
            against some maximum range.  ``None`` until explicitly set
            or computed via :meth:`as_input_value`.
    """

    angle_offset: float
    hit_x: int
    hit_y: int
    distance: float
    normalized_distance: float | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def hit_point(self) -> tuple[int, int]:
        """Return the hit coordinates as an (x, y) tuple."""
        return (self.hit_x, self.hit_y)

    # ------------------------------------------------------------------
    # AI input helpers
    # ------------------------------------------------------------------

    def as_input_value(self, max_distance: float) -> float:
        """Return the distance normalised to [0.0, 1.0] for use as a
        neural-network input.

        The result is also cached in :attr:`normalized_distance` so
        repeated calls are cheap.

        Args:
            max_distance: The maximum sensor range used for normalisation.
                Must be > 0.

        Returns:
            ``distance / max_distance``, clamped to [0.0, 1.0].

        Raises:
            ValueError: If *max_distance* is not positive.
        """
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        self.normalized_distance = min(self.distance / max_distance, 1.0)
        return self.normalized_distance

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "angle_offset": self.angle_offset,
            "hit_x": self.hit_x,
            "hit_y": self.hit_y,
            "distance": self.distance,
            "normalized_distance": self.normalized_distance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SensorReading:
        """Deserialise from a plain dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A new :class:`SensorReading` instance.
        """
        return cls(
            angle_offset=float(data["angle_offset"]),
            hit_x=int(data["hit_x"]),
            hit_y=int(data["hit_y"]),
            distance=float(data["distance"]),
            normalized_distance=(
                float(data["normalized_distance"])
                if data.get("normalized_distance") is not None
                else None
            ),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SensorReading(angle={self.angle_offset}°, "
            f"hit=({self.hit_x}, {self.hit_y}), dist={self.distance:.1f})"
        )
