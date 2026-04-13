"""Track domain model representing a drivable map.

Holds the track name, path to the map image, spawn position and angle,
border color used for collision detection, and optional checkpoint
line segments for progress measurement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# A checkpoint is a pair of (x, y) endpoints defining a line segment across the track.
Checkpoint = tuple[tuple[int, int], tuple[int, int]]
BorderColor = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass
class Track:
    """Represents a drivable map and all metadata the simulation needs to use it.

    Attributes:
        name: Human-readable track name.
        map_image_path: Relative or absolute path to the map PNG asset.
        spawn_position: (x, y) pixel coordinates where cars are placed at the start.
        spawn_angle: Initial heading in degrees (0 = right, 90 = up in pygame coords).
        border_color: RGBA or RGB pixel color that marks the track boundary.
        checkpoints: Optional list of line segments used to measure lap progress.
        description: Free-text description shown in the UI.
    """

    name: str
    map_image_path: str
    spawn_position: tuple[float, float]
    spawn_angle: float
    border_color: BorderColor = (255, 255, 255, 255)
    checkpoints: list[Checkpoint] | None = None
    description: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the track to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "map_image_path": self.map_image_path,
            "spawn_position": list(self.spawn_position),
            "spawn_angle": self.spawn_angle,
            "border_color": list(self.border_color),
            "checkpoints": (
                [[list(a), list(b)] for a, b in self.checkpoints]
                if self.checkpoints is not None
                else None
            ),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        """Deserialise a track from a dictionary (e.g. loaded from JSON).

        Args:
            data: Dictionary with keys matching the Track fields.

        Returns:
            A new Track instance.

        Raises:
            KeyError: If a required field is missing from *data*.
            ValueError: If field values cannot be converted to the expected types.
        """
        checkpoints_raw = data.get("checkpoints")
        checkpoints: list[Checkpoint] | None = None
        if checkpoints_raw is not None:
            checkpoints = [
                ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
                for a, b in checkpoints_raw
            ]

        border_raw = data.get("border_color", [255, 255, 255, 255])
        border_color: BorderColor = (
            (int(border_raw[0]), int(border_raw[1]), int(border_raw[2]), int(border_raw[3]))
            if len(border_raw) == 4
            else (int(border_raw[0]), int(border_raw[1]), int(border_raw[2]))
        )

        spawn = data["spawn_position"]

        return cls(
            name=str(data["name"]),
            map_image_path=str(data["map_image_path"]),
            spawn_position=(float(spawn[0]), float(spawn[1])),
            spawn_angle=float(data["spawn_angle"]),
            border_color=border_color,
            checkpoints=checkpoints,
            description=str(data.get("description", "")),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_map_path(self, base_dir: str) -> str:
        """Return an absolute path to the map image, resolved against *base_dir*.

        If ``map_image_path`` is already absolute it is returned unchanged.

        Args:
            base_dir: Directory to resolve relative paths against (typically the
                project root or the ``assets/`` directory).

        Returns:
            Absolute path string.
        """
        if os.path.isabs(self.map_image_path):
            return self.map_image_path
        return os.path.normpath(os.path.join(base_dir, self.map_image_path))

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if the track is valid).

        Checks:
        - ``name`` is non-empty.
        - ``map_image_path`` is non-empty.
        - ``border_color`` has 3 or 4 components, each in [0, 255].
        - Each checkpoint is a pair of two (x, y) tuples.
        """
        errors: list[str] = []

        if not self.name.strip():
            errors.append("Track name must not be empty.")

        if not self.map_image_path.strip():
            errors.append("map_image_path must not be empty.")

        if len(self.border_color) not in (3, 4):
            errors.append("border_color must have 3 or 4 components.")
        elif not all(0 <= c <= 255 for c in self.border_color):
            errors.append("border_color components must be in the range [0, 255].")

        if self.checkpoints is not None:
            for i, cp in enumerate(self.checkpoints):
                if len(cp) != 2:
                    errors.append(f"Checkpoint {i} must be a pair of two points.")

        return errors
