"""
Simulation configuration domain model.

Centralises all tunable parameters for physics, sensors, training loop,
screen sizing, and runtime modes.  Replaces the hardcoded constants that
lived in the original single-file script.

Only standard-library dependencies are used (``json``, ``pathlib``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class SimulationConfig:
    """All configurable parameters for one simulation run.

    Defaults mirror the constants from the original ``newcar.py`` script
    so existing behaviour is preserved out of the box.

    Args:
        screen_width: Horizontal resolution in pixels.
        screen_height: Vertical resolution in pixels.
        car_size_x: Car sprite width in pixels.
        car_size_y: Car sprite height in pixels.
        border_color: RGBA tuple used for collision detection.
        max_radar_distance: Maximum ray-cast length in pixels.
        radar_angles: Degree offsets at which radar rays are cast,
            relative to the car's heading.
        default_speed: Initial speed assigned to each car (px/tick).
        max_speed: Hard upper limit on car speed (px/tick).
        min_speed: Hard lower limit on car speed (px/tick).
        fps: Target frames per second for the pygame loop.
        max_generations: NEAT evolution stops after this many generations.
        steps_per_generation: Tick budget per generation before forced
            termination (``fps * seconds``).
        fullscreen: Whether to open the pygame window in fullscreen mode.
        map_path: Relative or absolute path to the track image.
        output_dir: Directory where genomes, stats, and replays are saved.
        neat_config_path: Path to the NEAT library configuration file.
    """

    # Screen
    screen_width: int = 1920
    screen_height: int = 1080
    fullscreen: bool = True

    # Car sprite
    car_size_x: int = 60
    car_size_y: int = 60

    # Collision
    border_color: tuple[int, int, int, int] = (255, 255, 255, 255)

    # Radar / sensors
    max_radar_distance: int = 300
    radar_angles: list[int] = field(default_factory=lambda: [-90, -45, 0, 45, 90])

    # Physics — training / AI mode
    default_speed: float = 20.0
    max_speed: float = 40.0
    min_speed: float = 12.0

    # Physics — manual / player mode (slower, more controllable)
    manual_default_speed: float = 8.0
    manual_max_speed: float = 16.0
    manual_min_speed: float = 4.0
    manual_turn_step: float = 5.0   # degrees per tick (vs 10 for AI)

    # Simulation loop
    fps: int = 60
    max_generations: int = 1000
    steps_per_generation: int = 1200  # 60 fps * 20 s

    # Paths
    map_path: str = "assets/maps/map.png"
    output_dir: str = "outputs"
    neat_config_path: str = "configs/config.txt"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def generation_timeout_seconds(self) -> float:
        """Return the per-generation time budget in seconds."""
        return self.steps_per_generation / self.fps

    def car_half_size(self) -> float:
        """Return half the car's X dimension, used for reward scaling."""
        return self.car_size_x / 2.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dictionary.

        ``border_color`` and ``radar_angles`` are converted to lists so
        the result is JSON-safe.
        """
        d = asdict(self)
        d["border_color"] = list(self.border_color)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Create a :class:`SimulationConfig` from a plain dictionary.

        Unknown keys are silently ignored so configs from older versions
        remain loadable.

        Args:
            data: Dictionary, e.g. loaded from JSON.

        Returns:
            A new :class:`SimulationConfig` instance.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}

        # Coerce border_color back to a tuple
        if "border_color" in filtered:
            filtered["border_color"] = tuple(filtered["border_color"])

        return cls(**filtered)

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def save_to_json(self, path: str | Path) -> None:
        """Write the config to a JSON file.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path: str | Path) -> SimulationConfig:
        """Load a config from a JSON file.

        Args:
            path: Path to a JSON file previously written by
                :meth:`save_to_json`.

        Returns:
            A new :class:`SimulationConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SimulationConfig(screen={self.screen_width}x{self.screen_height}, "
            f"fps={self.fps}, generations={self.max_generations}, "
            f"map={self.map_path!r})"
        )
