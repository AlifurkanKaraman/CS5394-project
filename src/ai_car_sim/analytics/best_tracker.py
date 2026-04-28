"""Persistent best-performance tracker for the AI car simulation.

Records the all-time best results across generations and training runs,
and persists them to a JSON file so they survive between sessions.

Design rules:
- A failed save is non-fatal (logs a warning, never crashes training).
- A failed load returns a fresh tracker (no prior bests).
- The tracker is purely a data object — no pygame, no NEAT imports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path relative to the project root
_DEFAULT_PATH = "outputs/best_performance.json"


@dataclass
class BestPerformanceTracker:
    """Tracks all-time best results across generations and runs.

    All fields are updated in-place via :meth:`update`.  Call
    :meth:`save` after each generation to persist results.

    Args:
        best_fitness: Highest fitness value ever recorded.
        best_fitness_generation: Generation number where best_fitness occurred.
        best_fitness_track: Track name where best_fitness occurred.
        best_distance: Longest distance (px) ever driven in one episode.
        best_distance_generation: Generation where best_distance occurred.
        best_distance_track: Track name where best_distance occurred.
        best_checkpoints: Most checkpoints ever reached in one episode.
        best_checkpoints_generation: Generation where best_checkpoints occurred.
        best_checkpoints_track: Track name where best_checkpoints occurred.
        total_generations_trained: Cumulative generations trained across all runs.
    """

    best_fitness: float = 0.0
    best_fitness_generation: int = 0
    best_fitness_track: str = ""

    best_distance: float = 0.0
    best_distance_generation: int = 0
    best_distance_track: str = ""

    best_checkpoints: int = 0
    best_checkpoints_generation: int = 0
    best_checkpoints_track: str = ""

    total_generations_trained: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        generation: int,
        track_name: str,
        best_fitness: float,
        best_distance: float,
        best_checkpoints: int,
    ) -> bool:
        """Update records if any new value beats the stored all-time best.

        Args:
            generation: Current generation number.
            track_name: Name of the track being trained on.
            best_fitness: Best fitness in this generation.
            best_distance: Best distance (px) in this generation.
            best_checkpoints: Most checkpoints reached in this generation.

        Returns:
            ``True`` if at least one record was broken, ``False`` otherwise.
        """
        improved = False
        self.total_generations_trained += 1

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_fitness_generation = generation
            self.best_fitness_track = track_name
            improved = True
            logger.info(
                "New all-time best fitness: %.1f (gen %d, %s)",
                best_fitness, generation, track_name,
            )

        if best_distance > self.best_distance:
            self.best_distance = best_distance
            self.best_distance_generation = generation
            self.best_distance_track = track_name
            improved = True
            logger.info(
                "New all-time best distance: %.0fpx (gen %d, %s)",
                best_distance, generation, track_name,
            )

        if best_checkpoints > self.best_checkpoints:
            self.best_checkpoints = best_checkpoints
            self.best_checkpoints_generation = generation
            self.best_checkpoints_track = track_name
            improved = True
            logger.info(
                "New all-time best checkpoints: %d (gen %d, %s)",
                best_checkpoints, generation, track_name,
            )

        return improved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = _DEFAULT_PATH) -> bool:
        """Persist the tracker to a JSON file.

        Non-fatal: logs a warning and returns ``False`` on failure.

        Args:
            path: Destination file path.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        dest = Path(path)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(asdict(self), indent=2), encoding="utf-8"
            )
            logger.debug("Saved best performance → %s", dest)
            return True
        except Exception as exc:
            logger.warning("Could not save best performance to %s: %s", dest, exc)
            return False

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_PATH) -> "BestPerformanceTracker":
        """Load a tracker from a JSON file.

        Returns a fresh tracker (all zeros) if the file does not exist or
        cannot be parsed — never raises.

        Args:
            path: Path to a JSON file previously written by :meth:`save`.

        Returns:
            A :class:`BestPerformanceTracker` instance.
        """
        src = Path(path)
        if not src.exists():
            logger.debug("No best-performance file at %s — starting fresh.", src)
            return cls()
        try:
            data: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
            known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            filtered = {k: v for k, v in data.items() if k in known}
            tracker = cls(**filtered)
            logger.info(
                "Loaded best performance ← %s (best_fitness=%.1f, gen=%d)",
                src, tracker.best_fitness, tracker.best_fitness_generation,
            )
            return tracker
        except Exception as exc:
            logger.warning(
                "Could not load best performance from %s: %s — starting fresh.",
                src, exc,
            )
            return cls()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def has_any_record(self) -> bool:
        """Return True if at least one non-zero record has been set."""
        return self.best_fitness > 0.0 or self.best_distance > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)
