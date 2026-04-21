"""Run metrics collection and aggregation for the AI car simulation.

Turns the project into a proper experiment platform by capturing per-generation
statistics and run-level summaries that can be saved to JSON/CSV and discussed
in a report.

No rendering or training logic lives here — this module is purely about
data collection and export.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# GenerationMetrics
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    """Statistics captured at the end of one NEAT generation.

    Args:
        generation: Generation number (1-based).
        alive_count: Number of cars still alive when the generation ended.
        best_fitness: Highest fitness value in this generation.
        avg_fitness: Mean fitness across all genomes in this generation.
        elapsed_seconds: Wall-clock time the generation took to run.
        track_name: Name of the track used for this generation.
        population_size: Total number of genomes evaluated.
        artifact_paths: Optional mapping of label → file path for any
            artefacts saved during this generation (e.g. best genome pkl).
    """

    generation: int
    alive_count: int
    best_fitness: float
    avg_fitness: float
    elapsed_seconds: float
    track_name: str = ""
    population_size: int = 0
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    def to_row(self) -> list[Any]:
        """Return a flat list suitable for a CSV row.

        Column order matches :meth:`RunMetricsCollector.csv_headers`.
        """
        return [
            self.generation,
            self.alive_count,
            self.population_size,
            round(self.best_fitness, 6),
            round(self.avg_fitness, 6),
            round(self.elapsed_seconds, 3),
            self.track_name,
        ]


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """Aggregate statistics for a complete training run.

    Args:
        total_generations: Number of generations completed.
        total_elapsed_seconds: Total wall-clock time for the run.
        peak_fitness: Highest fitness seen across all generations.
        final_avg_fitness: Average fitness in the last generation.
        track_name: Track used for the run.
        neat_config_path: Path to the NEAT config file used.
        genome_path: Path to the saved best genome (if any).
    """

    total_generations: int
    total_elapsed_seconds: float
    peak_fitness: float
    final_avg_fitness: float
    track_name: str = ""
    neat_config_path: str = ""
    genome_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# RunMetricsCollector
# ---------------------------------------------------------------------------

class RunMetricsCollector:
    """Accumulates per-generation metrics and produces run-level summaries.

    Typical usage::

        collector = RunMetricsCollector(track_name="Circuit Alpha")
        # inside the generation callback:
        collector.record_generation(
            generation=gen,
            fitnesses=[g.fitness for _, g in genomes],
            alive_count=alive,
            elapsed_seconds=elapsed,
        )
        summary = collector.summarise(neat_config_path="configs/config.txt")

    Args:
        track_name: Name of the track used for this run.
    """

    CSV_HEADERS: list[str] = [
        "generation",
        "alive_count",
        "population_size",
        "best_fitness",
        "avg_fitness",
        "elapsed_seconds",
        "track_name",
    ]

    def __init__(self, track_name: str = "") -> None:
        self.track_name = track_name
        self._generations: list[GenerationMetrics] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_generation(
        self,
        generation: int,
        fitnesses: list[float],
        alive_count: int,
        elapsed_seconds: float,
        artifact_paths: dict[str, str] | None = None,
    ) -> GenerationMetrics:
        """Record statistics for one completed generation.

        Args:
            generation: Generation number (1-based).
            fitnesses: Fitness value for every genome in the population.
            alive_count: Cars still alive when the generation ended.
            elapsed_seconds: Wall-clock seconds the generation took.
            artifact_paths: Optional label → path mapping for saved files.

        Returns:
            The :class:`GenerationMetrics` object that was recorded.

        Raises:
            ValueError: If *fitnesses* is empty.
        """
        if not fitnesses:
            raise ValueError("fitnesses must not be empty.")

        metrics = GenerationMetrics(
            generation=generation,
            alive_count=alive_count,
            best_fitness=max(fitnesses),
            avg_fitness=statistics.mean(fitnesses),
            elapsed_seconds=elapsed_seconds,
            track_name=self.track_name,
            population_size=len(fitnesses),
            artifact_paths=artifact_paths or {},
        )
        self._generations.append(metrics)
        return metrics

    def record_run_summary(
        self,
        neat_config_path: str = "",
        genome_path: str = "",
    ) -> RunSummary:
        """Build and return a :class:`RunSummary` from all recorded generations.

        Args:
            neat_config_path: Path to the NEAT config used for this run.
            genome_path: Path to the saved best genome file (if any).

        Returns:
            A :class:`RunSummary` aggregating all recorded generations.

        Raises:
            ValueError: If no generations have been recorded yet.
        """
        if not self._generations:
            raise ValueError("No generations recorded yet.")

        total_elapsed = sum(g.elapsed_seconds for g in self._generations)
        peak_fitness = max(g.best_fitness for g in self._generations)
        final_avg = self._generations[-1].avg_fitness

        return RunSummary(
            total_generations=len(self._generations),
            total_elapsed_seconds=total_elapsed,
            peak_fitness=peak_fitness,
            final_avg_fitness=final_avg,
            track_name=self.track_name,
            neat_config_path=neat_config_path,
            genome_path=genome_path,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise all recorded generations to a list-of-dicts structure.

        Returns:
            ``{"track": ..., "generations": [...]}``
        """
        return {
            "track": self.track_name,
            "generations": [g.to_dict() for g in self._generations],
        }

    def to_rows(self) -> list[list[Any]]:
        """Return all generations as flat rows for CSV export.

        The first row is the header (see :attr:`CSV_HEADERS`).

        Returns:
            List of rows where ``rows[0]`` is the header.
        """
        return [self.CSV_HEADERS] + [g.to_row() for g in self._generations]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def generation_count(self) -> int:
        """Number of generations recorded so far."""
        return len(self._generations)

    @property
    def generations(self) -> list[GenerationMetrics]:
        """Read-only view of all recorded :class:`GenerationMetrics`."""
        return list(self._generations)

    def best_generation(self) -> GenerationMetrics | None:
        """Return the generation with the highest best_fitness, or ``None``."""
        if not self._generations:
            return None
        return max(self._generations, key=lambda g: g.best_fitness)
