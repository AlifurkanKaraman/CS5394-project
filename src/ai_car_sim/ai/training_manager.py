"""NEAT training manager for the AI car simulation.

Orchestrates NEAT population setup, reporter registration, generation
callbacks, and genome evaluation.  Moves the training bootstrap that
lived inline in ``newcar.py`` into an explicit, configurable class.

The manager depends on a narrow :class:`EvaluationEngine` protocol so
the real pygame-based engine can be swapped for a headless stub in tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import neat  # type: ignore[import]

from ai_car_sim.domain.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvaluationEngine protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EvaluationEngine(Protocol):
    """Minimal interface the training manager requires from the sim engine.

    The real :class:`~ai_car_sim.simulation.engine.SimulationEngine` must
    satisfy this protocol.  A lightweight stub is sufficient for tests.
    """

    def evaluate_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        """Run one generation and assign ``genome.fitness`` for each genome.

        Args:
            genomes: List of ``(genome_id, genome)`` pairs as passed by
                ``neat.Population.run``.
            neat_config: The active NEAT configuration object.
        """
        ...


# ---------------------------------------------------------------------------
# TrainingManager
# ---------------------------------------------------------------------------

class TrainingManager:
    """Orchestrates NEAT population lifecycle for one training run.

    Typical usage::

        manager = TrainingManager(config, engine)
        manager.load_neat_config()
        manager.create_population()
        winner = manager.run_training()

    Args:
        sim_config: Simulation configuration (paths, generation budget, …).
        engine: Object satisfying :class:`EvaluationEngine` that runs each
            generation and assigns genome fitness values.
        on_generation: Optional callback invoked after every generation with
            the current generation number and best fitness.  Useful for
            progress reporting or early-stopping logic.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        engine: EvaluationEngine,
        on_generation: Callable[[int, float], None] | None = None,
    ) -> None:
        self.sim_config = sim_config
        self.engine = engine
        self.on_generation = on_generation

        self._neat_config: neat.Config | None = None
        self._population: neat.Population | None = None
        self._stats: neat.StatisticsReporter | None = None
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_neat_config(self, path: str | Path | None = None) -> neat.Config:
        """Load and return the NEAT configuration from *path*.

        Args:
            path: Path to the NEAT config file.  Defaults to
                ``sim_config.neat_config_path`` when ``None``.

        Returns:
            The loaded :class:`neat.Config` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        config_path = Path(path or self.sim_config.neat_config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"NEAT config not found: {config_path}"
            )
        self._neat_config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        logger.info("Loaded NEAT config from %s", config_path)
        return self._neat_config

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def create_population(
        self,
        neat_config: neat.Config | None = None,
    ) -> neat.Population:
        """Create a fresh NEAT population and attach standard reporters.

        Attaches :class:`neat.StdOutReporter` and
        :class:`neat.StatisticsReporter` automatically.

        Args:
            neat_config: Config to use.  Falls back to the one loaded by
                :meth:`load_neat_config` when ``None``.

        Returns:
            The new :class:`neat.Population`.

        Raises:
            RuntimeError: If no NEAT config is available.
        """
        cfg = neat_config or self._neat_config
        if cfg is None:
            raise RuntimeError(
                "No NEAT config available. Call load_neat_config() first."
            )
        self._population = neat.Population(cfg)
        self._population.add_reporter(neat.StdOutReporter(True))
        self._stats = neat.StatisticsReporter()
        self._population.add_reporter(self._stats)
        logger.info("Created NEAT population")
        return self._population

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_training(
        self,
        max_generations: int | None = None,
        population: neat.Population | None = None,
    ) -> neat.DefaultGenome:
        """Run the NEAT evolutionary loop and return the best genome.

        Args:
            max_generations: Number of generations to run.  Defaults to
                ``sim_config.max_generations``.
            population: Population to use.  Falls back to the one created
                by :meth:`create_population` when ``None``.

        Returns:
            The best genome found across all generations.

        Raises:
            RuntimeError: If no population is available.
        """
        pop = population or self._population
        if pop is None:
            raise RuntimeError(
                "No population available. Call create_population() first."
            )
        n = max_generations or self.sim_config.max_generations
        winner = pop.run(self._eval_genomes, n)
        logger.info("Training complete. Best fitness: %.4f", winner.fitness)
        return winner

    # ------------------------------------------------------------------
    # Genome evaluation callback (passed to neat.Population.run)
    # ------------------------------------------------------------------

    def _eval_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        """Callback invoked by NEAT each generation.

        Initialises genome fitness to 0, delegates evaluation to the
        engine, then fires the optional :attr:`on_generation` hook.

        Args:
            genomes: List of ``(genome_id, genome)`` pairs.
            neat_config: Active NEAT configuration.
        """
        # Reset fitness before evaluation so stale values from a previous
        # generation never bleed into the current one.
        for _, genome in genomes:
            genome.fitness = 0.0

        self.engine.evaluate_genomes(genomes, neat_config)

        self._generation += 1

        # Push species count to the engine for HUD display (optional — only
        # if the engine exposes set_species_count and the population is ready).
        if self._population is not None and hasattr(self.engine, "set_species_count"):
            try:
                species_count = len(self._population.species.species)
                self.engine.set_species_count(species_count)  # type: ignore[attr-defined]
            except Exception:
                pass  # non-critical; don't break training over HUD data

        if self.on_generation is not None:
            best = max(
                (g.fitness for _, g in genomes if g.fitness is not None),
                default=0.0,
            )
            self.on_generation(self._generation, best)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        """Current generation number (incremented after each eval)."""
        return self._generation

    @property
    def statistics(self) -> neat.StatisticsReporter | None:
        """The attached statistics reporter, or ``None`` before population creation."""
        return self._stats

    def best_genome(self) -> neat.DefaultGenome | None:
        """Return the most-fit genome seen so far, or ``None`` if unavailable."""
        if self._stats is None:
            return None
        return self._stats.best_genome()
