"""Replay genome loader for the AI car simulation.

Provides a dedicated path for loading a previously saved best genome and
rebuilding a :class:`~ai_car_sim.ai.neat_driver.NeatDriver` from it,
without touching the training orchestration code.

Serialisation uses :mod:`pickle` (same format as ``neat-python``'s own
checkpointing) so saved genomes are directly compatible.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import neat  # type: ignore[import]

from ai_car_sim.ai.neat_driver import NeatDriver
from ai_car_sim.domain.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReplayLoadError(Exception):
    """Raised when a genome file cannot be loaded or is incompatible."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_best_genome(file_path: str | Path) -> Any:
    """Deserialise a genome from a pickle file.

    Args:
        file_path: Path to the ``.pkl`` file written by the persistence
            layer (or ``neat-python``'s checkpointer).

    Returns:
        The deserialised genome object (typically a
        :class:`neat.DefaultGenome`).

    Raises:
        ReplayLoadError: If the file does not exist, cannot be read, or
            the pickle data is corrupt / incompatible.
    """
    path = Path(file_path)
    if not path.exists():
        raise ReplayLoadError(f"Genome file not found: {path}")

    try:
        with path.open("rb") as fh:
            genome = pickle.load(fh)  # noqa: S301 – trusted local file
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise ReplayLoadError(
            f"Failed to deserialise genome from {path}: {exc}"
        ) from exc

    logger.info("Loaded genome from %s (fitness=%s)", path, getattr(genome, "fitness", "?"))
    return genome


def build_driver_from_saved_genome(
    genome: Any,
    neat_config: neat.Config,
    expected_outputs: int | None = 4,
) -> NeatDriver:
    """Reconstruct a :class:`NeatDriver` from a saved genome and NEAT config.

    Args:
        genome: A deserialised :class:`neat.DefaultGenome` (e.g. from
            :func:`load_best_genome`).
        neat_config: The NEAT configuration used when the genome was
            trained.  Must match the genome's node/connection structure.
        expected_outputs: Number of output neurons to validate against.
            Pass ``None`` to skip validation.

    Returns:
        A :class:`NeatDriver` ready to be passed to the simulation engine.

    Raises:
        ReplayLoadError: If the network cannot be built from the genome.
    """
    try:
        network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    except Exception as exc:
        raise ReplayLoadError(
            f"Could not build FeedForwardNetwork from genome: {exc}"
        ) from exc

    return NeatDriver(network, expected_outputs=expected_outputs)


def load_replay_driver(
    genome_path: str | Path,
    neat_config_path: str | Path,
    expected_outputs: int | None = 4,
) -> NeatDriver:
    """Convenience one-call loader: genome file + NEAT config → driver.

    Combines :func:`load_best_genome`, NEAT config loading, and
    :func:`build_driver_from_saved_genome` into a single call for the
    common replay use-case.

    Args:
        genome_path: Path to the pickled genome file.
        neat_config_path: Path to the NEAT ``.cfg`` / ``.txt`` config file.
        expected_outputs: Passed through to :func:`build_driver_from_saved_genome`.

    Returns:
        A :class:`NeatDriver` ready for replay.

    Raises:
        ReplayLoadError: If any step fails (missing files, corrupt data,
            incompatible genome/config).
    """
    neat_cfg_path = Path(neat_config_path)
    if not neat_cfg_path.exists():
        raise ReplayLoadError(f"NEAT config not found: {neat_cfg_path}")

    try:
        neat_config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_cfg_path),
        )
    except Exception as exc:
        raise ReplayLoadError(
            f"Failed to load NEAT config from {neat_cfg_path}: {exc}"
        ) from exc

    genome = load_best_genome(genome_path)
    return build_driver_from_saved_genome(genome, neat_config, expected_outputs)


def validate_saved_run(genome_path: str | Path) -> dict[str, Any]:
    """Inspect a saved genome file and return a summary without fully loading it.

    Useful for quick sanity checks (e.g. in a CLI or test harness) before
    committing to a full replay.

    Args:
        genome_path: Path to the pickled genome file.

    Returns:
        Dictionary with keys:
        - ``"path"``: resolved absolute path string
        - ``"exists"``: bool
        - ``"size_bytes"``: file size (0 if missing)
        - ``"fitness"``: genome fitness if loadable, else ``None``
        - ``"error"``: error message string if loading failed, else ``None``
    """
    path = Path(genome_path).resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "fitness": None,
        "error": None,
    }
    if result["exists"]:
        try:
            genome = load_best_genome(path)
            result["fitness"] = getattr(genome, "fitness", None)
        except ReplayLoadError as exc:
            result["error"] = str(exc)
    return result
