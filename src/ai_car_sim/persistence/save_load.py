"""Persistence helpers for the AI car simulation.

Saves and loads best genomes, run metrics JSON, and CSV exports.
All I/O uses the standard library only (pickle, json, csv, pathlib).

Design rules:
- A failed *save* logs a warning and returns ``None`` — it is non-fatal
  so a crash mid-training doesn't lose the whole run.
- A failed *load* raises a descriptive exception so callers know
  immediately that the file is missing or corrupt.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(*paths: str | Path) -> list[Path]:
    """Create one or more directories (and any missing parents).

    Args:
        *paths: Directory paths to create.  Existing directories are
            left unchanged.

    Returns:
        List of resolved :class:`~pathlib.Path` objects that were ensured.
    """
    created: list[Path] = []
    for p in paths:
        resolved = Path(p).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        created.append(resolved)
        logger.debug("Ensured directory: %s", resolved)
    return created


# ---------------------------------------------------------------------------
# Genome persistence
# ---------------------------------------------------------------------------

def save_best_genome(
    genome: Any,
    path: str | Path,
    *,
    mkdir: bool = True,
) -> Path | None:
    """Serialise *genome* to a pickle file.

    A failed save is non-fatal: the exception is logged as a warning and
    ``None`` is returned so training can continue.

    Args:
        genome: Any picklable object (typically a ``neat.DefaultGenome``).
        path: Destination file path (e.g. ``outputs/best_genome.pkl``).
        mkdir: When ``True``, create parent directories automatically.

    Returns:
        The resolved :class:`~pathlib.Path` on success, or ``None`` on failure.
    """
    dest = Path(path)
    try:
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            pickle.dump(genome, fh)
        logger.info("Saved best genome → %s", dest)
        return dest.resolve()
    except Exception as exc:
        logger.warning("Could not save genome to %s: %s", dest, exc)
        return None


def load_best_genome(path: str | Path) -> Any:
    """Deserialise a genome from a pickle file.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        The deserialised genome object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file exists but cannot be unpickled.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Genome file not found: {src}")
    try:
        with src.open("rb") as fh:
            genome = pickle.load(fh)  # noqa: S301 – trusted local file
        logger.info("Loaded genome ← %s (fitness=%s)", src, getattr(genome, "fitness", "?"))
        return genome
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise ValueError(f"Failed to deserialise genome from {src}: {exc}") from exc


# ---------------------------------------------------------------------------
# Metrics JSON
# ---------------------------------------------------------------------------

def save_metrics_json(
    data: dict[str, Any] | list[Any],
    path: str | Path,
    *,
    mkdir: bool = True,
    indent: int = 2,
) -> Path | None:
    """Write *data* to a JSON file.

    A failed save is non-fatal.

    Args:
        data: JSON-serialisable dict or list (e.g. from
            :meth:`~ai_car_sim.analytics.run_metrics.RunMetricsCollector.to_dict`).
        path: Destination file path.
        mkdir: When ``True``, create parent directories automatically.
        indent: JSON indentation level.

    Returns:
        The resolved :class:`~pathlib.Path` on success, or ``None`` on failure.
    """
    dest = Path(path)
    try:
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=indent), encoding="utf-8")
        logger.info("Saved metrics JSON → %s", dest)
        return dest.resolve()
    except Exception as exc:
        logger.warning("Could not save metrics JSON to %s: %s", dest, exc)
        return None


def load_metrics_json(path: str | Path) -> Any:
    """Load metrics from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Metrics file not found: {src}")
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {src}: {exc}") from exc


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def save_metrics_csv(
    rows: list[list[Any]],
    path: str | Path,
    *,
    mkdir: bool = True,
) -> Path | None:
    """Write *rows* to a CSV file.

    The first row is treated as the header.  A failed save is non-fatal.

    Args:
        rows: List of rows where ``rows[0]`` is the header (e.g. from
            :meth:`~ai_car_sim.analytics.run_metrics.RunMetricsCollector.to_rows`).
        path: Destination file path.
        mkdir: When ``True``, create parent directories automatically.

    Returns:
        The resolved :class:`~pathlib.Path` on success, or ``None`` on failure.
    """
    dest = Path(path)
    try:
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        logger.info("Saved metrics CSV → %s (%d rows)", dest, max(0, len(rows) - 1))
        return dest.resolve()
    except Exception as exc:
        logger.warning("Could not save metrics CSV to %s: %s", dest, exc)
        return None


def load_metrics_csv(path: str | Path) -> list[list[str]]:
    """Load all rows from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        List of rows (each row is a list of strings), including the header.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"CSV file not found: {src}")
    with src.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))
