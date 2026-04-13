"""Tests for ai_car_sim.persistence.save_load."""

import csv
import json
import pickle
import pytest
from dataclasses import dataclass
from pathlib import Path

from ai_car_sim.persistence.save_load import (
    ensure_output_dirs,
    save_best_genome,
    load_best_genome,
    save_metrics_json,
    load_metrics_json,
    save_metrics_csv,
    load_metrics_csv,
)


# ---------------------------------------------------------------------------
# Picklable genome stub
# ---------------------------------------------------------------------------

@dataclass
class FakeGenome:
    fitness: float = 7.5
    key: int = 1


# ---------------------------------------------------------------------------
# ensure_output_dirs
# ---------------------------------------------------------------------------

def test_ensure_output_dirs_creates_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_output_dirs(target)
    assert target.exists()
    assert target.is_dir()
    assert result[0] == target.resolve()


def test_ensure_output_dirs_multiple(tmp_path):
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    ensure_output_dirs(d1, d2)
    assert d1.exists()
    assert d2.exists()


def test_ensure_output_dirs_existing_is_noop(tmp_path):
    ensure_output_dirs(tmp_path)  # already exists – should not raise
    assert tmp_path.exists()


def test_ensure_output_dirs_returns_resolved_paths(tmp_path):
    result = ensure_output_dirs(tmp_path)
    assert result[0].is_absolute()


# ---------------------------------------------------------------------------
# save_best_genome / load_best_genome
# ---------------------------------------------------------------------------

def test_save_and_load_genome_roundtrip(tmp_path):
    genome = FakeGenome(fitness=42.0)
    path = tmp_path / "best.pkl"
    result = save_best_genome(genome, path)
    assert result is not None
    loaded = load_best_genome(path)
    assert loaded.fitness == 42.0


def test_save_genome_creates_parent_dirs(tmp_path):
    path = tmp_path / "sub" / "dir" / "genome.pkl"
    save_best_genome(FakeGenome(), path)
    assert path.exists()


def test_save_genome_returns_none_on_failure(tmp_path):
    # Pass a non-picklable object
    import threading
    result = save_best_genome(threading.Lock(), tmp_path / "bad.pkl")
    assert result is None


def test_load_genome_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_best_genome(tmp_path / "missing.pkl")


def test_load_genome_corrupt_file_raises(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Failed to deserialise"):
        load_best_genome(bad)


def test_save_genome_returns_resolved_path(tmp_path):
    path = tmp_path / "genome.pkl"
    result = save_best_genome(FakeGenome(), path)
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# save_metrics_json / load_metrics_json
# ---------------------------------------------------------------------------

def test_save_and_load_json_roundtrip(tmp_path):
    data = {"track": "Alpha", "generations": [{"gen": 1, "fitness": 3.5}]}
    path = tmp_path / "metrics.json"
    save_metrics_json(data, path)
    loaded = load_metrics_json(path)
    assert loaded["track"] == "Alpha"
    assert loaded["generations"][0]["fitness"] == 3.5


def test_save_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "metrics.json"
    save_metrics_json({"x": 1}, path)
    assert path.exists()


def test_save_json_returns_none_on_failure(tmp_path):
    # Pass a non-serialisable object
    result = save_metrics_json({"bad": object()}, tmp_path / "out.json")
    assert result is None


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics_json(tmp_path / "missing.json")


def test_load_json_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_metrics_json(bad)


def test_save_json_is_indented(tmp_path):
    path = tmp_path / "m.json"
    save_metrics_json({"a": 1}, path, indent=2)
    raw = path.read_text()
    assert "\n" in raw  # indented output has newlines


# ---------------------------------------------------------------------------
# save_metrics_csv / load_metrics_csv
# ---------------------------------------------------------------------------

def test_save_and_load_csv_roundtrip(tmp_path):
    rows = [["gen", "fitness"], ["1", "3.5"], ["2", "7.0"]]
    path = tmp_path / "metrics.csv"
    save_metrics_csv(rows, path)
    loaded = load_metrics_csv(path)
    assert loaded[0] == ["gen", "fitness"]
    assert loaded[1] == ["1", "3.5"]
    assert loaded[2] == ["2", "7.0"]


def test_save_csv_creates_parent_dirs(tmp_path):
    path = tmp_path / "sub" / "metrics.csv"
    save_metrics_csv([["h1", "h2"], ["1", "2"]], path)
    assert path.exists()


def test_save_csv_returns_none_on_failure(tmp_path, monkeypatch):
    # Make the directory read-only to force a write failure
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(0o444)
    result = save_metrics_csv([["a"]], ro_dir / "out.csv", mkdir=False)
    assert result is None
    ro_dir.chmod(0o755)  # restore for cleanup


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics_csv(tmp_path / "missing.csv")


def test_save_csv_returns_resolved_path(tmp_path):
    path = tmp_path / "m.csv"
    result = save_metrics_csv([["h"], ["v"]], path)
    assert result.is_absolute()


def test_save_csv_row_count(tmp_path):
    rows = [["h"]] + [[str(i)] for i in range(5)]
    path = tmp_path / "m.csv"
    save_metrics_csv(rows, path)
    loaded = load_metrics_csv(path)
    assert len(loaded) == 6  # header + 5 data rows
