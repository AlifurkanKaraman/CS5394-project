"""Focused tests for ai_car_sim.persistence.save_load.

Covers directory creation, genome pickle round-trips, JSON/CSV
serialisation, error handling, and integration with RunMetricsCollector.
"""

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
from ai_car_sim.analytics.run_metrics import RunMetricsCollector


# ------------------------------------------------------------------
# Picklable genome stub
# ------------------------------------------------------------------

@dataclass
class FakeGenome:
    fitness: float = 5.0
    key: int = 0


# ------------------------------------------------------------------
# ensure_output_dirs
# ------------------------------------------------------------------

def test_creates_nested_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    ensure_output_dirs(target)
    assert target.is_dir()


def test_existing_dir_no_error(tmp_path):
    ensure_output_dirs(tmp_path)  # already exists


def test_returns_resolved_absolute_paths(tmp_path):
    result = ensure_output_dirs(tmp_path / "x")
    assert result[0].is_absolute()


def test_multiple_dirs_created(tmp_path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    ensure_output_dirs(d1, d2)
    assert d1.is_dir() and d2.is_dir()


# ------------------------------------------------------------------
# save_best_genome / load_best_genome
# ------------------------------------------------------------------

def test_genome_round_trip(tmp_path):
    g = FakeGenome(fitness=42.0)
    path = tmp_path / "genome.pkl"
    save_best_genome(g, path)
    loaded = load_best_genome(path)
    assert loaded.fitness == 42.0


def test_save_genome_auto_creates_parents(tmp_path):
    path = tmp_path / "sub" / "genome.pkl"
    save_best_genome(FakeGenome(), path)
    assert path.exists()


def test_save_genome_returns_path_on_success(tmp_path):
    result = save_best_genome(FakeGenome(), tmp_path / "g.pkl")
    assert result is not None
    assert result.is_absolute()


def test_save_genome_returns_none_on_unpicklable(tmp_path):
    import threading
    result = save_best_genome(threading.Lock(), tmp_path / "bad.pkl")
    assert result is None


def test_load_genome_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_best_genome(tmp_path / "missing.pkl")


def test_load_genome_corrupt_raises(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Failed to deserialise"):
        load_best_genome(bad)


def test_load_genome_accepts_string_path(tmp_path):
    path = tmp_path / "g.pkl"
    save_best_genome(FakeGenome(), path)
    loaded = load_best_genome(str(path))
    assert loaded is not None


# ------------------------------------------------------------------
# save_metrics_json / load_metrics_json
# ------------------------------------------------------------------

def test_json_round_trip(tmp_path):
    data = {"track": "Alpha", "gen": 1, "fitness": 3.14}
    path = tmp_path / "m.json"
    save_metrics_json(data, path)
    loaded = load_metrics_json(path)
    assert loaded["track"] == "Alpha"
    assert loaded["fitness"] == pytest.approx(3.14)


def test_json_auto_creates_parents(tmp_path):
    path = tmp_path / "nested" / "m.json"
    save_metrics_json({"x": 1}, path)
    assert path.exists()


def test_json_is_indented(tmp_path):
    path = tmp_path / "m.json"
    save_metrics_json({"a": 1}, path, indent=2)
    assert "\n" in path.read_text()


def test_json_returns_none_on_non_serialisable(tmp_path):
    result = save_metrics_json({"bad": object()}, tmp_path / "m.json")
    assert result is None


def test_load_json_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics_json(tmp_path / "missing.json")


def test_load_json_invalid_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_metrics_json(bad)


def test_json_list_round_trip(tmp_path):
    data = [1, 2, 3]
    path = tmp_path / "list.json"
    save_metrics_json(data, path)
    assert load_metrics_json(path) == [1, 2, 3]


# ------------------------------------------------------------------
# save_metrics_csv / load_metrics_csv
# ------------------------------------------------------------------

def test_csv_round_trip(tmp_path):
    rows = [["gen", "fitness"], ["1", "3.5"], ["2", "7.0"]]
    path = tmp_path / "m.csv"
    save_metrics_csv(rows, path)
    loaded = load_metrics_csv(path)
    assert loaded[0] == ["gen", "fitness"]
    assert loaded[2] == ["2", "7.0"]


def test_csv_auto_creates_parents(tmp_path):
    path = tmp_path / "sub" / "m.csv"
    save_metrics_csv([["h"], ["v"]], path)
    assert path.exists()


def test_csv_row_count(tmp_path):
    rows = [["h"]] + [[str(i)] for i in range(10)]
    path = tmp_path / "m.csv"
    save_metrics_csv(rows, path)
    assert len(load_metrics_csv(path)) == 11


def test_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics_csv(tmp_path / "missing.csv")


def test_csv_returns_none_on_write_failure(tmp_path, monkeypatch):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o444)
    result = save_metrics_csv([["h"]], ro / "m.csv", mkdir=False)
    assert result is None
    ro.chmod(0o755)


# ------------------------------------------------------------------
# Integration: RunMetricsCollector → persistence
# ------------------------------------------------------------------

def test_collector_to_rows_saved_and_loaded(tmp_path):
    col = RunMetricsCollector(track_name="Beta")
    col.record_generation(1, [1.0, 2.0, 3.0], alive_count=3, elapsed_seconds=1.5)
    col.record_generation(2, [4.0, 5.0, 6.0], alive_count=2, elapsed_seconds=2.0)

    path = tmp_path / "gens.csv"
    save_metrics_csv(col.to_rows(), path)
    loaded = load_metrics_csv(path)

    assert loaded[0] == RunMetricsCollector.CSV_HEADERS
    assert len(loaded) == 3  # header + 2 data rows
    assert loaded[1][0] == "1"   # generation 1
    assert loaded[2][0] == "2"   # generation 2


def test_collector_to_dict_saved_and_loaded(tmp_path):
    col = RunMetricsCollector(track_name="Gamma")
    col.record_generation(1, [2.0, 4.0], alive_count=5, elapsed_seconds=1.0)

    path = tmp_path / "metrics.json"
    save_metrics_json(col.to_dict(), path)
    loaded = load_metrics_json(path)

    assert loaded["track"] == "Gamma"
    assert len(loaded["generations"]) == 1
    assert loaded["generations"][0]["generation"] == 1
