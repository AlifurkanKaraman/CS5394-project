"""Tests for ai_car_sim.analytics.run_metrics."""

import pytest
from ai_car_sim.analytics.run_metrics import (
    GenerationMetrics,
    RunSummary,
    RunMetricsCollector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collector(track="Test Track") -> RunMetricsCollector:
    return RunMetricsCollector(track_name=track)


def _record(collector: RunMetricsCollector, gen=1, fitnesses=None, alive=5, elapsed=1.0):
    return collector.record_generation(
        generation=gen,
        fitnesses=fitnesses or [1.0, 2.0, 3.0],
        alive_count=alive,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# GenerationMetrics
# ---------------------------------------------------------------------------

def test_generation_metrics_fields():
    m = GenerationMetrics(
        generation=3, alive_count=10, best_fitness=5.0,
        avg_fitness=3.0, elapsed_seconds=2.5, track_name="Alpha",
        population_size=20,
    )
    assert m.generation == 3
    assert m.alive_count == 10
    assert m.best_fitness == 5.0
    assert m.avg_fitness == 3.0
    assert m.elapsed_seconds == 2.5
    assert m.population_size == 20


def test_generation_metrics_to_dict_keys():
    m = GenerationMetrics(1, 5, 3.0, 2.0, 1.0)
    d = m.to_dict()
    for key in ("generation", "alive_count", "best_fitness", "avg_fitness", "elapsed_seconds"):
        assert key in d


def test_generation_metrics_to_row_length():
    m = GenerationMetrics(1, 5, 3.0, 2.0, 1.0, track_name="T", population_size=10)
    row = m.to_row()
    assert len(row) == len(RunMetricsCollector.CSV_HEADERS)


def test_generation_metrics_to_row_values():
    m = GenerationMetrics(2, 3, 9.5, 4.5, 0.75, track_name="Beta", population_size=15)
    row = m.to_row()
    assert row[0] == 2          # generation
    assert row[1] == 3          # alive_count
    assert row[2] == 15         # population_size
    assert row[3] == pytest.approx(9.5)
    assert row[4] == pytest.approx(4.5)
    assert row[5] == pytest.approx(0.75)
    assert row[6] == "Beta"


def test_generation_metrics_artifact_paths_default_empty():
    m = GenerationMetrics(1, 0, 0.0, 0.0, 0.0)
    assert m.artifact_paths == {}


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

def test_run_summary_to_dict():
    s = RunSummary(
        total_generations=10, total_elapsed_seconds=120.0,
        peak_fitness=8.0, final_avg_fitness=4.0,
        track_name="Gamma", neat_config_path="cfg.txt", genome_path="best.pkl",
    )
    d = s.to_dict()
    assert d["total_generations"] == 10
    assert d["peak_fitness"] == 8.0
    assert d["genome_path"] == "best.pkl"


# ---------------------------------------------------------------------------
# RunMetricsCollector – record_generation
# ---------------------------------------------------------------------------

def test_record_generation_returns_metrics():
    col = _collector()
    m = _record(col)
    assert isinstance(m, GenerationMetrics)


def test_record_generation_computes_best_fitness():
    col = _collector()
    m = _record(col, fitnesses=[1.0, 5.0, 3.0])
    assert m.best_fitness == pytest.approx(5.0)


def test_record_generation_computes_avg_fitness():
    col = _collector()
    m = _record(col, fitnesses=[2.0, 4.0])
    assert m.avg_fitness == pytest.approx(3.0)


def test_record_generation_stores_population_size():
    col = _collector()
    m = _record(col, fitnesses=[1.0, 2.0, 3.0, 4.0])
    assert m.population_size == 4


def test_record_generation_empty_fitnesses_raises():
    col = _collector()
    with pytest.raises(ValueError, match="empty"):
        col.record_generation(1, [], 0, 1.0)


def test_record_generation_increments_count():
    col = _collector()
    _record(col, gen=1)
    _record(col, gen=2)
    assert col.generation_count == 2


def test_record_generation_stores_track_name():
    col = _collector(track="Circuit X")
    m = _record(col)
    assert m.track_name == "Circuit X"


def test_record_generation_stores_artifact_paths():
    col = _collector()
    m = col.record_generation(1, [1.0], 1, 1.0, artifact_paths={"genome": "out/best.pkl"})
    assert m.artifact_paths["genome"] == "out/best.pkl"


# ---------------------------------------------------------------------------
# RunMetricsCollector – record_run_summary
# ---------------------------------------------------------------------------

def test_record_run_summary_raises_with_no_generations():
    col = _collector()
    with pytest.raises(ValueError, match="No generations"):
        col.record_run_summary()


def test_record_run_summary_total_generations():
    col = _collector()
    _record(col, gen=1)
    _record(col, gen=2)
    summary = col.record_run_summary()
    assert summary.total_generations == 2


def test_record_run_summary_total_elapsed():
    col = _collector()
    _record(col, gen=1, elapsed=3.0)
    _record(col, gen=2, elapsed=5.0)
    summary = col.record_run_summary()
    assert summary.total_elapsed_seconds == pytest.approx(8.0)


def test_record_run_summary_peak_fitness():
    col = _collector()
    _record(col, gen=1, fitnesses=[1.0, 2.0])
    _record(col, gen=2, fitnesses=[3.0, 4.0])
    summary = col.record_run_summary()
    assert summary.peak_fitness == pytest.approx(4.0)


def test_record_run_summary_final_avg_fitness():
    col = _collector()
    _record(col, gen=1, fitnesses=[1.0, 3.0])   # avg=2.0
    _record(col, gen=2, fitnesses=[5.0, 7.0])   # avg=6.0
    summary = col.record_run_summary()
    assert summary.final_avg_fitness == pytest.approx(6.0)


def test_record_run_summary_paths_forwarded():
    col = _collector()
    _record(col)
    summary = col.record_run_summary(neat_config_path="cfg.txt", genome_path="best.pkl")
    assert summary.neat_config_path == "cfg.txt"
    assert summary.genome_path == "best.pkl"


# ---------------------------------------------------------------------------
# to_dict / to_rows
# ---------------------------------------------------------------------------

def test_to_dict_structure():
    col = _collector(track="Delta")
    _record(col, gen=1)
    d = col.to_dict()
    assert d["track"] == "Delta"
    assert len(d["generations"]) == 1


def test_to_rows_first_row_is_header():
    col = _collector()
    _record(col)
    rows = col.to_rows()
    assert rows[0] == RunMetricsCollector.CSV_HEADERS


def test_to_rows_data_rows_count():
    col = _collector()
    _record(col, gen=1)
    _record(col, gen=2)
    rows = col.to_rows()
    assert len(rows) == 3  # header + 2 data rows


def test_to_rows_empty_collector_returns_header_only():
    rows = _collector().to_rows()
    assert rows == [RunMetricsCollector.CSV_HEADERS]


# ---------------------------------------------------------------------------
# best_generation / generations accessor
# ---------------------------------------------------------------------------

def test_best_generation_none_when_empty():
    assert _collector().best_generation() is None


def test_best_generation_returns_highest_fitness():
    col = _collector()
    _record(col, gen=1, fitnesses=[1.0, 2.0])
    _record(col, gen=2, fitnesses=[5.0, 6.0])
    _record(col, gen=3, fitnesses=[3.0, 4.0])
    best = col.best_generation()
    assert best.generation == 2


def test_generations_returns_copy():
    col = _collector()
    _record(col)
    gens = col.generations
    gens.clear()
    assert col.generation_count == 1  # original unaffected
