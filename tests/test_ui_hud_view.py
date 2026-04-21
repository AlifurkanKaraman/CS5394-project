"""Tests for ai_car_sim.ui.hud_view (pure layout helpers only – no pygame)."""

import pytest
from ai_car_sim.ui.hud_view import HudView, HudMetrics, SimMode


def _metrics(**kwargs) -> HudMetrics:
    defaults = dict(
        generation=5,
        alive_count=12,
        best_fitness=3.14,
        track_name="Test Track",
        mode=SimMode.TRAINING,
    )
    defaults.update(kwargs)
    return HudMetrics(**defaults)


# ---------------------------------------------------------------------------
# HudMetrics defaults
# ---------------------------------------------------------------------------

def test_hud_metrics_defaults():
    m = HudMetrics()
    assert m.generation == 0
    assert m.alive_count == 0
    assert m.best_fitness == 0.0
    assert m.mode is SimMode.TRAINING
    assert m.avg_fitness is None
    assert m.best_speed is None
    assert m.elapsed_seconds is None


# ---------------------------------------------------------------------------
# SimMode
# ---------------------------------------------------------------------------

def test_sim_mode_members():
    assert SimMode.TRAINING
    assert SimMode.REPLAY
    assert SimMode.MANUAL


# ---------------------------------------------------------------------------
# _build_lines – pure, no pygame
# ---------------------------------------------------------------------------

def _lines(metrics: HudMetrics) -> list[tuple[str, object]]:
    hud = HudView()
    return hud._build_lines(metrics)


def test_build_lines_contains_generation():
    lines = _lines(_metrics(generation=7))
    texts = [t for t, _ in lines]
    assert any("7" in t for t in texts)


def test_build_lines_contains_alive_count():
    lines = _lines(_metrics(alive_count=42))
    texts = [t for t, _ in lines]
    assert any("42" in t for t in texts)


def test_build_lines_contains_best_fitness():
    lines = _lines(_metrics(best_fitness=9.99))
    texts = [t for t, _ in lines]
    assert any("9.99" in t for t in texts)


def test_build_lines_contains_mode_training():
    lines = _lines(_metrics(mode=SimMode.TRAINING))
    texts = [t for t, _ in lines]
    assert any("Training" in t for t in texts)


def test_build_lines_contains_mode_replay():
    lines = _lines(_metrics(mode=SimMode.REPLAY))
    texts = [t for t, _ in lines]
    assert any("Replay" in t for t in texts)


def test_build_lines_contains_track_name():
    lines = _lines(_metrics(track_name="Circuit Alpha"))
    texts = [t for t, _ in lines]
    assert any("Circuit Alpha" in t for t in texts)


def test_build_lines_omits_track_when_empty():
    lines = _lines(_metrics(track_name=""))
    texts = [t for t, _ in lines]
    assert not any("Track" in t for t in texts)


def test_build_lines_includes_avg_fitness_when_set():
    lines = _lines(_metrics(avg_fitness=1.23))
    texts = [t for t, _ in lines]
    assert any("1.23" in t for t in texts)


def test_build_lines_omits_avg_fitness_when_none():
    lines = _lines(_metrics(avg_fitness=None))
    texts = [t for t, _ in lines]
    assert not any("Avg" in t for t in texts)


def test_build_lines_includes_speed_when_set():
    lines = _lines(_metrics(best_speed=25.5))
    texts = [t for t, _ in lines]
    assert any("25.5" in t for t in texts)


def test_build_lines_includes_elapsed_when_set():
    lines = _lines(_metrics(elapsed_seconds=12.3))
    texts = [t for t, _ in lines]
    assert any("12.3" in t for t in texts)


def test_build_lines_minimum_four_entries():
    # generation, alive, best_fit, mode are always present
    lines = _lines(_metrics())
    assert len(lines) >= 4


def test_build_lines_all_entries_are_tuples():
    for item in _lines(_metrics()):
        assert isinstance(item, tuple)
        assert len(item) == 2


# ---------------------------------------------------------------------------
# panel_rect – pure static helper
# ---------------------------------------------------------------------------

def test_panel_rect_height_grows_with_lines():
    _, _, _, h1 = HudView.panel_rect(4)
    _, _, _, h2 = HudView.panel_rect(8)
    assert h2 > h1


def test_panel_rect_position():
    x, y, _, _ = HudView.panel_rect(4)
    assert x >= 0
    assert y >= 0


def test_panel_rect_positive_dimensions():
    _, _, w, h = HudView.panel_rect(1)
    assert w > 0
    assert h > 0
