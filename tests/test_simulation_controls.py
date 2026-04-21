"""Tests for the new simulation controls: speed, spawn spread,
generation overlay, and Q/ESC key semantics."""

import math
import pytest
from unittest.mock import MagicMock

from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.domain.track import Track
from ai_car_sim.simulation.engine import SimulationEngine, _SPEED_STEPS, _DEFAULT_SPEED_IDX
from ai_car_sim.ui.generation_overlay import GenerationOverlay
from ai_car_sim.ui.hud_view import HudMetrics, HudView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> SimulationConfig:
    defaults = dict(
        screen_width=1920, screen_height=1080,
        car_size_x=60, car_size_y=60,
        default_speed=20.0, min_speed=12.0, max_speed=40.0,
        border_color=(255, 255, 255, 255),
        radar_angles=[-90, -45, 0, 45, 90],
        max_radar_distance=300,
        fps=60, steps_per_generation=50,
    )
    defaults.update(kw)
    return SimulationConfig(**defaults)


def _track(**kw) -> Track:
    defaults = dict(
        name="Test", map_image_path="assets/maps/map.png",
        spawn_position=(500.0, 500.0), spawn_angle=0.0,
    )
    defaults.update(kw)
    return Track(**defaults)


def _engine(**kw) -> SimulationEngine:
    return SimulationEngine(_cfg(), _track(), headless=True, **kw)


# ---------------------------------------------------------------------------
# Speed control – pure property methods
# ---------------------------------------------------------------------------

def test_default_speed_is_1x():
    eng = _engine()
    assert eng.sim_speed == pytest.approx(1.0)


def test_speed_up_increases():
    eng = _engine()
    new_speed = eng.speed_up()
    assert new_speed > 1.0


def test_speed_down_decreases():
    eng = _engine()
    new_speed = eng.speed_down()
    assert new_speed < 1.0


def test_speed_reset_returns_1x():
    eng = _engine()
    eng.speed_up()
    eng.speed_up()
    result = eng.speed_reset()
    assert result == pytest.approx(1.0)


def test_speed_up_clamps_at_max():
    eng = _engine()
    for _ in range(20):
        eng.speed_up()
    assert eng.sim_speed == pytest.approx(_SPEED_STEPS[-1])


def test_speed_down_clamps_at_min():
    eng = _engine()
    for _ in range(20):
        eng.speed_down()
    assert eng.sim_speed == pytest.approx(_SPEED_STEPS[0])


def test_speed_steps_are_positive():
    for s in _SPEED_STEPS:
        assert s > 0


def test_default_speed_idx_points_to_1x():
    assert _SPEED_STEPS[_DEFAULT_SPEED_IDX] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Spawn spread
# ---------------------------------------------------------------------------

class OpenSurface:
    def __init__(self, w=1920, h=1080):
        self._w, self._h = w, h
    def get_at(self, pos): return (0, 0, 0, 255)
    def get_size(self): return (self._w, self._h)


def test_spawn_spread_cars_have_different_positions():
    eng = _engine()
    eng.initialize()
    eng._game_map = OpenSurface()

    genomes = [(i, MagicMock()) for i in range(5)]
    for _, g in genomes:
        g.fitness = 0.0

    import neat as _neat
    class FakeNet:
        def activate(self, inputs): return [0.0, 0.0, 0.0, 1.0]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(_neat.nn.FeedForwardNetwork, "create", lambda g, c: FakeNet())
        eng.create_cars_for_genomes(genomes, MagicMock())

    positions = [(c.state.x, c.state.y) for c in eng.cars]
    # All positions should be distinct
    assert len(set(positions)) == len(positions)


def test_spawn_spread_all_same_angle():
    eng = _engine()
    eng.initialize()
    eng._game_map = OpenSurface()

    genomes = [(i, MagicMock()) for i in range(4)]
    for _, g in genomes:
        g.fitness = 0.0

    import neat as _neat
    class FakeNet:
        def activate(self, inputs): return [0.0, 0.0, 0.0, 1.0]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(_neat.nn.FeedForwardNetwork, "create", lambda g, c: FakeNet())
        eng.create_cars_for_genomes(genomes, MagicMock())

    angles = [c.state.angle for c in eng.cars]
    assert all(a == pytest.approx(angles[0]) for a in angles)


def test_single_car_spawns_at_exact_position():
    eng = _engine()
    eng.initialize()
    eng._game_map = OpenSurface()

    genomes = [(0, MagicMock())]
    genomes[0][1].fitness = 0.0

    import neat as _neat
    class FakeNet:
        def activate(self, inputs): return [0.0, 0.0, 0.0, 1.0]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(_neat.nn.FeedForwardNetwork, "create", lambda g, c: FakeNet())
        eng.create_cars_for_genomes(genomes, MagicMock())

    car = eng.cars[0]
    assert car.state.x == pytest.approx(500.0)
    assert car.state.y == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Generation overlay
# ---------------------------------------------------------------------------

def test_overlay_inactive_initially():
    ov = GenerationOverlay()
    assert ov.active is False


def test_overlay_active_after_show():
    ov = GenerationOverlay()
    ov.show(1)
    assert ov.active is True


def test_overlay_expires_after_full_duration():
    ov = GenerationOverlay(display_duration=0.5, fade_duration=0.1)
    ov.show(1)
    # Tick past total duration
    import pygame
    pygame.init()
    screen = pygame.Surface((100, 100))
    ov.draw(screen, dt=1.0)
    assert ov.active is False
    pygame.quit()


def test_overlay_text_contains_generation_number():
    ov = GenerationOverlay()
    ov.show(7)
    assert "7" in ov._text


def test_overlay_show_resets_timer():
    ov = GenerationOverlay(display_duration=1.0, fade_duration=0.5)
    ov.show(1)
    ov._remaining = 0.1
    ov.show(2)  # re-trigger
    assert ov._remaining == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# HudMetrics – sim_speed field
# ---------------------------------------------------------------------------

def test_hud_metrics_default_speed():
    m = HudMetrics()
    assert m.sim_speed == pytest.approx(1.0)


def test_hud_metrics_custom_speed():
    m = HudMetrics(sim_speed=4.0)
    assert m.sim_speed == pytest.approx(4.0)


def test_hud_build_lines_shows_speed():
    hud = HudView()
    lines = hud._build_lines(HudMetrics(sim_speed=2.0))
    texts = [t for t, _ in lines]
    assert any("2.0" in t for t in texts)


def test_hud_build_lines_shows_1x_speed():
    hud = HudView()
    lines = hud._build_lines(HudMetrics(sim_speed=1.0))
    texts = [t for t, _ in lines]
    assert any("1.0" in t for t in texts)


# ---------------------------------------------------------------------------
# Engine flags
# ---------------------------------------------------------------------------

def test_return_to_menu_false_initially():
    assert _engine().return_to_menu is False


def test_skip_generation_false_initially():
    assert _engine()._skip_generation is False


def test_engine_exposes_generation_overlay():
    eng = _engine()
    assert isinstance(eng._gen_overlay, GenerationOverlay)
