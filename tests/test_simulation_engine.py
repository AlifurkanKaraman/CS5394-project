"""Tests for ai_car_sim.simulation.engine (headless, no pygame display)."""

import pytest
from unittest.mock import MagicMock

from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.domain.track import Track
from ai_car_sim.core.car import Car
from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

class OpenSurface:
    """All pixels are track colour — no collisions."""
    def __init__(self, w=1920, h=1080):
        self._w, self._h = w, h
    def get_at(self, pos): return (0, 0, 0, 255)
    def get_size(self): return (self._w, self._h)


class BorderSurface(OpenSurface):
    """Every pixel is border colour — instant crash."""
    def get_at(self, pos): return (255, 255, 255, 255)


class ConstantDriver(DriverInterface):
    def __init__(self, action=Action.SPEED_UP):
        self._action = action
    def decide_action(self, inputs, state):
        return self._action


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


def _engine(headless=True, **kw) -> SimulationEngine:
    return SimulationEngine(_cfg(**kw), _track(), headless=headless)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_engine_starts_uninitialised():
    eng = _engine()
    assert not eng._initialized


def test_generation_starts_at_zero():
    assert _engine().generation == 0


def test_cars_empty_before_creation():
    assert _engine().cars == []


# ---------------------------------------------------------------------------
# initialize / shutdown
# ---------------------------------------------------------------------------

def test_initialize_sets_flag(tmp_path):
    eng = _engine()
    eng.initialize()
    assert eng._initialized


def test_initialize_idempotent():
    eng = _engine()
    eng.initialize()
    eng.initialize()  # second call should not raise
    assert eng._initialized


def test_shutdown_clears_flag():
    eng = _engine()
    eng.initialize()
    eng.shutdown()
    assert not eng._initialized


# ---------------------------------------------------------------------------
# step – pure orchestration, no pygame
# ---------------------------------------------------------------------------

def test_step_returns_alive_count():
    eng = _engine()
    eng.initialize()
    cars = [Car(_cfg()) for _ in range(3)]
    for c in cars:
        c.reset(500.0, 500.0)
    drivers = [ConstantDriver() for _ in cars]
    alive = eng.step(cars, drivers, OpenSurface())
    assert alive == 3


def test_step_crashes_cars_on_border_surface():
    eng = _engine()
    eng.initialize()
    cars = [Car(_cfg()) for _ in range(2)]
    for c in cars:
        c.reset(500.0, 500.0)
    drivers = [ConstantDriver() for _ in cars]
    alive = eng.step(cars, drivers, BorderSurface())
    assert alive == 0


def test_step_skips_already_crashed_cars():
    eng = _engine()
    eng.initialize()
    car = Car(_cfg())
    car.reset(500.0, 500.0)
    car.state.mark_crashed()
    alive = eng.step([car], [ConstantDriver()], OpenSurface())
    assert alive == 0
    assert car.state.time_steps == 0  # no update happened


def test_step_applies_driver_action():
    eng = _engine()
    eng.initialize()
    car = Car(_cfg(default_speed=20.0))
    car.reset(500.0, 500.0)
    initial_speed = car.state.speed
    # SPEED_UP should increase speed
    eng.step([car], [ConstantDriver(Action.SPEED_UP)], OpenSurface())
    assert car.state.speed > initial_speed


def test_step_empty_lists_returns_zero():
    eng = _engine()
    eng.initialize()
    assert eng.step([], [], OpenSurface()) == 0


# ---------------------------------------------------------------------------
# create_replay_car
# ---------------------------------------------------------------------------

def test_create_replay_car_returns_car():
    eng = _engine()
    eng.initialize()
    car = eng.create_replay_car(ConstantDriver())
    assert isinstance(car, Car)


def test_create_replay_car_resets_to_spawn():
    eng = _engine()
    eng.initialize()
    car = eng.create_replay_car(ConstantDriver())
    assert car.state.x == pytest.approx(500.0)
    assert car.state.y == pytest.approx(500.0)


def test_create_replay_car_sets_internal_lists():
    eng = _engine()
    eng.initialize()
    eng.create_replay_car(ConstantDriver())
    assert len(eng._cars) == 1
    assert len(eng._drivers) == 1


# ---------------------------------------------------------------------------
# evaluate_genomes (EvaluationEngine protocol)
# ---------------------------------------------------------------------------

def test_evaluate_genomes_assigns_fitness():
    eng = _engine()
    eng.initialize()
    # Patch _step_all to immediately end the generation
    eng._game_map = OpenSurface()

    genome1, genome2 = MagicMock(), MagicMock()
    genome1.fitness = 0.0
    genome2.fitness = 0.0

    neat_cfg = MagicMock()

    import neat as _neat
    with pytest.MonkeyPatch().context() as mp:
        # Stub FeedForwardNetwork.create to return a fixed-output network
        class FakeNet:
            def activate(self, inputs): return [0.0, 0.0, 0.0, 1.0]

        mp.setattr(_neat.nn.FeedForwardNetwork, "create", lambda g, c: FakeNet())
        eng.evaluate_genomes([(0, genome1), (1, genome2)], neat_cfg)

    # Both genomes should have fitness assigned (≥ 0)
    assert genome1.fitness >= 0.0
    assert genome2.fitness >= 0.0


def test_evaluate_genomes_increments_generation():
    eng = _engine()
    eng.initialize()
    eng._game_map = OpenSurface()

    genome = MagicMock()
    genome.fitness = 0.0
    neat_cfg = MagicMock()

    import neat as _neat
    class FakeNet:
        def activate(self, inputs): return [0.0, 0.0, 0.0, 1.0]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(_neat.nn.FeedForwardNetwork, "create", lambda g, c: FakeNet())
        eng.evaluate_genomes([(0, genome)], neat_cfg)

    assert eng.generation == 1
