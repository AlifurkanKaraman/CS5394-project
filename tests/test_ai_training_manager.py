"""Tests for ai_car_sim.ai.training_manager."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.ai.training_manager import TrainingManager, EvaluationEngine


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> SimulationConfig:
    defaults = dict(
        neat_config_path="configs/config.txt",
        max_generations=10,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


class StubEngine:
    """Minimal EvaluationEngine that assigns fixed fitness values."""

    def __init__(self, fitness: float = 1.0) -> None:
        self.calls: list = []
        self._fitness = fitness

    def evaluate_genomes(self, genomes, neat_config) -> None:
        self.calls.append(genomes)
        for _, genome in genomes:
            genome.fitness = self._fitness


def _fake_genome(fitness: float = 0.0):
    g = MagicMock()
    g.fitness = fitness
    return g


def _fake_genomes(n: int = 3) -> list[tuple[int, object]]:
    return [(i, _fake_genome()) for i in range(n)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_initial_generation_is_zero():
    manager = TrainingManager(_cfg(), StubEngine())
    assert manager.generation == 0


def test_statistics_none_before_population_created():
    manager = TrainingManager(_cfg(), StubEngine())
    assert manager.statistics is None


# ---------------------------------------------------------------------------
# load_neat_config
# ---------------------------------------------------------------------------

def test_load_neat_config_raises_if_file_missing(tmp_path):
    manager = TrainingManager(_cfg(neat_config_path=str(tmp_path / "missing.txt")), StubEngine())
    with pytest.raises(FileNotFoundError, match="NEAT config not found"):
        manager.load_neat_config()


def test_load_neat_config_uses_sim_config_path_by_default(tmp_path):
    cfg_path = tmp_path / "neat.cfg"
    cfg_path.write_text("")  # empty file – we'll mock neat.config.Config
    manager = TrainingManager(_cfg(neat_config_path=str(cfg_path)), StubEngine())

    with patch("ai_car_sim.ai.training_manager.neat") as mock_neat:
        mock_neat.config.Config.return_value = MagicMock()
        mock_neat.DefaultGenome = MagicMock()
        mock_neat.DefaultReproduction = MagicMock()
        mock_neat.DefaultSpeciesSet = MagicMock()
        mock_neat.DefaultStagnation = MagicMock()
        result = manager.load_neat_config()

    mock_neat.config.Config.assert_called_once()
    assert result is mock_neat.config.Config.return_value


def test_load_neat_config_explicit_path_overrides_sim_config(tmp_path):
    explicit = tmp_path / "explicit.cfg"
    explicit.write_text("")
    manager = TrainingManager(_cfg(neat_config_path="nonexistent.txt"), StubEngine())

    with patch("ai_car_sim.ai.training_manager.neat") as mock_neat:
        mock_neat.config.Config.return_value = MagicMock()
        mock_neat.DefaultGenome = MagicMock()
        mock_neat.DefaultReproduction = MagicMock()
        mock_neat.DefaultSpeciesSet = MagicMock()
        mock_neat.DefaultStagnation = MagicMock()
        manager.load_neat_config(path=explicit)

    args = mock_neat.config.Config.call_args[0]
    assert str(explicit) in args


# ---------------------------------------------------------------------------
# create_population
# ---------------------------------------------------------------------------

def test_create_population_raises_without_config():
    manager = TrainingManager(_cfg(), StubEngine())
    with pytest.raises(RuntimeError, match="No NEAT config"):
        manager.create_population()


def test_create_population_attaches_reporters():
    manager = TrainingManager(_cfg(), StubEngine())

    mock_pop = MagicMock()
    mock_stats = MagicMock()
    mock_neat_cfg = MagicMock()

    with patch("ai_car_sim.ai.training_manager.neat") as mock_neat:
        mock_neat.Population.return_value = mock_pop
        mock_neat.StatisticsReporter.return_value = mock_stats
        mock_neat.StdOutReporter.return_value = MagicMock()

        result = manager.create_population(neat_config=mock_neat_cfg)

    assert result is mock_pop
    assert mock_pop.add_reporter.call_count == 2
    assert manager.statistics is mock_stats


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------

def test_run_training_raises_without_population():
    manager = TrainingManager(_cfg(), StubEngine())
    with pytest.raises(RuntimeError, match="No population"):
        manager.run_training()


def test_run_training_calls_population_run():
    manager = TrainingManager(_cfg(max_generations=5), StubEngine())
    mock_pop = MagicMock()
    mock_pop.run.return_value = _fake_genome(fitness=10.0)
    manager._population = mock_pop

    winner = manager.run_training()

    mock_pop.run.assert_called_once_with(manager._eval_genomes, 5)
    assert winner is mock_pop.run.return_value


def test_run_training_respects_explicit_max_generations():
    manager = TrainingManager(_cfg(max_generations=100), StubEngine())
    mock_pop = MagicMock()
    mock_pop.run.return_value = _fake_genome()
    manager._population = mock_pop

    manager.run_training(max_generations=3)

    mock_pop.run.assert_called_once_with(manager._eval_genomes, 3)


# ---------------------------------------------------------------------------
# _eval_genomes
# ---------------------------------------------------------------------------

def test_eval_genomes_resets_fitness_before_engine_call():
    seen_before: list[float] = []

    class InspectingEngine:
        def evaluate_genomes(self, genomes, neat_config):
            for _, g in genomes:
                seen_before.append(g.fitness)
                g.fitness = 99.0

    manager = TrainingManager(_cfg(), InspectingEngine())
    genomes = _fake_genomes(3)
    for _, g in genomes:
        g.fitness = 42.0  # pre-set to non-zero

    manager._eval_genomes(genomes, MagicMock())

    assert all(f == 0.0 for f in seen_before)


def test_eval_genomes_delegates_to_engine():
    engine = StubEngine(fitness=5.0)
    manager = TrainingManager(_cfg(), engine)
    genomes = _fake_genomes(2)

    manager._eval_genomes(genomes, MagicMock())

    assert engine.calls[0] is genomes
    for _, g in genomes:
        assert g.fitness == 5.0


def test_eval_genomes_increments_generation_counter():
    manager = TrainingManager(_cfg(), StubEngine())
    manager._eval_genomes(_fake_genomes(), MagicMock())
    manager._eval_genomes(_fake_genomes(), MagicMock())
    assert manager.generation == 2


def test_on_generation_callback_fired_with_best_fitness():
    recorded: list[tuple[int, float]] = []

    def cb(gen: int, best: float) -> None:
        recorded.append((gen, best))

    engine = StubEngine(fitness=7.0)
    manager = TrainingManager(_cfg(), engine, on_generation=cb)
    manager._eval_genomes(_fake_genomes(3), MagicMock())

    assert recorded == [(1, 7.0)]


def test_on_generation_callback_not_required():
    manager = TrainingManager(_cfg(), StubEngine(), on_generation=None)
    manager._eval_genomes(_fake_genomes(), MagicMock())  # should not raise
