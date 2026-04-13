"""Tests for ai_car_sim.ai.replay_loader."""

import pickle
import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_car_sim.ai.replay_loader import (
    load_best_genome,
    build_driver_from_saved_genome,
    load_replay_driver,
    validate_saved_run,
    ReplayLoadError,
)
from ai_car_sim.ai.neat_driver import NeatDriver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeGenome:
    """Picklable stand-in for neat.DefaultGenome."""
    fitness: float = 42.0


def _fake_genome(fitness: float = 42.0) -> FakeGenome:
    return FakeGenome(fitness=fitness)


def _write_genome(path: Path, genome: object) -> None:
    with path.open("wb") as fh:
        pickle.dump(genome, fh)


# ---------------------------------------------------------------------------
# load_best_genome
# ---------------------------------------------------------------------------

def test_load_best_genome_returns_genome(tmp_path):
    genome = _fake_genome(fitness=7.5)
    p = tmp_path / "best.pkl"
    _write_genome(p, genome)

    loaded = load_best_genome(p)
    assert loaded.fitness == 7.5


def test_load_best_genome_missing_file_raises(tmp_path):
    with pytest.raises(ReplayLoadError, match="not found"):
        load_best_genome(tmp_path / "nonexistent.pkl")


def test_load_best_genome_corrupt_file_raises(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ReplayLoadError, match="Failed to deserialise"):
        load_best_genome(bad)


def test_load_best_genome_accepts_string_path(tmp_path):
    p = tmp_path / "genome.pkl"
    _write_genome(p, _fake_genome())
    loaded = load_best_genome(str(p))
    assert loaded is not None


# ---------------------------------------------------------------------------
# build_driver_from_saved_genome
# ---------------------------------------------------------------------------

def test_build_driver_returns_neat_driver():
    genome = _fake_genome()
    mock_config = MagicMock()
    mock_network = MagicMock()

    with patch("ai_car_sim.ai.replay_loader.neat") as mock_neat:
        mock_neat.nn.FeedForwardNetwork.create.return_value = mock_network
        driver = build_driver_from_saved_genome(genome, mock_config)

    assert isinstance(driver, NeatDriver)
    mock_neat.nn.FeedForwardNetwork.create.assert_called_once_with(genome, mock_config)


def test_build_driver_propagates_expected_outputs():
    genome = _fake_genome()
    mock_config = MagicMock()
    mock_network = MagicMock()
    mock_network.activate.return_value = [0.1, 0.2, 0.3, 0.4]

    with patch("ai_car_sim.ai.replay_loader.neat") as mock_neat:
        mock_neat.nn.FeedForwardNetwork.create.return_value = mock_network
        driver = build_driver_from_saved_genome(genome, mock_config, expected_outputs=4)

    # Driver should validate 4 outputs without raising
    from ai_car_sim.domain.vehicle_state import VehicleState
    state = VehicleState(x=0.0, y=0.0, angle=0.0, speed=0.0)
    driver.decide_action([0.5] * 5, state)  # should not raise


def test_build_driver_raises_on_network_build_failure():
    genome = _fake_genome()
    mock_config = MagicMock()

    with patch("ai_car_sim.ai.replay_loader.neat") as mock_neat:
        mock_neat.nn.FeedForwardNetwork.create.side_effect = RuntimeError("bad genome")
        with pytest.raises(ReplayLoadError, match="Could not build"):
            build_driver_from_saved_genome(genome, mock_config)


# ---------------------------------------------------------------------------
# load_replay_driver
# ---------------------------------------------------------------------------

def test_load_replay_driver_missing_neat_config_raises(tmp_path):
    genome_path = tmp_path / "genome.pkl"
    _write_genome(genome_path, _fake_genome())

    with pytest.raises(ReplayLoadError, match="NEAT config not found"):
        load_replay_driver(genome_path, tmp_path / "missing.cfg")


def test_load_replay_driver_missing_genome_raises(tmp_path):
    neat_cfg = tmp_path / "neat.cfg"
    neat_cfg.write_text("")

    with patch("ai_car_sim.ai.replay_loader.neat") as mock_neat:
        mock_neat.config.Config.return_value = MagicMock()
        mock_neat.DefaultGenome = MagicMock()
        mock_neat.DefaultReproduction = MagicMock()
        mock_neat.DefaultSpeciesSet = MagicMock()
        mock_neat.DefaultStagnation = MagicMock()

        with pytest.raises(ReplayLoadError, match="not found"):
            load_replay_driver(tmp_path / "missing.pkl", neat_cfg)


def test_load_replay_driver_full_success(tmp_path):
    genome = _fake_genome(fitness=99.0)
    genome_path = tmp_path / "best.pkl"
    _write_genome(genome_path, genome)
    neat_cfg = tmp_path / "neat.cfg"
    neat_cfg.write_text("")

    mock_network = MagicMock()
    mock_network.activate.return_value = [0.1, 0.9, 0.2, 0.3]

    with patch("ai_car_sim.ai.replay_loader.neat") as mock_neat:
        mock_neat.config.Config.return_value = MagicMock()
        mock_neat.DefaultGenome = MagicMock()
        mock_neat.DefaultReproduction = MagicMock()
        mock_neat.DefaultSpeciesSet = MagicMock()
        mock_neat.DefaultStagnation = MagicMock()
        mock_neat.nn.FeedForwardNetwork.create.return_value = mock_network

        driver = load_replay_driver(genome_path, neat_cfg)

    assert isinstance(driver, NeatDriver)


# ---------------------------------------------------------------------------
# validate_saved_run
# ---------------------------------------------------------------------------

def test_validate_missing_file(tmp_path):
    result = validate_saved_run(tmp_path / "missing.pkl")
    assert result["exists"] is False
    assert result["size_bytes"] == 0
    assert result["fitness"] is None
    assert result["error"] is None


def test_validate_valid_genome(tmp_path):
    genome = _fake_genome(fitness=55.0)
    p = tmp_path / "genome.pkl"
    _write_genome(p, genome)

    result = validate_saved_run(p)
    assert result["exists"] is True
    assert result["size_bytes"] > 0
    assert result["fitness"] == 55.0
    assert result["error"] is None


def test_validate_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"garbage")

    result = validate_saved_run(bad)
    assert result["exists"] is True
    assert result["error"] is not None
    assert result["fitness"] is None


def test_validate_returns_absolute_path(tmp_path):
    p = tmp_path / "genome.pkl"
    _write_genome(p, _fake_genome())
    result = validate_saved_run(p)
    assert Path(result["path"]).is_absolute()
