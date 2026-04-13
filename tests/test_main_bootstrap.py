"""Tests for ai_car_sim.main bootstrap helpers."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_car_sim.main import (
    check_dependencies,
    build_default_config,
    build_default_tracks,
    setup_output_dirs,
    parse_args,
    main,
)
from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.domain.track import Track


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------

def test_check_dependencies_passes_when_installed():
    # neat and pygame are installed in the project venv
    check_dependencies()  # should not raise


def test_check_dependencies_raises_on_missing_package():
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *args, **kwargs):
        if name == "neat":
            raise ImportError("no module named neat")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="neat-python"):
            check_dependencies()


# ---------------------------------------------------------------------------
# build_default_config
# ---------------------------------------------------------------------------

def test_build_default_config_returns_simulation_config():
    cfg = build_default_config()
    assert isinstance(cfg, SimulationConfig)


def test_build_default_config_uses_defaults_when_no_path():
    cfg = build_default_config(config_path=None)
    assert cfg.fps == 60
    assert cfg.max_generations == 1000


def test_build_default_config_loads_json(tmp_path):
    custom = SimulationConfig(fps=30, max_generations=5)
    p = tmp_path / "cfg.json"
    custom.save_to_json(p)
    loaded = build_default_config(config_path=str(p))
    assert loaded.fps == 30
    assert loaded.max_generations == 5


def test_build_default_config_falls_back_on_missing_path():
    cfg = build_default_config(config_path="/nonexistent/path.json")
    assert isinstance(cfg, SimulationConfig)


# ---------------------------------------------------------------------------
# build_default_tracks
# ---------------------------------------------------------------------------

def test_build_default_tracks_returns_list():
    cfg = SimulationConfig()
    tracks = build_default_tracks(cfg)
    assert isinstance(tracks, list)
    assert len(tracks) > 0


def test_build_default_tracks_are_track_instances():
    cfg = SimulationConfig()
    for t in build_default_tracks(cfg):
        assert isinstance(t, Track)


def test_build_default_tracks_have_names():
    cfg = SimulationConfig()
    for t in build_default_tracks(cfg):
        assert t.name


def test_build_default_tracks_inherit_border_color():
    cfg = SimulationConfig(border_color=(255, 255, 255, 255))
    for t in build_default_tracks(cfg):
        assert t.border_color == (255, 255, 255, 255)


# ---------------------------------------------------------------------------
# setup_output_dirs
# ---------------------------------------------------------------------------

def test_setup_output_dirs_creates_directories(tmp_path):
    cfg = SimulationConfig(output_dir=str(tmp_path / "outputs"))
    setup_output_dirs(cfg)
    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "outputs" / "genomes").exists()
    assert (tmp_path / "outputs" / "metrics").exists()


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = parse_args([])
    assert args.config is None
    assert args.replay is False
    assert args.headless is False
    assert args.track == 0
    assert args.log_level == "INFO"


def test_parse_args_replay_flag():
    args = parse_args(["--replay"])
    assert args.replay is True


def test_parse_args_headless_flag():
    args = parse_args(["--headless"])
    assert args.headless is True


def test_parse_args_config_path():
    args = parse_args(["--config", "my_config.json"])
    assert args.config == "my_config.json"


def test_parse_args_genome_path():
    args = parse_args(["--replay", "--genome", "outputs/best.pkl"])
    assert args.genome == "outputs/best.pkl"


def test_parse_args_track_index():
    args = parse_args(["--track", "2"])
    assert args.track == 2


def test_parse_args_log_level():
    args = parse_args(["--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# main() – smoke tests
# ---------------------------------------------------------------------------

def test_main_returns_1_on_missing_dependency():
    with patch("ai_car_sim.main.check_dependencies",
               side_effect=ImportError("missing neat-python")):
        code = main([])
    assert code == 1


def test_main_returns_0_on_keyboard_interrupt():
    with patch("ai_car_sim.main.check_dependencies"):
        with patch("ai_car_sim.main.build_default_config", return_value=SimulationConfig()):
            with patch("ai_car_sim.main.build_default_tracks", return_value=[
                Track("T", "map.png", (0.0, 0.0), 0.0)
            ]):
                with patch("ai_car_sim.main.setup_output_dirs"):
                    with patch("ai_car_sim.main.run_menu",
                               side_effect=KeyboardInterrupt):
                        code = main([])
    assert code == 0


def test_main_returns_1_on_unhandled_exception():
    with patch("ai_car_sim.main.check_dependencies"):
        with patch("ai_car_sim.main.build_default_config", return_value=SimulationConfig()):
            with patch("ai_car_sim.main.build_default_tracks", return_value=[
                Track("T", "map.png", (0.0, 0.0), 0.0)
            ]):
                with patch("ai_car_sim.main.setup_output_dirs"):
                    with patch("ai_car_sim.main.run_menu",
                               side_effect=RuntimeError("boom")):
                        code = main([])
    assert code == 1


def test_main_headless_calls_run_training():
    with patch("ai_car_sim.main.check_dependencies"):
        with patch("ai_car_sim.main.build_default_config", return_value=SimulationConfig()):
            with patch("ai_car_sim.main.build_default_tracks", return_value=[
                Track("T", "map.png", (0.0, 0.0), 0.0)
            ]):
                with patch("ai_car_sim.main.setup_output_dirs"):
                    with patch("ai_car_sim.main.run_training") as mock_train:
                        main(["--headless"])
    mock_train.assert_called_once()
    _, kwargs = mock_train.call_args
    assert kwargs.get("headless") is True or mock_train.call_args[0][2] is True


def test_main_replay_calls_run_replay():
    with patch("ai_car_sim.main.check_dependencies"):
        with patch("ai_car_sim.main.build_default_config", return_value=SimulationConfig()):
            with patch("ai_car_sim.main.build_default_tracks", return_value=[
                Track("T", "map.png", (0.0, 0.0), 0.0)
            ]):
                with patch("ai_car_sim.main.setup_output_dirs"):
                    with patch("ai_car_sim.main.run_replay") as mock_replay:
                        main(["--replay"])
    mock_replay.assert_called_once()
