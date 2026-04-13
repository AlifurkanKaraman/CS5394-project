"""Tests for ai_car_sim.domain.simulation_config."""

import json
import pytest
from pathlib import Path
from ai_car_sim.domain.simulation_config import SimulationConfig


# ------------------------------------------------------------------
# Construction & defaults
# ------------------------------------------------------------------

def test_default_values():
    cfg = SimulationConfig()
    assert cfg.screen_width == 1920
    assert cfg.screen_height == 1080
    assert cfg.car_size_x == 60
    assert cfg.car_size_y == 60
    assert cfg.border_color == (255, 255, 255, 255)
    assert cfg.max_radar_distance == 300
    assert cfg.radar_angles == [-90, -45, 0, 45, 90]
    assert cfg.default_speed == 20.0
    assert cfg.fps == 60
    assert cfg.max_generations == 1000
    assert cfg.fullscreen is True


def test_custom_values():
    cfg = SimulationConfig(screen_width=1280, screen_height=720, fps=30, fullscreen=False)
    assert cfg.screen_width == 1280
    assert cfg.screen_height == 720
    assert cfg.fps == 30
    assert cfg.fullscreen is False


# ------------------------------------------------------------------
# Derived helpers
# ------------------------------------------------------------------

def test_generation_timeout_seconds():
    cfg = SimulationConfig(fps=60, steps_per_generation=1200)
    assert cfg.generation_timeout_seconds() == pytest.approx(20.0)


def test_car_half_size():
    cfg = SimulationConfig(car_size_x=60)
    assert cfg.car_half_size() == pytest.approx(30.0)


# ------------------------------------------------------------------
# to_dict / from_dict round-trip
# ------------------------------------------------------------------

def test_to_dict_is_json_serialisable():
    cfg = SimulationConfig()
    d = cfg.to_dict()
    # Should not raise
    json.dumps(d)


def test_to_dict_border_color_is_list():
    cfg = SimulationConfig()
    d = cfg.to_dict()
    assert isinstance(d["border_color"], list)
    assert d["border_color"] == [255, 255, 255, 255]


def test_from_dict_roundtrip():
    original = SimulationConfig(
        screen_width=1280,
        screen_height=720,
        fps=30,
        max_generations=500,
        map_path="assets/maps/map2.png",
    )
    restored = SimulationConfig.from_dict(original.to_dict())
    assert restored.screen_width == original.screen_width
    assert restored.screen_height == original.screen_height
    assert restored.fps == original.fps
    assert restored.max_generations == original.max_generations
    assert restored.map_path == original.map_path
    assert restored.border_color == original.border_color
    assert restored.radar_angles == original.radar_angles


def test_from_dict_border_color_coerced_to_tuple():
    d = SimulationConfig().to_dict()
    restored = SimulationConfig.from_dict(d)
    assert isinstance(restored.border_color, tuple)


def test_from_dict_ignores_unknown_keys():
    d = SimulationConfig().to_dict()
    d["unknown_future_field"] = "ignored"
    # Should not raise
    cfg = SimulationConfig.from_dict(d)
    assert cfg.fps == 60


# ------------------------------------------------------------------
# JSON persistence
# ------------------------------------------------------------------

def test_save_and_load_json(tmp_path):
    cfg = SimulationConfig(screen_width=800, screen_height=600, fps=30)
    dest = tmp_path / "config.json"
    cfg.save_to_json(dest)
    assert dest.exists()
    loaded = SimulationConfig.load_from_json(dest)
    assert loaded.screen_width == 800
    assert loaded.screen_height == 600
    assert loaded.fps == 30


def test_load_from_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        SimulationConfig.load_from_json("/nonexistent/path/config.json")


def test_load_from_json_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        SimulationConfig.load_from_json(bad)


def test_save_to_json_content(tmp_path):
    cfg = SimulationConfig(map_path="assets/maps/map3.png")
    dest = tmp_path / "cfg.json"
    cfg.save_to_json(dest)
    raw = json.loads(dest.read_text())
    assert raw["map_path"] == "assets/maps/map3.png"
