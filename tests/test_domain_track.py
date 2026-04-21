"""Tests for ai_car_sim.domain.track."""

import pytest
from ai_car_sim.domain.track import Track


def _default_track(**kwargs) -> Track:
    defaults = dict(
        name="Test Track",
        map_image_path="assets/maps/map.png",
        spawn_position=(830.0, 920.0),
        spawn_angle=0.0,
    )
    defaults.update(kwargs)
    return Track(**defaults)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_track_creation_minimal():
    t = _default_track()
    assert t.name == "Test Track"
    assert t.spawn_position == (830.0, 920.0)
    assert t.border_color == (255, 255, 255, 255)
    assert t.checkpoints is None
    assert t.description == ""


def test_track_creation_with_checkpoints():
    cp = [((100, 200), (300, 200)), ((400, 100), (400, 300))]
    t = _default_track(checkpoints=cp)
    assert len(t.checkpoints) == 2


# ------------------------------------------------------------------
# Serialisation round-trip
# ------------------------------------------------------------------

def test_to_dict_and_from_dict_roundtrip():
    cp = [((10, 20), (30, 40))]
    original = _default_track(
        description="A test track",
        checkpoints=cp,
        border_color=(255, 255, 255, 255),
    )
    data = original.to_dict()
    restored = Track.from_dict(data)

    assert restored.name == original.name
    assert restored.map_image_path == original.map_image_path
    assert restored.spawn_position == original.spawn_position
    assert restored.spawn_angle == original.spawn_angle
    assert restored.border_color == original.border_color
    assert restored.checkpoints == original.checkpoints
    assert restored.description == original.description


def test_from_dict_missing_required_key_raises():
    with pytest.raises(KeyError):
        Track.from_dict({"name": "x", "map_image_path": "y"})  # missing spawn_position


def test_from_dict_rgb_border_color():
    data = {
        "name": "RGB Track",
        "map_image_path": "assets/maps/map.png",
        "spawn_position": [100, 200],
        "spawn_angle": 45.0,
        "border_color": [255, 0, 0],
    }
    t = Track.from_dict(data)
    assert t.border_color == (255, 0, 0)


# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------

def test_resolve_map_path_relative():
    t = _default_track(map_image_path="assets/maps/map.png")
    resolved = t.resolve_map_path("/project")
    assert resolved == "/project/assets/maps/map.png"


def test_resolve_map_path_absolute():
    t = _default_track(map_image_path="/absolute/path/map.png")
    assert t.resolve_map_path("/project") == "/absolute/path/map.png"


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_validate_valid_track():
    assert _default_track().validate() == []


def test_validate_empty_name():
    errors = _default_track(name="  ").validate()
    assert any("name" in e.lower() for e in errors)


def test_validate_bad_border_color():
    t = _default_track()
    t.border_color = (300, 0, 0)  # type: ignore[assignment]
    errors = t.validate()
    assert any("border_color" in e for e in errors)
