"""Focused tests for ai_car_sim.domain.track.Track.

Covers construction, validation edge cases, path resolution,
and serialisation round-trips not exercised elsewhere.
"""

import pytest
from ai_car_sim.domain.track import Track


def _track(**kw) -> Track:
    defaults = dict(
        name="Circuit Alpha",
        map_image_path="assets/maps/map.png",
        spawn_position=(830.0, 920.0),
        spawn_angle=0.0,
    )
    defaults.update(kw)
    return Track(**defaults)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_track_name_stored():
    assert _track(name="Beta").name == "Beta"


def test_track_spawn_position_stored():
    t = _track(spawn_position=(100.0, 200.0))
    assert t.spawn_position == (100.0, 200.0)


def test_track_spawn_angle_stored():
    assert _track(spawn_angle=45.0).spawn_angle == 45.0


def test_track_default_border_color():
    assert _track().border_color == (255, 255, 255, 255)


def test_track_default_checkpoints_none():
    assert _track().checkpoints is None


def test_track_default_description_empty():
    assert _track().description == ""


def test_track_with_checkpoints():
    cp = [((0, 0), (100, 0)), ((200, 0), (200, 100))]
    t = _track(checkpoints=cp)
    assert len(t.checkpoints) == 2


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_validate_valid_track_no_errors():
    assert _track().validate() == []


def test_validate_blank_name_error():
    errors = _track(name="   ").validate()
    assert any("name" in e.lower() for e in errors)


def test_validate_blank_map_path_error():
    errors = _track(map_image_path="  ").validate()
    assert any("map_image_path" in e for e in errors)


def test_validate_border_color_out_of_range():
    t = _track()
    t.border_color = (300, 0, 0, 255)
    errors = t.validate()
    assert any("border_color" in e for e in errors)


def test_validate_border_color_wrong_length():
    t = _track()
    t.border_color = (255, 255)  # type: ignore
    errors = t.validate()
    assert any("border_color" in e for e in errors)


def test_validate_returns_list():
    assert isinstance(_track().validate(), list)


# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------

def test_resolve_map_path_relative():
    t = _track(map_image_path="assets/maps/map.png")
    assert t.resolve_map_path("/project") == "/project/assets/maps/map.png"


def test_resolve_map_path_absolute_unchanged():
    t = _track(map_image_path="/abs/path/map.png")
    assert t.resolve_map_path("/project") == "/abs/path/map.png"


def test_resolve_map_path_normalises_separators():
    t = _track(map_image_path="assets/../assets/maps/map.png")
    resolved = t.resolve_map_path("/project")
    assert ".." not in resolved


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------

def test_to_dict_round_trip():
    original = _track(
        name="Delta",
        spawn_angle=90.0,
        checkpoints=[((10, 20), (30, 40))],
        description="Test track",
    )
    restored = Track.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.spawn_angle == original.spawn_angle
    assert restored.checkpoints == original.checkpoints
    assert restored.description == original.description


def test_from_dict_missing_required_key_raises():
    with pytest.raises(KeyError):
        Track.from_dict({"name": "x", "map_image_path": "y"})


def test_to_dict_border_color_is_list():
    d = _track().to_dict()
    assert isinstance(d["border_color"], list)


def test_from_dict_rgb_border_color():
    d = _track().to_dict()
    d["border_color"] = [255, 0, 0]
    t = Track.from_dict(d)
    assert t.border_color == (255, 0, 0)


def test_from_dict_null_checkpoints():
    d = _track().to_dict()
    d["checkpoints"] = None
    t = Track.from_dict(d)
    assert t.checkpoints is None


def test_to_dict_spawn_position_is_list():
    d = _track(spawn_position=(10.0, 20.0)).to_dict()
    assert d["spawn_position"] == [10.0, 20.0]
