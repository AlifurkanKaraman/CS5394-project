"""Tests for ai_car_sim.ui.menu_controller (pure navigation logic – no pygame)."""

import pytest
from ai_car_sim.ui.hud_view import SimMode
from ai_car_sim.ui.menu_controller import MenuController, MenuAction, MenuItem


TRACKS = ["map1.png", "map2.png", "map3.png"]


def _ctrl(**kwargs) -> MenuController:
    return MenuController(tracks=TRACKS, **kwargs)


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_default_mode_index_zero():
    ctrl = _ctrl()
    assert ctrl.get_selected_mode() is SimMode.TRAINING


def test_initial_mode_index_respected():
    ctrl = _ctrl(initial_mode_index=1)
    assert ctrl.get_selected_mode() is SimMode.REPLAY


def test_initial_track_is_first():
    ctrl = _ctrl()
    assert ctrl.get_selected_track() == "map1.png"


def test_not_confirmed_initially():
    assert _ctrl().is_confirmed() is False


def test_not_quit_initially():
    assert _ctrl().is_quit_requested() is False


# ---------------------------------------------------------------------------
# move_up / move_down
# ---------------------------------------------------------------------------

def test_move_down_advances_selection():
    ctrl = _ctrl(initial_mode_index=0)
    ctrl.move_down()
    assert ctrl.get_selected_mode() is SimMode.REPLAY


def test_move_up_retreats_selection():
    ctrl = _ctrl(initial_mode_index=1)
    ctrl.move_up()
    assert ctrl.get_selected_mode() is SimMode.TRAINING


def test_move_down_wraps_around():
    ctrl = _ctrl(initial_mode_index=len(MenuController._MENU_ITEMS) - 1)
    ctrl.move_down()
    assert ctrl.get_selected_mode() is MenuController._MENU_ITEMS[0].mode


def test_move_up_wraps_around():
    ctrl = _ctrl(initial_mode_index=0)
    ctrl.move_up()
    # Should wrap to last item (Quit → None)
    assert ctrl.get_selected_mode() is None


# ---------------------------------------------------------------------------
# next_track / prev_track
# ---------------------------------------------------------------------------

def test_next_track_advances():
    ctrl = _ctrl()
    ctrl.next_track()
    assert ctrl.get_selected_track() == "map2.png"


def test_prev_track_retreats():
    ctrl = _ctrl()
    ctrl.next_track()
    ctrl.prev_track()
    assert ctrl.get_selected_track() == "map1.png"


def test_next_track_wraps():
    ctrl = _ctrl()
    for _ in range(len(TRACKS)):
        ctrl.next_track()
    assert ctrl.get_selected_track() == "map1.png"


def test_prev_track_wraps():
    ctrl = _ctrl()
    ctrl.prev_track()
    assert ctrl.get_selected_track() == "map3.png"


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------

def test_confirm_training_returns_confirm_action():
    ctrl = _ctrl(initial_mode_index=0)  # TRAINING
    action = ctrl.confirm()
    assert action is MenuAction.CONFIRM
    assert ctrl.is_confirmed() is True


def test_confirm_quit_item_returns_quit_action():
    ctrl = _ctrl()
    # Navigate to Quit (last item)
    while ctrl.get_selected_mode() is not None:
        ctrl.move_down()
    action = ctrl.confirm()
    assert action is MenuAction.QUIT
    assert ctrl.is_quit_requested() is True


def test_confirm_replay_sets_confirmed():
    ctrl = _ctrl(initial_mode_index=1)  # REPLAY
    ctrl.confirm()
    assert ctrl.is_confirmed() is True


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_confirmed():
    ctrl = _ctrl()
    ctrl.confirm()
    ctrl.reset()
    assert ctrl.is_confirmed() is False


def test_reset_clears_quit():
    ctrl = _ctrl()
    while ctrl.get_selected_mode() is not None:
        ctrl.move_down()
    ctrl.confirm()
    ctrl.reset()
    assert ctrl.is_quit_requested() is False


# ---------------------------------------------------------------------------
# MenuItem
# ---------------------------------------------------------------------------

def test_menu_items_cover_all_modes():
    modes = {item.mode for item in MenuController._MENU_ITEMS if item.mode is not None}
    assert SimMode.TRAINING in modes
    assert SimMode.REPLAY in modes
    assert SimMode.MANUAL in modes


def test_menu_has_quit_item():
    quit_items = [i for i in MenuController._MENU_ITEMS if i.mode is None]
    assert len(quit_items) == 1


# ---------------------------------------------------------------------------
# MenuAction enum
# ---------------------------------------------------------------------------

def test_menu_action_members():
    assert MenuAction.NONE
    assert MenuAction.CONFIRM
    assert MenuAction.QUIT
