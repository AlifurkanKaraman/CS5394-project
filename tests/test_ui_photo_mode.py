"""Tests for ai_car_sim.ui.photo_mode.PhotoMode (headless — no pygame display)."""

import pickle
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_car_sim.ui.photo_mode import PhotoMode, _CONFIRM_DURATION_S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pm(**kw) -> PhotoMode:
    return PhotoMode(**kw)


def _key_event(key: int):
    """Build a minimal fake KEYDOWN event."""
    import pygame
    event = MagicMock()
    event.type = pygame.KEYDOWN
    event.key = key
    return event


def _non_key_event():
    import pygame
    event = MagicMock()
    event.type = pygame.MOUSEMOTION
    return event


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_not_paused_initially():
    assert _pm().paused is False


def test_hud_visible_initially():
    assert _pm().hud_visible is True


def test_cinematic_off_initially():
    assert _pm().cinematic is False


def test_confirm_remaining_zero_initially():
    assert _pm()._confirm_remaining == 0.0


# ---------------------------------------------------------------------------
# handle_event – SPACE (pause toggle)
# ---------------------------------------------------------------------------

def test_space_toggles_pause_on():
    import pygame
    pm = _pm()
    assert pm.handle_event(_key_event(pygame.K_SPACE)) is True
    assert pm.paused is True


def test_space_toggles_pause_off():
    import pygame
    pm = _pm()
    pm.paused = True
    pm.handle_event(_key_event(pygame.K_SPACE))
    assert pm.paused is False


def test_space_consumed():
    import pygame
    pm = _pm()
    assert pm.handle_event(_key_event(pygame.K_SPACE)) is True


# ---------------------------------------------------------------------------
# handle_event – H (HUD toggle)
# ---------------------------------------------------------------------------

def test_h_hides_hud():
    import pygame
    pm = _pm()
    pm.handle_event(_key_event(pygame.K_h))
    assert pm.hud_visible is False


def test_h_restores_hud():
    import pygame
    pm = _pm()
    pm.hud_visible = False
    pm.handle_event(_key_event(pygame.K_h))
    assert pm.hud_visible is True


def test_h_exits_cinematic_first():
    import pygame
    pm = _pm()
    pm.cinematic = True
    pm.hud_visible = False
    pm.handle_event(_key_event(pygame.K_h))
    assert pm.cinematic is False
    # After exiting cinematic, H toggles hud: was False → True
    assert pm.hud_visible is True


# ---------------------------------------------------------------------------
# handle_event – C (cinematic toggle)
# ---------------------------------------------------------------------------

def test_c_enables_cinematic():
    import pygame
    pm = _pm()
    pm.handle_event(_key_event(pygame.K_c))
    assert pm.cinematic is True


def test_c_hides_hud_when_enabling():
    import pygame
    pm = _pm()
    pm.handle_event(_key_event(pygame.K_c))
    assert pm.hud_visible is False


def test_c_disables_cinematic():
    import pygame
    pm = _pm()
    pm.cinematic = True
    pm.handle_event(_key_event(pygame.K_c))
    assert pm.cinematic is False


def test_c_restores_hud_when_disabling():
    import pygame
    pm = _pm()
    pm.cinematic = True
    pm.hud_visible = False
    pm.handle_event(_key_event(pygame.K_c))
    assert pm.hud_visible is True


# ---------------------------------------------------------------------------
# handle_event – P (screenshot key consumed)
# ---------------------------------------------------------------------------

def test_p_key_consumed():
    import pygame
    pm = _pm()
    assert pm.handle_event(_key_event(pygame.K_p)) is True


# ---------------------------------------------------------------------------
# handle_event – non-key event not consumed
# ---------------------------------------------------------------------------

def test_non_key_event_not_consumed():
    pm = _pm()
    assert pm.handle_event(_non_key_event()) is False


# ---------------------------------------------------------------------------
# take_screenshot
# ---------------------------------------------------------------------------

def test_take_screenshot_creates_file(tmp_path):
    import pygame
    pygame.init()
    screen = pygame.Surface((100, 100))
    screen.fill((0, 128, 255))

    pm = PhotoMode(screenshot_dir=tmp_path / "shots")
    result = pm.take_screenshot(screen)

    assert result is not None
    assert result.exists()
    assert result.suffix == ".png"
    pygame.quit()


def test_take_screenshot_filename_format(tmp_path):
    import pygame
    pygame.init()
    screen = pygame.Surface((50, 50))

    pm = PhotoMode(screenshot_dir=tmp_path)
    result = pm.take_screenshot(screen)

    assert result.name.startswith("screenshot_")
    assert result.name.endswith(".png")
    pygame.quit()


def test_take_screenshot_creates_dir(tmp_path):
    import pygame
    pygame.init()
    screen = pygame.Surface((50, 50))

    shots_dir = tmp_path / "new" / "dir"
    pm = PhotoMode(screenshot_dir=shots_dir)
    pm.take_screenshot(screen)

    assert shots_dir.exists()
    pygame.quit()


def test_take_screenshot_sets_confirmation(tmp_path):
    import pygame
    pygame.init()
    screen = pygame.Surface((50, 50))

    pm = PhotoMode(screenshot_dir=tmp_path)
    pm.take_screenshot(screen)

    assert pm._confirm_remaining > 0
    assert "Screenshot Saved" in pm._confirm_text
    pygame.quit()


def test_take_screenshot_returns_none_on_failure(tmp_path):
    pm = PhotoMode(screenshot_dir=tmp_path)
    # Pass a non-surface object to force failure
    result = pm.take_screenshot("not_a_surface")
    assert result is None
    assert "FAILED" in pm._confirm_text


# ---------------------------------------------------------------------------
# Confirmation timer
# ---------------------------------------------------------------------------

def test_tick_confirmation_decrements():
    pm = _pm()
    pm._show_confirmation("Test")
    pm._tick_confirmation(0.5)
    assert pm._confirm_remaining == pytest.approx(_CONFIRM_DURATION_S - 0.5)


def test_tick_confirmation_clamps_to_zero():
    pm = _pm()
    pm._show_confirmation("Test")
    pm._tick_confirmation(_CONFIRM_DURATION_S + 10.0)
    assert pm._confirm_remaining == 0.0


def test_show_confirmation_sets_text_and_timer():
    pm = _pm()
    pm._show_confirmation("Hello")
    assert pm._confirm_text == "Hello"
    assert pm._confirm_remaining == pytest.approx(_CONFIRM_DURATION_S)


# ---------------------------------------------------------------------------
# Engine integration – photo_mode property
# ---------------------------------------------------------------------------

def test_engine_exposes_photo_mode():
    from ai_car_sim.simulation.engine import SimulationEngine
    from ai_car_sim.domain.simulation_config import SimulationConfig
    from ai_car_sim.domain.track import Track

    eng = SimulationEngine(
        SimulationConfig(), Track("T", "map.png", (0.0, 0.0), 0.0), headless=True
    )
    assert isinstance(eng.photo_mode, PhotoMode)


def test_engine_photo_mode_pause_state():
    from ai_car_sim.simulation.engine import SimulationEngine
    from ai_car_sim.domain.simulation_config import SimulationConfig
    from ai_car_sim.domain.track import Track

    eng = SimulationEngine(
        SimulationConfig(), Track("T", "map.png", (0.0, 0.0), 0.0), headless=True
    )
    assert eng.photo_mode.paused is False
    eng.photo_mode.paused = True
    assert eng.photo_mode.paused is True
