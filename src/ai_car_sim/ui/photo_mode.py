"""Photo Mode / Screenshot System for the AI car simulation.

Provides a self-contained ``PhotoMode`` controller that the simulation
engine attaches to its render loop.  All state (paused, HUD hidden,
cinematic, confirmation timer) lives here so the engine stays clean.

Controls
--------
P      — save a PNG screenshot to ``outputs/screenshots/``
SPACE  — toggle pause / resume
H      — toggle HUD visibility
C      — toggle cinematic mode (hides HUD + draws letterbox bars)
ESC    — quit (handled by engine; listed here for the overlay)

pygame is imported lazily so the module loads in headless / test
environments without a display.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout / style constants
# ---------------------------------------------------------------------------

_CONFIRM_DURATION_S: float = 2.0          # seconds to show "Screenshot Saved"
_BAR_HEIGHT: int = 60                     # cinematic letterbox bar height px
_OVERLAY_ALPHA: int = 160                 # semi-transparent panel alpha

# Colours
_BLACK  = (  0,   0,   0)
_WHITE  = (255, 255, 255)
_YELLOW = (255, 220,  50)
_GREEN  = ( 80, 220, 120)
_GRAY   = (160, 160, 160)
_RED    = (220,  60,  60)

_CONTROLS_TRAINING: list[tuple[str, str]] = [
    ("P",         "Screenshot"),
    ("SPACE",     "Pause"),
    ("H",         "Hide HUD"),
    ("C",         "Cinematic"),
    ("Q",         "Main Menu"),
    ("ESC",       "Next Gen"),
    ("+ / -",     "Speed Up/Down"),
    ("0",         "Reset Speed"),
]

_CONTROLS_MANUAL: list[tuple[str, str]] = [
    ("LEFT",      "Steer Left"),
    ("RIGHT",     "Steer Right"),
    ("UP",        "Accelerate"),
    ("DOWN",      "Brake"),
    ("P",         "Screenshot"),
    ("SPACE",     "Pause"),
    ("H",         "Hide HUD"),
    ("C",         "Cinematic"),
    ("Q",         "Main Menu"),
]

# Default (training) controls kept for backward compat
_CONTROLS = _CONTROLS_TRAINING


class PhotoMode:
    """Manages screenshot, pause, HUD-hide, and cinematic state.

    Attach one instance to :class:`~ai_car_sim.simulation.engine.SimulationEngine`
    and call :meth:`handle_event` from the event loop and
    :meth:`draw_overlays` after the main scene is rendered each frame.

    Args:
        screenshot_dir: Directory where PNG files are saved.
            Created automatically on first screenshot.
    """

    def __init__(
        self,
        screenshot_dir: str | Path = "outputs/screenshots",
        controls: list[tuple[str, str]] | None = None,
    ) -> None:
        self._screenshot_dir = Path(screenshot_dir)
        self._controls = controls if controls is not None else _CONTROLS_TRAINING

        self.paused: bool = False
        self.hud_visible: bool = True
        self.cinematic: bool = False

        # Confirmation banner state
        self._confirm_text: str = ""
        self._confirm_remaining: float = 0.0   # seconds left to show banner

        # Fonts (lazy)
        self._font_large: Any = None
        self._font_small: Any = None
        self._font_tiny: Any = None

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: Any) -> bool:
        """Process one pygame event.

        Args:
            event: A ``pygame.event.Event``.

        Returns:
            ``True`` if the event was consumed (caller should not process
            it further), ``False`` otherwise.
        """
        import pygame

        if event.type != pygame.KEYDOWN:
            return False

        key = event.key

        if key == pygame.K_p:
            # Screenshot is taken by the engine which passes the screen
            # surface; we just set a flag here and let draw_overlays handle
            # the confirmation.  Actual save is triggered via take_screenshot().
            return True  # engine will call take_screenshot()

        if key == pygame.K_SPACE:
            self.paused = not self.paused
            logger.info("Simulation %s", "paused" if self.paused else "resumed")
            return True

        if key == pygame.K_h:
            # Cinematic mode overrides HUD; toggling H exits cinematic first
            if self.cinematic:
                self.cinematic = False
            self.hud_visible = not self.hud_visible
            logger.info("HUD %s", "shown" if self.hud_visible else "hidden")
            return True

        if key == pygame.K_c:
            self.cinematic = not self.cinematic
            if self.cinematic:
                self.hud_visible = False
            else:
                self.hud_visible = True
            logger.info("Cinematic mode %s", "on" if self.cinematic else "off")
            return True

        return False

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def take_screenshot(self, screen: Any) -> Path | None:
        """Save the current screen surface as a timestamped PNG.

        Args:
            screen: The ``pygame.Surface`` to capture.

        Returns:
            The :class:`~pathlib.Path` of the saved file, or ``None`` on
            failure.
        """
        import pygame

        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._screenshot_dir / f"screenshot_{timestamp}.png"

        try:
            pygame.image.save(screen, str(path))
            logger.info("Screenshot saved → %s", path)
            self._show_confirmation(f"Screenshot Saved  →  {path.name}")
            return path
        except Exception as exc:
            logger.warning("Screenshot failed: %s", exc)
            self._show_confirmation("Screenshot FAILED")
            return None

    # ------------------------------------------------------------------
    # Per-frame overlay rendering
    # ------------------------------------------------------------------

    def draw_overlays(self, screen: Any, dt: float) -> None:
        """Draw all photo-mode overlays onto *screen*.

        Call this **after** the main scene (map + cars + HUD) has been
        rendered so overlays appear on top.

        Args:
            screen: The ``pygame.Surface`` to draw onto.
            dt: Elapsed seconds since the last frame (used to tick the
                confirmation banner timer).
        """
        self._ensure_fonts()
        self._tick_confirmation(dt)

        sw: int = screen.get_width()
        sh: int = screen.get_height()

        if self.cinematic:
            self._draw_letterbox(screen, sw, sh)

        if self.paused:
            self._draw_paused_banner(screen, sw, sh)

        if self._confirm_remaining > 0:
            self._draw_confirmation(screen, sw, sh)

        if self.hud_visible:
            self._draw_controls_overlay(screen, sw, sh)

    # ------------------------------------------------------------------
    # Internal overlay helpers
    # ------------------------------------------------------------------

    def _draw_letterbox(self, screen: Any, sw: int, sh: int) -> None:
        """Draw black cinematic bars at top and bottom."""
        import pygame
        bar = pygame.Surface((sw, _BAR_HEIGHT))
        bar.fill(_BLACK)
        screen.blit(bar, (0, 0))
        screen.blit(bar, (0, sh - _BAR_HEIGHT))

    def _draw_paused_banner(self, screen: Any, sw: int, sh: int) -> None:
        """Draw a centred PAUSED banner near the top of the screen."""
        import pygame
        text_surf = self._font_large.render("[ PAUSED ]", True, _YELLOW)
        rect = text_surf.get_rect(centerx=sw // 2, y=18)

        # Background pill
        pad = 14
        bg = pygame.Surface((rect.width + pad * 2, rect.height + pad), pygame.SRCALPHA)
        bg.fill((*_BLACK, 200))
        screen.blit(bg, (rect.x - pad, rect.y - pad // 2))
        screen.blit(text_surf, rect)

    def _draw_confirmation(self, screen: Any, sw: int, sh: int) -> None:
        """Draw the screenshot-saved confirmation banner."""
        import pygame
        alpha = min(255, int(255 * self._confirm_remaining / _CONFIRM_DURATION_S))
        text_surf = self._font_small.render(self._confirm_text, True, _GREEN)
        text_surf.set_alpha(alpha)
        rect = text_surf.get_rect(centerx=sw // 2, y=sh - 48)

        bg = pygame.Surface((text_surf.get_width() + 24, text_surf.get_height() + 10),
                             pygame.SRCALPHA)
        bg.fill((*_BLACK, min(180, alpha)))
        screen.blit(bg, (rect.x - 12, rect.y - 5))
        screen.blit(text_surf, rect)

    def _draw_controls_overlay(self, screen: Any, sw: int, sh: int) -> None:
        """Draw the small controls guide in the bottom-right corner."""
        import pygame
        line_h = 20
        pad = 8
        lines = [f"{key}  {label}" for key, label in self._controls]
        panel_w = 220
        panel_h = len(lines) * line_h + pad * 2

        x = sw - panel_w - 10
        y = sh - panel_h - 10

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((*_BLACK, _OVERLAY_ALPHA))
        screen.blit(bg, (x, y))

        for i, line in enumerate(lines):
            surf = self._font_tiny.render(line, True, _GRAY)
            screen.blit(surf, (x + pad, y + pad + i * line_h))

    # ------------------------------------------------------------------
    # Confirmation timer
    # ------------------------------------------------------------------

    def _show_confirmation(self, text: str) -> None:
        self._confirm_text = text
        self._confirm_remaining = _CONFIRM_DURATION_S

    def _tick_confirmation(self, dt: float) -> None:
        if self._confirm_remaining > 0:
            self._confirm_remaining = max(0.0, self._confirm_remaining - dt)

    # ------------------------------------------------------------------
    # Font initialisation
    # ------------------------------------------------------------------

    def _ensure_fonts(self) -> None:
        if self._font_large is not None:
            return
        import pygame
        pygame.font.init()
        self._font_large = pygame.font.SysFont(None, 42)
        self._font_small = pygame.font.SysFont(None, 26)
        self._font_tiny  = pygame.font.SysFont(None, 20)
