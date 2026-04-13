"""Generation transition overlay for the AI car simulation.

Displays a centred banner (e.g. "Generation 3") for a short duration
whenever a new generation starts.  All state lives here; the engine
simply calls :meth:`show` at the start of each generation and
:meth:`draw` each frame.

pygame is imported lazily so the module loads cleanly in headless
environments.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLAY_DURATION_S: float = 1.5   # seconds the banner stays visible
_FADE_DURATION_S: float    = 0.4   # seconds of fade-out at the end

_BLACK  = (  0,   0,   0)
_WHITE  = (255, 255, 255)
_YELLOW = (255, 220,  50)
_CYAN   = ( 80, 220, 220)


class GenerationOverlay:
    """Renders a temporary generation-start banner.

    Args:
        display_duration: Seconds the banner is fully visible.
        fade_duration: Seconds over which the banner fades out.
    """

    def __init__(
        self,
        display_duration: float = _DISPLAY_DURATION_S,
        fade_duration: float = _FADE_DURATION_S,
    ) -> None:
        self._display_duration = display_duration
        self._fade_duration = fade_duration
        self._remaining: float = 0.0
        self._text: str = ""
        self._font_large: Any = None
        self._font_small: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, generation: int) -> None:
        """Trigger the banner for *generation*.

        Args:
            generation: The generation number to display.
        """
        self._text = f"Generation  {generation}"
        self._remaining = self._display_duration + self._fade_duration
        logger.debug("GenerationOverlay: showing '%s'", self._text)

    @property
    def active(self) -> bool:
        """``True`` while the banner is still visible."""
        return self._remaining > 0.0

    def draw(self, screen: Any, dt: float) -> None:
        """Draw the banner onto *screen* and advance the timer.

        No-op when the banner is not active.

        Args:
            screen: A ``pygame.Surface`` to draw onto.
            dt: Elapsed seconds since the last frame.
        """
        if not self.active:
            return

        self._ensure_fonts()
        self._remaining = max(0.0, self._remaining - dt)

        # Compute alpha: fully opaque during display phase, fading during fade phase
        if self._remaining > self._fade_duration:
            alpha = 255
        else:
            alpha = int(255 * self._remaining / max(self._fade_duration, 1e-6))

        self._render(screen, alpha)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self, screen: Any, alpha: int) -> None:
        import pygame

        sw: int = screen.get_width()
        sh: int = screen.get_height()

        # Main generation text
        text_surf = self._font_large.render(self._text, True, _YELLOW)
        text_surf.set_alpha(alpha)

        # Sub-label
        sub_surf = self._font_small.render("NEW GENERATION", True, _CYAN)
        sub_surf.set_alpha(alpha)

        total_h = sub_surf.get_height() + 8 + text_surf.get_height()
        center_y = sh // 2 - total_h // 2

        # Background panel
        panel_w = max(text_surf.get_width(), sub_surf.get_width()) + 60
        panel_h = total_h + 40
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((*_BLACK, min(200, alpha)))
        panel_rect = panel.get_rect(center=(sw // 2, sh // 2))
        screen.blit(panel, panel_rect)

        # Sub-label (top)
        sub_rect = sub_surf.get_rect(centerx=sw // 2, y=panel_rect.y + 16)
        screen.blit(sub_surf, sub_rect)

        # Main text (below sub-label)
        text_rect = text_surf.get_rect(centerx=sw // 2, y=sub_rect.bottom + 8)
        screen.blit(text_surf, text_rect)

    def _ensure_fonts(self) -> None:
        if self._font_large is not None:
            return
        import pygame
        pygame.font.init()
        self._font_large = pygame.font.SysFont(None, 64)
        self._font_small = pygame.font.SysFont(None, 28)
