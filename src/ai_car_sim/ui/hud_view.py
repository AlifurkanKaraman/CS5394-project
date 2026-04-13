"""HUD view for the AI car simulation.

Renders simulation status information onto a pygame surface in a clean,
reusable component.  All rendering is stateless — ``HudView`` holds font
objects and layout constants but never mutates simulation state.

pygame is imported lazily so the module can be imported in headless
environments (tests, CI) without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class SimMode(Enum):
    """Current simulation operating mode shown in the HUD."""
    TRAINING = auto()
    REPLAY   = auto()
    MANUAL   = auto()


# ---------------------------------------------------------------------------
# HudMetrics – plain data object passed to draw()
# ---------------------------------------------------------------------------

@dataclass
class HudMetrics:
    """All values the HUD needs to render one frame.

    Args:
        generation: Current NEAT generation number.
        alive_count: Number of cars still active this generation.
        best_fitness: Highest fitness seen in the current generation.
        track_name: Display name of the selected track.
        mode: Current operating mode.
        avg_fitness: Optional average fitness across the population.
        best_speed: Optional speed of the leading car.
        elapsed_seconds: Optional wall-clock seconds for this generation.
        sim_speed: Current simulation speed multiplier (e.g. 1.0, 2.0).
    """

    generation: int = 0
    alive_count: int = 0
    total_spawned: int = 0
    best_fitness: float = 0.0
    track_name: str = ""
    mode: SimMode = SimMode.TRAINING
    avg_fitness: float | None = None
    best_speed: float | None = None
    elapsed_seconds: float | None = None
    sim_speed: float = 1.0


# ---------------------------------------------------------------------------
# DrawSurface protocol – avoids hard pygame import at module level
# ---------------------------------------------------------------------------

@runtime_checkable
class DrawSurface(Protocol):
    """Minimal pygame.Surface interface needed by HudView."""

    def blit(self, source: Any, dest: Any) -> Any: ...
    def get_width(self) -> int: ...
    def get_height(self) -> int: ...


# ---------------------------------------------------------------------------
# HudView
# ---------------------------------------------------------------------------

# Layout constants
_MARGIN_X   = 12
_MARGIN_Y   = 12
_LINE_H     = 28
_FONT_SIZE  = 24
_SMALL_SIZE = 18

# Colours (R, G, B)
_WHITE  = (255, 255, 255)
_YELLOW = (255, 220,  50)
_CYAN   = ( 80, 220, 220)
_GRAY   = (180, 180, 180)
_BLACK  = (  0,   0,   0)


class HudView:
    """Renders simulation status information onto a pygame surface.

    Fonts are initialised lazily on the first :meth:`draw` call so the
    object can be constructed without an active pygame display.

    Args:
        font_name: pygame font name.  ``None`` uses the default system font.
        alpha: Background panel alpha (0 = fully transparent, 255 = opaque).
    """

    def __init__(
        self,
        font_name: str | None = None,
        alpha: int = 160,
    ) -> None:
        self._font_name = font_name
        self._alpha = alpha
        self._font: Any = None
        self._small_font: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, screen: DrawSurface, metrics: HudMetrics) -> None:
        """Render the HUD onto *screen* using *metrics*.

        Args:
            screen: The pygame surface to draw onto.
            metrics: Current simulation metrics to display.
        """
        self._ensure_fonts()
        lines = self._build_lines(metrics)
        self._render_panel(screen, lines)

    # ------------------------------------------------------------------
    # Layout helpers (pure – easy to unit test)
    # ------------------------------------------------------------------

    def _build_lines(self, metrics: HudMetrics) -> list[tuple[str, Any]]:
        """Return a list of ``(text, colour)`` pairs for the current frame.

        Args:
            metrics: Current simulation metrics.

        Returns:
            Ordered list of ``(label_string, rgb_colour)`` tuples.
        """
        mode_label = metrics.mode.name.capitalize()
        lines: list[tuple[str, Any]] = [
            (f"Mode:       {mode_label}",          _CYAN),
            (f"Generation: {metrics.generation}",  _WHITE),
            (f"Cars:       {metrics.alive_count}/{metrics.total_spawned}", _WHITE),
            (f"Best fit:   {metrics.best_fitness:.2f}", _YELLOW),
        ]

        if metrics.sim_speed != 1.0 or True:  # always show speed
            speed_label = f"{metrics.sim_speed:.1f}x"
            colour = _YELLOW if metrics.sim_speed != 1.0 else _GRAY
            lines.append((f"Speed:      {speed_label}", colour))

        if metrics.avg_fitness is not None:
            lines.append((f"Avg fit:    {metrics.avg_fitness:.2f}", _GRAY))

        if metrics.track_name:
            lines.append((f"Track:      {metrics.track_name}", _GRAY))

        if metrics.best_speed is not None:
            lines.append((f"Car spd:    {metrics.best_speed:.1f}", _GRAY))

        if metrics.elapsed_seconds is not None:
            lines.append((f"Time:       {metrics.elapsed_seconds:.1f}s", _GRAY))

        return lines

    @staticmethod
    def panel_rect(
        line_count: int,
        margin_x: int = _MARGIN_X,
        margin_y: int = _MARGIN_Y,
        line_height: int = _LINE_H,
        padding: int = 8,
    ) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` for the background panel.

        Pure helper — no pygame required.

        Args:
            line_count: Number of text lines to accommodate.
            margin_x: Left margin in pixels.
            margin_y: Top margin in pixels.
            line_height: Height per line in pixels.
            padding: Extra padding inside the panel.

        Returns:
            ``(x, y, w, h)`` rectangle tuple.
        """
        w = 260
        h = line_count * line_height + padding * 2
        return (margin_x, margin_y, w, h)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _ensure_fonts(self) -> None:
        """Initialise pygame fonts on first use."""
        if self._font is not None:
            return
        import pygame
        pygame.font.init()
        self._font       = pygame.font.SysFont(self._font_name, _FONT_SIZE)
        self._small_font = pygame.font.SysFont(self._font_name, _SMALL_SIZE)

    def _render_panel(
        self,
        screen: DrawSurface,
        lines: list[tuple[str, Any]],
    ) -> None:
        """Draw a semi-transparent background panel then blit each text line.

        Args:
            screen: Target surface.
            lines: ``(text, colour)`` pairs from :meth:`_build_lines`.
        """
        import pygame

        x, y, w, h = self.panel_rect(len(lines))

        # Semi-transparent background
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((*_BLACK, self._alpha))
        screen.blit(panel, (x, y))

        # Text lines
        for i, (text, colour) in enumerate(lines):
            surf = self._font.render(text, True, colour)
            screen.blit(surf, (x + 8, y + 8 + i * _LINE_H))
