"""Menu controller for the AI car simulation.

Provides a keyboard-driven menu layer for selecting the operating mode
(train / replay / quit) and choosing a track before the simulation starts.
All navigation state is pure Python — pygame is only imported when
:meth:`draw` is called, so the controller is fully testable headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ai_car_sim.ui.hud_view import SimMode


# ---------------------------------------------------------------------------
# MenuAction – what the caller should do after an event
# ---------------------------------------------------------------------------

class MenuAction(Enum):
    """Signal returned by :meth:`MenuController.handle_event`."""
    NONE     = auto()  # nothing changed
    CONFIRM  = auto()  # user confirmed selection → start
    QUIT     = auto()  # user wants to exit the application


# ---------------------------------------------------------------------------
# MenuItem
# ---------------------------------------------------------------------------

@dataclass
class MenuItem:
    """One selectable row in the main menu.

    Args:
        label: Display text shown on screen.
        mode: The :class:`~ai_car_sim.ui.hud_view.SimMode` this item triggers,
            or ``None`` for the quit option.
    """
    label: str
    mode: SimMode | None  # None → quit


# ---------------------------------------------------------------------------
# MenuController
# ---------------------------------------------------------------------------

# Layout / style constants
_TITLE       = "AI Car Simulation"
_FONT_SIZE   = 36
_SMALL_SIZE  = 24
_LINE_H      = 52
_WHITE       = (255, 255, 255)
_YELLOW      = (255, 220,  50)
_GRAY        = (160, 160, 160)
_BLACK       = (  0,   0,   0)
_HIGHLIGHT   = ( 80, 200, 120)


class MenuController:
    """Keyboard-driven pre-simulation menu.

    Navigation:
    - ``UP`` / ``DOWN`` arrows move between menu items.
    - ``LEFT`` / ``RIGHT`` arrows cycle through available tracks.
    - ``ENTER`` / ``SPACE`` confirms the current selection.
    - ``ESCAPE`` quits.

    Args:
        tracks: Ordered list of track names / paths to cycle through.
        initial_mode_index: Index into the built-in menu items to
            pre-select on startup.
    """

    _MENU_ITEMS: list[MenuItem] = [
        MenuItem("Start Training",  SimMode.TRAINING),
        MenuItem("Replay Best Run", SimMode.REPLAY),
        MenuItem("Manual Drive",    SimMode.MANUAL),
        MenuItem("Quit",            None),
    ]

    def __init__(
        self,
        tracks: list[str] | None = None,
        initial_mode_index: int = 0,
    ) -> None:
        self._tracks: list[str] = tracks or [
            "assets/maps/map.png",
            "assets/maps/map2.png",
            "assets/maps/map3.png",
            "assets/maps/map4.png",
            "assets/maps/map5.png",
        ]
        self._mode_index: int = max(0, min(initial_mode_index, len(self._MENU_ITEMS) - 1))
        self._track_index: int = 0
        self._confirmed: bool = False
        self._quit: bool = False

        # Fonts initialised lazily
        self._font: Any = None
        self._small_font: Any = None
        self._title_font: Any = None

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_selected_mode(self) -> SimMode | None:
        """Return the currently highlighted :class:`SimMode`, or ``None`` for quit."""
        return self._MENU_ITEMS[self._mode_index].mode

    def get_selected_track(self) -> str:
        """Return the currently selected track path / name."""
        return self._tracks[self._track_index]

    def is_confirmed(self) -> bool:
        """Return ``True`` after the user has pressed Enter/Space."""
        return self._confirmed

    def is_quit_requested(self) -> bool:
        """Return ``True`` if the user chose to quit."""
        return self._quit

    # ------------------------------------------------------------------
    # Navigation helpers (pure – no pygame)
    # ------------------------------------------------------------------

    def move_up(self) -> None:
        """Move the selection cursor one item up (wraps around)."""
        self._mode_index = (self._mode_index - 1) % len(self._MENU_ITEMS)

    def move_down(self) -> None:
        """Move the selection cursor one item down (wraps around)."""
        self._mode_index = (self._mode_index + 1) % len(self._MENU_ITEMS)

    def next_track(self) -> None:
        """Cycle to the next available track (wraps around)."""
        self._track_index = (self._track_index + 1) % len(self._tracks)

    def prev_track(self) -> None:
        """Cycle to the previous available track (wraps around)."""
        self._track_index = (self._track_index - 1) % len(self._tracks)

    def confirm(self) -> MenuAction:
        """Confirm the current selection and return the appropriate action.

        Returns:
            :attr:`MenuAction.QUIT` if the quit item is selected, otherwise
            :attr:`MenuAction.CONFIRM`.
        """
        if self.get_selected_mode() is None:
            self._quit = True
            return MenuAction.QUIT
        self._confirmed = True
        return MenuAction.CONFIRM

    def reset(self) -> None:
        """Clear confirmed / quit flags so the menu can be re-shown."""
        self._confirmed = False
        self._quit = False

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: Any) -> MenuAction:
        """Process one pygame event and update navigation state.

        Args:
            event: A ``pygame.event.Event`` (or any object with a ``type``
                attribute and a ``key`` attribute for ``KEYDOWN`` events).

        Returns:
            A :class:`MenuAction` indicating what the caller should do next.
        """
        import pygame

        if event.type == pygame.QUIT:
            self._quit = True
            return MenuAction.QUIT

        if event.type != pygame.KEYDOWN:
            return MenuAction.NONE

        key = event.key
        if key == pygame.K_UP:
            self.move_up()
        elif key == pygame.K_DOWN:
            self.move_down()
        elif key == pygame.K_RIGHT:
            self.next_track()
        elif key == pygame.K_LEFT:
            self.prev_track()
        elif key in (pygame.K_RETURN, pygame.K_SPACE):
            return self.confirm()
        elif key == pygame.K_ESCAPE:
            self._quit = True
            return MenuAction.QUIT

        return MenuAction.NONE

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, screen: Any) -> None:
        """Render the menu onto *screen*.

        Args:
            screen: A ``pygame.Surface`` to draw onto.
        """
        import pygame

        self._ensure_fonts()
        sw, sh = screen.get_width(), screen.get_height()
        screen.fill(_BLACK)

        # Title
        title_surf = self._title_font.render(_TITLE, True, _YELLOW)
        screen.blit(title_surf, title_surf.get_rect(centerx=sw // 2, y=sh // 6))

        # Menu items
        total_h = len(self._MENU_ITEMS) * _LINE_H
        start_y = sh // 2 - total_h // 2

        for i, item in enumerate(self._MENU_ITEMS):
            colour = _HIGHLIGHT if i == self._mode_index else _WHITE
            prefix = "▶ " if i == self._mode_index else "  "
            surf = self._font.render(f"{prefix}{item.label}", True, colour)
            screen.blit(surf, surf.get_rect(centerx=sw // 2, y=start_y + i * _LINE_H))

        # Track selector
        track_label = f"Track: ◀  {self.get_selected_track()}  ▶"
        track_surf = self._small_font.render(track_label, True, _GRAY)
        screen.blit(track_surf, track_surf.get_rect(centerx=sw // 2, y=sh - 80))

        # Controls hint
        hint = "↑↓ navigate   ←→ track   Enter confirm   Esc quit"
        hint_surf = self._small_font.render(hint, True, _GRAY)
        screen.blit(hint_surf, hint_surf.get_rect(centerx=sw // 2, y=sh - 44))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_fonts(self) -> None:
        if self._font is not None:
            return
        import pygame
        pygame.font.init()
        self._title_font = pygame.font.SysFont(None, 52)
        self._font       = pygame.font.SysFont(None, _FONT_SIZE)
        self._small_font = pygame.font.SysFont(None, _SMALL_SIZE)
