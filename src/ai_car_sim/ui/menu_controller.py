"""Menu controller for the AI car simulation.

Provides a keyboard-driven menu layer for selecting the operating mode
(train / replay / quit) and choosing a track before the simulation starts.
All navigation state is pure Python — pygame is only imported when
:meth:`draw` is called, so the controller is fully testable headlessly.

Track preview
-------------
When :class:`TrackInfo` objects are passed instead of plain strings, the
menu renders a thumbnail of the selected map image alongside a difficulty
label.  Plain string lists still work for backward compatibility.
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
# TrackInfo – rich track metadata for the preview panel
# ---------------------------------------------------------------------------

class Difficulty(Enum):
    """Subjective difficulty label shown in the track preview."""
    EASY   = "Easy"
    MEDIUM = "Medium"
    HARD   = "Hard"


@dataclass
class TrackInfo:
    """Metadata for one track entry in the menu.

    Args:
        name: Display name shown in the selector and passed back to the
            caller via :meth:`MenuController.get_selected_track`.
        map_image_path: Relative or absolute path to the map PNG used for
            the thumbnail preview.  ``None`` skips the thumbnail.
        difficulty: Subjective difficulty label.
        description: Optional one-line description shown below the thumbnail.
    """
    name: str
    map_image_path: str | None = None
    difficulty: Difficulty = Difficulty.MEDIUM
    description: str = ""


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
_TINY_SIZE   = 20
_LINE_H      = 52
_WHITE       = (255, 255, 255)
_YELLOW      = (255, 220,  50)
_GRAY        = (160, 160, 160)
_BLACK       = (  0,   0,   0)
_HIGHLIGHT   = ( 80, 200, 120)
_CYAN        = ( 80, 220, 220)

# Difficulty badge colours
_DIFF_COLORS: dict[Difficulty, tuple[int, int, int]] = {
    Difficulty.EASY:   ( 80, 200, 120),   # green
    Difficulty.MEDIUM: (255, 180,  50),   # amber
    Difficulty.HARD:   (220,  60,  60),   # red
}

# Thumbnail dimensions (pixels)
_THUMB_W = 320
_THUMB_H = 180


class MenuController:
    """Keyboard-driven pre-simulation menu with track preview panel.

    Navigation:
    - ``UP`` / ``DOWN`` arrows move between menu items.
    - ``LEFT`` / ``RIGHT`` arrows cycle through available tracks.
    - ``ENTER`` / ``SPACE`` confirms the current selection.
    - ``ESCAPE`` quits.

    Args:
        tracks: Either a list of :class:`TrackInfo` objects (full metadata)
            or a plain list of strings (names only, backward-compatible).
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
        tracks: list[TrackInfo | str] | None = None,
        initial_mode_index: int = 0,
    ) -> None:
        # Normalise plain strings to TrackInfo for uniform handling
        raw: list[TrackInfo | str] = tracks or [
            "assets/maps/map.png",
            "assets/maps/map2.png",
            "assets/maps/map3.png",
            "assets/maps/map4.png",
            "assets/maps/map5.png",
        ]
        self._tracks: list[TrackInfo] = [
            t if isinstance(t, TrackInfo) else TrackInfo(name=t)
            for t in raw
        ]

        self._mode_index: int = max(0, min(initial_mode_index, len(self._MENU_ITEMS) - 1))
        self._track_index: int = 0
        self._confirmed: bool = False
        self._quit: bool = False

        # Fonts initialised lazily
        self._font: Any = None
        self._small_font: Any = None
        self._title_font: Any = None
        self._tiny_font: Any = None

        # Thumbnail cache: map_image_path → scaled pygame.Surface
        self._thumb_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_selected_mode(self) -> SimMode | None:
        """Return the currently highlighted :class:`SimMode`, or ``None`` for quit."""
        return self._MENU_ITEMS[self._mode_index].mode

    def get_selected_track(self) -> str:
        """Return the name of the currently selected track."""
        return self._tracks[self._track_index].name

    def get_selected_track_info(self) -> TrackInfo:
        """Return the full :class:`TrackInfo` for the currently selected track."""
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

        Draws the title, mode list, and a track preview panel on the right
        side of the screen showing the thumbnail, track name, difficulty
        badge, and optional description.

        Args:
            screen: A ``pygame.Surface`` to draw onto.
        """
        import pygame

        self._ensure_fonts()
        sw, sh = screen.get_width(), screen.get_height()
        screen.fill(_BLACK)

        # ------------------------------------------------------------------
        # Title
        # ------------------------------------------------------------------
        title_surf = self._title_font.render(_TITLE, True, _YELLOW)
        screen.blit(title_surf, title_surf.get_rect(centerx=sw // 2, y=sh // 10))

        # ------------------------------------------------------------------
        # Menu items (left-centre column)
        # ------------------------------------------------------------------
        menu_cx = sw // 3          # horizontal centre of the menu column
        total_h = len(self._MENU_ITEMS) * _LINE_H
        start_y = sh // 2 - total_h // 2

        for i, item in enumerate(self._MENU_ITEMS):
            colour = _HIGHLIGHT if i == self._mode_index else _WHITE
            prefix = "▶ " if i == self._mode_index else "  "
            surf = self._font.render(f"{prefix}{item.label}", True, colour)
            screen.blit(surf, surf.get_rect(centerx=menu_cx, y=start_y + i * _LINE_H))

        # ------------------------------------------------------------------
        # Track preview panel (right column)
        # ------------------------------------------------------------------
        self._draw_track_preview(screen, sw, sh)

        # ------------------------------------------------------------------
        # Controls hint (bottom)
        # ------------------------------------------------------------------
        hint = "↑↓ mode   ←→ track   Enter/Space confirm   Esc quit"
        hint_surf = self._small_font.render(hint, True, _GRAY)
        screen.blit(hint_surf, hint_surf.get_rect(centerx=sw // 2, y=sh - 36))

    def _draw_track_preview(self, screen: Any, sw: int, sh: int) -> None:
        """Render the track preview panel on the right side of the screen.

        Shows: thumbnail image, track name, difficulty badge, description,
        and a ◀ N/N ▶ counter.

        Args:
            screen: Target pygame surface.
            sw: Screen width in pixels.
            sh: Screen height in pixels.
        """
        import pygame

        info = self._tracks[self._track_index]
        panel_cx = sw * 3 // 4          # horizontal centre of the preview column
        panel_top = sh // 4

        # ---- Thumbnail ----
        thumb = self._get_thumbnail(info.map_image_path)
        if thumb is not None:
            thumb_rect = thumb.get_rect(centerx=panel_cx, y=panel_top)
            # Subtle border
            border_rect = thumb_rect.inflate(4, 4)
            pygame.draw.rect(screen, _GRAY, border_rect, 2, border_radius=4)
            screen.blit(thumb, thumb_rect)
            text_y = panel_top + _THUMB_H + 16
        else:
            # Placeholder box when image is unavailable
            placeholder = pygame.Rect(0, 0, _THUMB_W, _THUMB_H)
            placeholder.centerx = panel_cx
            placeholder.y = panel_top
            pygame.draw.rect(screen, (40, 40, 40), placeholder, border_radius=4)
            pygame.draw.rect(screen, _GRAY, placeholder, 2, border_radius=4)
            no_img = self._small_font.render("No preview", True, _GRAY)
            screen.blit(no_img, no_img.get_rect(center=placeholder.center))
            text_y = panel_top + _THUMB_H + 16

        # ---- Track name ----
        name_surf = self._font.render(info.name, True, _WHITE)
        screen.blit(name_surf, name_surf.get_rect(centerx=panel_cx, y=text_y))
        text_y += _FONT_SIZE + 8

        # ---- Difficulty badge ----
        diff_colour = _DIFF_COLORS.get(info.difficulty, _GRAY)
        diff_text = f"Difficulty:  {info.difficulty.value}"
        diff_surf = self._small_font.render(diff_text, True, diff_colour)
        screen.blit(diff_surf, diff_surf.get_rect(centerx=panel_cx, y=text_y))
        text_y += _SMALL_SIZE + 6

        # ---- Description ----
        if info.description:
            desc_surf = self._tiny_font.render(info.description, True, _GRAY)
            screen.blit(desc_surf, desc_surf.get_rect(centerx=panel_cx, y=text_y))
            text_y += _TINY_SIZE + 4

        # ---- Track counter  ◀ N / total ▶ ----
        n = self._track_index + 1
        total = len(self._tracks)
        counter = f"◀  {n} / {total}  ▶"
        counter_surf = self._small_font.render(counter, True, _CYAN)
        screen.blit(counter_surf, counter_surf.get_rect(centerx=panel_cx, y=text_y + 8))

    def _get_thumbnail(self, image_path: str | None) -> Any:
        """Load, scale, and cache a thumbnail for *image_path*.

        Returns ``None`` if the path is ``None`` or the file cannot be loaded.

        Args:
            image_path: Path to the map PNG.

        Returns:
            A scaled ``pygame.Surface`` or ``None``.
        """
        if image_path is None:
            return None

        if image_path in self._thumb_cache:
            return self._thumb_cache[image_path]

        import pygame
        try:
            raw = pygame.image.load(image_path).convert()
            thumb = pygame.transform.smoothscale(raw, (_THUMB_W, _THUMB_H))
            self._thumb_cache[image_path] = thumb
            return thumb
        except Exception:
            # File missing or pygame not fully initialised — return None
            self._thumb_cache[image_path] = None
            return None

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
        self._tiny_font  = pygame.font.SysFont(None, _TINY_SIZE)
