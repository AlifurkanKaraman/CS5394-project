"""Border-collision detection service for the AI car simulation.

Moves corner calculation and pixel-level collision checks out of the car
entity so the logic is reusable, isolated, and testable without a full
pygame display.

The only pygame dependency is a ``Surface``-like object that exposes
``get_at(pos)`` and ``get_size()`` — both are easy to mock in tests.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ai_car_sim.domain.vehicle_state import VehicleState
from ai_car_sim.core.vector_utils import rotate_point


# ---------------------------------------------------------------------------
# Surface protocol – lets tests pass a mock without importing pygame
# ---------------------------------------------------------------------------

@runtime_checkable
class MapSurface(Protocol):
    """Minimal interface required from a pygame-like map surface."""

    def get_at(self, pos: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return the RGBA colour of the pixel at *pos*."""
        ...

    def get_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the surface."""
        ...


# ---------------------------------------------------------------------------
# Corner offsets (degrees) used in the original newcar.py
# left-top=30, right-top=150, left-bottom=210, right-bottom=330
# ---------------------------------------------------------------------------
_CORNER_OFFSETS: tuple[float, ...] = (30.0, 150.0, 210.0, 330.0)


def compute_corners(
    state: VehicleState,
    car_size_x: float,
    car_size_y: float | None = None,
) -> list[tuple[float, float]]:
    """Compute the four rotated corner positions of the car bounding box.

    Replicates the corner formula from ``newcar.py``::

        length = 0.5 * car_size_x
        corner = rotate_point(cx, cy, angle + offset, length)

    The car centre is derived from ``state.x``, ``state.y``, and
    ``car_size_x`` / ``car_size_y`` (matching pygame's top-left origin).

    Args:
        state: Current :class:`~ai_car_sim.domain.vehicle_state.VehicleState`.
        car_size_x: Sprite width in pixels.
        car_size_y: Sprite height in pixels.  Defaults to *car_size_x* when
            omitted (square sprite).

    Returns:
        List of four ``(x, y)`` corner coordinates in the order
        left-top, right-top, left-bottom, right-bottom.
    """
    if car_size_y is None:
        car_size_y = car_size_x

    cx = state.x + car_size_x / 2.0
    cy = state.y + car_size_y / 2.0
    length = 0.5 * car_size_x

    return [
        rotate_point(cx, cy, state.angle + offset, length)
        for offset in _CORNER_OFFSETS
    ]


def is_out_of_bounds(
    corners: list[tuple[float, float]],
    surface: MapSurface,
) -> bool:
    """Return ``True`` if any corner lies outside the surface dimensions.

    Args:
        corners: Corner coordinates from :func:`compute_corners`.
        surface: Map surface used to obtain width/height bounds.

    Returns:
        ``True`` when at least one corner is outside ``[0, width) x [0, height)``.
    """
    width, height = surface.get_size()
    return any(
        not (0 <= x < width and 0 <= y < height)
        for x, y in corners
    )


def is_collision(
    corners: list[tuple[float, float]],
    game_map: MapSurface,
    border_color: tuple[int, ...],
) -> bool:
    """Return ``True`` if any corner pixel matches *border_color*.

    Out-of-bounds corners are treated as collisions (the car has left
    the track entirely).

    Args:
        corners: Corner coordinates from :func:`compute_corners`.
        game_map: Pygame-like surface supporting ``get_at`` and ``get_size``.
        border_color: RGBA or RGB colour tuple that represents a wall/border.

    Returns:
        ``True`` when a collision is detected.
    """
    width, height = game_map.get_size()

    for x, y in corners:
        ix, iy = int(x), int(y)

        # Treat out-of-bounds as a collision
        if not (0 <= ix < width and 0 <= iy < height):
            return True

        pixel = game_map.get_at((ix, iy))

        # Support both RGB and RGBA border_color comparisons
        if tuple(pixel[: len(border_color)]) == tuple(border_color):
            return True

    return False
