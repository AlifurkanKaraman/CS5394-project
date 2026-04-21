"""Radar sensor system for the AI car simulation.

Encapsulates the ray-marching loop that scans outward from the car at
configured angle offsets, returning structured :class:`SensorReading`
objects.  Drawing concerns are kept entirely separate — this module only
produces data.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ai_car_sim.domain.sensor_reading import SensorReading
from ai_car_sim.domain.vehicle_state import VehicleState


# ---------------------------------------------------------------------------
# Surface protocol – mirrors the one in collision_service; no pygame import
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
# RadarSensorSystem
# ---------------------------------------------------------------------------

class RadarSensorSystem:
    """Casts radar rays from the car centre and returns distance readings.

    Replicates the ``check_radar`` logic from ``newcar.py`` but returns
    typed :class:`~ai_car_sim.domain.sensor_reading.SensorReading` objects
    instead of mutating a list on the car.

    Args:
        angle_offsets: Degree offsets relative to the car's heading at which
            rays are cast.  Defaults to ``[-90, -45, 0, 45, 90]``.
        max_distance: Maximum ray length in pixels.  Defaults to ``300``.
        border_color: RGBA or RGB tuple that marks a wall/border pixel.
            Defaults to ``(255, 255, 255, 255)``.
        car_size_x: Sprite width used to compute the car centre X offset.
        car_size_y: Sprite height used to compute the car centre Y offset.
            Defaults to *car_size_x* when ``None``.
    """

    def __init__(
        self,
        angle_offsets: list[float] | None = None,
        max_distance: int = 300,
        border_color: tuple[int, ...] = (255, 255, 255, 255),
        car_size_x: float = 60.0,
        car_size_y: float | None = None,
    ) -> None:
        self.angle_offsets: list[float] = (
            angle_offsets if angle_offsets is not None else [-90.0, -45.0, 0.0, 45.0, 90.0]
        )
        self.max_distance = max_distance
        self.border_color = border_color
        self.car_size_x = car_size_x
        self.car_size_y = car_size_y if car_size_y is not None else car_size_x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self, vehicle_state: VehicleState, game_map: MapSurface
    ) -> list[SensorReading]:
        """Cast one ray per configured angle offset and return the readings.

        Args:
            vehicle_state: Current car state (position, angle).
            game_map: Map surface used for pixel colour lookup.

        Returns:
            List of :class:`~ai_car_sim.domain.sensor_reading.SensorReading`
            objects, one per angle offset, in the same order as
            :attr:`angle_offsets`.
        """
        cx = vehicle_state.x + self.car_size_x / 2.0
        cy = vehicle_state.y + self.car_size_y / 2.0

        return [
            self._cast_ray(cx, cy, vehicle_state.angle + offset, offset, game_map)
            for offset in self.angle_offsets
        ]

    def normalized_inputs(
        self, vehicle_state: VehicleState, game_map: MapSurface
    ) -> list[float]:
        """Return sensor distances normalised to ``[0, 1]`` for neural-network input.

        Equivalent to calling :meth:`scan` then
        :meth:`~ai_car_sim.domain.sensor_reading.SensorReading.as_input_value`
        on each reading.

        Args:
            vehicle_state: Current car state.
            game_map: Map surface.

        Returns:
            List of floats in ``[0.0, 1.0]``, one per angle offset.
        """
        readings = self.scan(vehicle_state, game_map)
        return [r.as_input_value(self.max_distance) for r in readings]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cast_ray(
        self,
        cx: float,
        cy: float,
        absolute_angle: float,
        angle_offset: float,
        game_map: MapSurface,
    ) -> SensorReading:
        """March a single ray outward until a border pixel or max distance.

        Uses the same ``360 - angle`` pygame convention as the original
        ``check_radar`` method in ``newcar.py``.

        Args:
            cx: Car centre X in pixels.
            cy: Car centre Y in pixels.
            absolute_angle: Heading + offset angle in degrees.
            angle_offset: The offset component (stored on the reading).
            game_map: Map surface for pixel lookup.

        Returns:
            A :class:`~ai_car_sim.domain.sensor_reading.SensorReading`
            with the hit coordinates and distance.
        """
        width, height = game_map.get_size()
        rad = math.radians(360.0 - absolute_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        length = 0
        x = int(cx)
        y = int(cy)

        while length < self.max_distance:
            x = int(cx + cos_a * length)
            y = int(cy + sin_a * length)

            # Treat out-of-bounds as a border hit
            if not (0 <= x < width and 0 <= y < height):
                break

            pixel = game_map.get_at((x, y))
            if tuple(pixel[: len(self.border_color)]) == tuple(self.border_color):
                break

            length += 1

        distance = math.hypot(x - cx, y - cy)
        return SensorReading(
            angle_offset=angle_offset,
            hit_x=x,
            hit_y=y,
            distance=distance,
        )
