"""2-D vector mathematics utilities for the AI car simulation.

Pure functions with no side effects used by car motion, radar
calculations, angle conversion, and distance measurements.
Only ``math`` from the standard library is required.
"""

from __future__ import annotations

import math


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the Euclidean distance between two 2-D points.

    Args:
        x1: X coordinate of the first point.
        y1: Y coordinate of the first point.
        x2: X coordinate of the second point.
        y2: Y coordinate of the second point.

    Returns:
        Non-negative distance in the same units as the inputs.
    """
    return math.hypot(x2 - x1, y2 - y1)


def rotate_point(
    cx: float, cy: float, angle_deg: float, length: float
) -> tuple[float, float]:
    """Return the endpoint of a ray cast from a centre point.

    Mirrors the radar formula used in ``newcar.py``::

        x = cx + cos(360 - angle) * length
        y = cy + sin(360 - angle) * length

    The ``360 - angle`` convention keeps angles consistent with pygame's
    coordinate system (y-axis pointing downward).

    Args:
        cx: X coordinate of the origin (car centre).
        cy: Y coordinate of the origin (car centre).
        angle_deg: Heading angle in degrees (pygame convention).
        length: Ray length in pixels.

    Returns:
        ``(x, y)`` endpoint of the ray.
    """
    rad = math.radians(360.0 - angle_deg)
    return (cx + math.cos(rad) * length, cy + math.sin(rad) * length)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp *value* to the closed interval [*min_value*, *max_value*].

    Args:
        value: The value to clamp.
        min_value: Lower bound (inclusive).
        max_value: Upper bound (inclusive).

    Returns:
        *value* unchanged if already within bounds, otherwise the
        nearest bound.

    Raises:
        ValueError: If *min_value* > *max_value*.
    """
    if min_value > max_value:
        raise ValueError(
            f"min_value ({min_value}) must not exceed max_value ({max_value})"
        )
    return max(min_value, min(value, max_value))


def heading_to_vector(angle_deg: float) -> tuple[float, float]:
    """Convert a heading angle to a unit direction vector.

    Uses the same ``360 - angle`` pygame convention as :func:`rotate_point`
    so the returned vector points in the direction the car is travelling.

    Args:
        angle_deg: Heading angle in degrees.

    Returns:
        ``(dx, dy)`` unit vector.  Both components are in ``[-1, 1]``.
    """
    rad = math.radians(360.0 - angle_deg)
    return (math.cos(rad), math.sin(rad))


def normalize_angle(angle_deg: float) -> float:
    """Normalise an angle to the half-open interval ``[0, 360)``.

    Args:
        angle_deg: Any angle in degrees.

    Returns:
        Equivalent angle in ``[0, 360)``.
    """
    return angle_deg % 360.0


def angle_between(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Return the angle in degrees from point 1 to point 2.

    Args:
        x1: X coordinate of the origin point.
        y1: Y coordinate of the origin point.
        x2: X coordinate of the target point.
        y2: Y coordinate of the target point.

    Returns:
        Angle in degrees in ``[0, 360)``.
    """
    return normalize_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))
