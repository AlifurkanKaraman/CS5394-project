"""Crash visual effects for the AI car simulation.

Provides a lightweight, self-contained particle system that renders:
- An expanding impact ring that fades out
- Smoke/debris particles that drift and fade
- A brief red screen-flash overlay
- Optional screen shake (offset applied by the caller)

All effects are time-limited and non-blocking — they run alongside the
simulation without pausing or interrupting training.

pygame is imported lazily so this module loads cleanly in headless tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Impact ring
_RING_MAX_RADIUS  = 40       # pixels
_RING_DURATION    = 18       # ticks until fully faded
_RING_COLOR       = (220, 80, 40)

# Smoke particles
_PARTICLE_COUNT   = 12       # spawned per crash
_PARTICLE_LIFE    = 30       # ticks
_PARTICLE_SPEED   = 2.5      # max initial speed (px/tick)
_PARTICLE_RADIUS  = 5        # starting radius (px)
_SMOKE_COLORS     = [
    (180, 80,  40),   # orange-red
    (200, 120, 40),   # amber
    (140, 140, 140),  # grey smoke
    (100, 100, 100),
]

# Screen flash
_FLASH_DURATION   = 8        # ticks
_FLASH_COLOR      = (180, 20, 20)
_FLASH_MAX_ALPHA  = 60       # 0-255

# Screen shake
_SHAKE_DURATION   = 10       # ticks
_SHAKE_MAGNITUDE  = 6        # max pixel offset


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------

@dataclass
class _Ring:
    x: float
    y: float
    age: int = 0

    def alive(self) -> bool:
        return self.age < _RING_DURATION

    def radius(self) -> float:
        return _RING_MAX_RADIUS * (self.age / _RING_DURATION)

    def alpha(self) -> int:
        return max(0, int(255 * (1.0 - self.age / _RING_DURATION)))


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    age: int = 0

    def alive(self) -> bool:
        return self.age < _PARTICLE_LIFE

    def radius(self) -> float:
        frac = 1.0 - self.age / _PARTICLE_LIFE
        return max(1.0, _PARTICLE_RADIUS * frac)

    def alpha(self) -> int:
        return max(0, int(220 * (1.0 - self.age / _PARTICLE_LIFE)))


@dataclass
class _Flash:
    age: int = 0

    def alive(self) -> bool:
        return self.age < _FLASH_DURATION

    def alpha(self) -> int:
        frac = 1.0 - self.age / _FLASH_DURATION
        return max(0, int(_FLASH_MAX_ALPHA * frac))


@dataclass
class _Shake:
    age: int = 0

    def alive(self) -> bool:
        return self.age < _SHAKE_DURATION

    def offset(self) -> tuple[int, int]:
        frac = 1.0 - self.age / _SHAKE_DURATION
        mag = int(_SHAKE_MAGNITUDE * frac)
        if mag == 0:
            return (0, 0)
        return (random.randint(-mag, mag), random.randint(-mag, mag))


# ---------------------------------------------------------------------------
# CrashEffectsSystem
# ---------------------------------------------------------------------------

class CrashEffectsSystem:
    """Manages all active crash visual effects.

    Usage::

        effects = CrashEffectsSystem()

        # When a car crashes:
        effects.register_crash(car_x, car_y)

        # Each render frame:
        shake_offset = effects.shake_offset()
        effects.draw(screen)
        effects.tick()

    The system is intentionally stateless with respect to the simulation —
    it only needs (x, y) crash coordinates and a surface to draw onto.
    """

    def __init__(self) -> None:
        self._rings:     list[_Ring]     = []
        self._particles: list[_Particle] = []
        self._flashes:   list[_Flash]    = []
        self._shakes:    list[_Shake]    = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_crash(self, x: float, y: float) -> None:
        """Spawn all effects for a crash at pixel position *(x, y)*.

        *(x, y)* should be the centre of the crashed car.

        Args:
            x: Crash X position in screen pixels.
            y: Crash Y position in screen pixels.
        """
        self._rings.append(_Ring(x, y))
        self._flashes.append(_Flash())
        self._shakes.append(_Shake())

        for _ in range(_PARTICLE_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, _PARTICLE_SPEED)
            color = random.choice(_SMOKE_COLORS)
            self._particles.append(_Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=color,
            ))

    def shake_offset(self) -> tuple[int, int]:
        """Return the current cumulative screen-shake pixel offset.

        The caller should add this to the blit position of the game map
        (or the entire scene) to produce the shake effect.

        Returns:
            ``(dx, dy)`` pixel offset.
        """
        if not self._shakes:
            return (0, 0)
        # Use the strongest active shake
        ox, oy = 0, 0
        for s in self._shakes:
            if s.alive():
                dx, dy = s.offset()
                if abs(dx) > abs(ox):
                    ox = dx
                if abs(dy) > abs(oy):
                    oy = dy
        return (ox, oy)

    def draw(self, screen: Any) -> None:
        """Draw all active effects onto *screen*.

        Must be called after the map and cars have been blitted so effects
        appear on top.

        Args:
            screen: A ``pygame.Surface``.
        """
        import pygame

        sw = screen.get_width()
        sh = screen.get_height()

        # ---- Screen flash (full-screen semi-transparent red overlay) ----
        for flash in self._flashes:
            if flash.alive():
                alpha = flash.alpha()
                if alpha > 0:
                    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                    overlay.fill((*_FLASH_COLOR, alpha))
                    screen.blit(overlay, (0, 0))
                break  # one flash overlay is enough per frame

        # ---- Impact rings ----
        for ring in self._rings:
            if ring.alive():
                r = int(ring.radius())
                if r > 0:
                    alpha = ring.alpha()
                    surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(
                        surf, (*_RING_COLOR, alpha),
                        (r + 2, r + 2), r, 3,
                    )
                    screen.blit(surf, (int(ring.x) - r - 2, int(ring.y) - r - 2))

        # ---- Smoke / debris particles ----
        for p in self._particles:
            if p.alive():
                r = int(p.radius())
                if r > 0:
                    alpha = p.alpha()
                    surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                    pygame.draw.circle(
                        surf, (*p.color, alpha),
                        (r + 1, r + 1), r,
                    )
                    screen.blit(surf, (int(p.x) - r - 1, int(p.y) - r - 1))

    def tick(self) -> None:
        """Advance all effects by one simulation tick and remove expired ones.

        Call once per rendered frame (not per physics tick when speed > 1x).
        """
        # Advance rings
        for r in self._rings:
            r.age += 1
        self._rings = [r for r in self._rings if r.alive()]

        # Advance and move particles
        for p in self._particles:
            p.x  += p.vx
            p.y  += p.vy
            p.vx *= 0.92   # drag
            p.vy *= 0.92
            p.age += 1
        self._particles = [p for p in self._particles if p.alive()]

        # Advance flashes
        for f in self._flashes:
            f.age += 1
        self._flashes = [f for f in self._flashes if f.alive()]

        # Advance shakes
        for s in self._shakes:
            s.age += 1
        self._shakes = [s for s in self._shakes if s.alive()]

    def has_active_effects(self) -> bool:
        """Return ``True`` if any effect is still running."""
        return bool(self._rings or self._particles or self._flashes or self._shakes)

    def clear(self) -> None:
        """Remove all active effects immediately (e.g. on generation reset)."""
        self._rings.clear()
        self._particles.clear()
        self._flashes.clear()
        self._shakes.clear()
