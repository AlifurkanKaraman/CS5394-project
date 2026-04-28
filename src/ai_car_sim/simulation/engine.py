"""Simulation engine for the AI car simulation.

Owns the pygame display, clock, and per-generation frame loop.
Composes all previously built modules — Car, DriverInterface, HudView,
Track, SimulationConfig — without duplicating their logic.

pygame is imported lazily so the module loads cleanly in headless
environments (tests, CI).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import neat  # type: ignore[import]

from ai_car_sim.domain.simulation_config import SimulationConfig
from ai_car_sim.domain.track import Track
from ai_car_sim.core.car import Car
from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.ai.neat_driver import NeatDriver
from ai_car_sim.ui.hud_view import HudView, HudMetrics, SimMode
from ai_car_sim.ui.photo_mode import PhotoMode
from ai_car_sim.ui.generation_overlay import GenerationOverlay
from ai_car_sim.analytics.best_tracker import BestPerformanceTracker
from ai_car_sim.ui.crash_effects import CrashEffectsSystem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Speed multiplier steps (index 2 = 1.0x = default)
# ---------------------------------------------------------------------------
_SPEED_STEPS: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
_DEFAULT_SPEED_IDX: int = 2   # 1.0x

# ---------------------------------------------------------------------------
# Spawn spread: small perpendicular offsets so cars don't stack
# ---------------------------------------------------------------------------
_SPAWN_SPREAD_PX: float = 4.0   # pixels between adjacent spawn slots — keep tight to avoid border hits

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Main simulation loop and orchestration layer.

    Responsibilities:
    - pygame initialisation and teardown
    - Loading the track map image
    - Creating and resetting :class:`~ai_car_sim.core.car.Car` instances
    - Stepping physics, applying driver decisions, updating the HUD
    - Terminating a run on timeout or when all cars have crashed
    - Satisfying the :class:`~ai_car_sim.ai.training_manager.EvaluationEngine`
      protocol so :class:`~ai_car_sim.ai.training_manager.TrainingManager`
      can delegate genome evaluation here

    Args:
        config: Simulation configuration.
        track: Track to run on.
        headless: When ``True`` skip all pygame display operations (useful
            for testing or server-side training without a monitor).
    """

    def __init__(
        self,
        config: SimulationConfig,
        track: Track,
        headless: bool = False,
    ) -> None:
        self.config = config
        self.track = track
        self.headless = headless

        self._screen: Any = None
        self._clock: Any = None
        self._game_map: Any = None
        self._car_sprite: Any = None
        self._hud = HudView()
        self._photo = PhotoMode()
        self._gen_overlay = GenerationOverlay()
        self._crash_fx = CrashEffectsSystem()
        self._initialized = False

        # Speed control
        self._speed_idx: int = _DEFAULT_SPEED_IDX

        # Generation control flags (set by event handler, read by loop)
        self._skip_generation: bool = False
        self._return_to_menu: bool = False

        # Runtime state reset each generation
        self._cars: list[Car] = []
        self._drivers: list[DriverInterface] = []
        self._genomes: list[Any] = []
        self._generation: int = 0

        # Cross-generation fitness tracking for HUD
        self._last_best_fitness: float = 0.0
        self._last_avg_fitness: float = 0.0
        self._last_species_count: int | None = None
        self._last_best_distance: float = 0.0
        self._last_best_checkpoints: int = 0

        # Persistent all-time best performance tracker
        self._best_tracker = BestPerformanceTracker.load(
            f"{config.output_dir}/best_performance.json"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise pygame, create the display, and load assets.

        Safe to call multiple times — subsequent calls are no-ops.

        Raises:
            FileNotFoundError: If the track map image cannot be found.
        """
        if self._initialized:
            return

        if not self.headless:
            import pygame
            pygame.init()
            flags = pygame.FULLSCREEN if self.config.fullscreen else 0
            self._screen = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height), flags
            )
            pygame.display.set_caption("AI Car Simulation")
            self._clock = pygame.time.Clock()
            self._game_map = self._load_map()
            self._car_sprite = self._load_car_sprite()

        self._initialized = True
        logger.info("SimulationEngine initialised (headless=%s)", self.headless)

    def shutdown(self) -> None:
        """Quit pygame and release resources."""
        if not self.headless:
            try:
                import pygame
                pygame.quit()
            except Exception:  # pragma: no cover
                pass
        self._initialized = False
        logger.info("SimulationEngine shut down")

    # ------------------------------------------------------------------
    # Car / driver creation
    # ------------------------------------------------------------------

    def create_cars_for_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> list[Car]:
        """Create one :class:`Car` and one :class:`NeatDriver` per genome.

        Cars are spawned with small perpendicular offsets so they do not
        visually stack on top of each other.

        Args:
            genomes: ``(genome_id, genome)`` pairs from NEAT.
            neat_config: Active NEAT configuration.

        Returns:
            The list of newly created cars (also stored on ``self._cars``).
        """
        self._genomes = [g for _, g in genomes]
        self._cars = []
        self._drivers = []

        spawn_x, spawn_y = self.track.spawn_position
        angle = self.track.spawn_angle
        n = len(genomes)

        # Perpendicular direction to the spawn heading
        perp_rad = math.radians(angle + 90.0)
        perp_dx = math.cos(perp_rad)
        perp_dy = math.sin(perp_rad)

        # Centre the spread: offsets go from -(n-1)/2 to +(n-1)/2
        for i, (_, genome) in enumerate(genomes):
            offset = (i - (n - 1) / 2.0) * _SPAWN_SPREAD_PX
            ox = spawn_x + perp_dx * offset
            oy = spawn_y + perp_dy * offset

            network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            driver = NeatDriver(network, expected_outputs=len(Action))
            car = Car(self.config, sprite_surface=self._car_sprite, track=self.track)
            car.reset(ox, oy, angle)
            self._cars.append(car)
            self._drivers.append(driver)

        logger.debug("Created %d cars for generation (spread=%.1fpx)", n, _SPAWN_SPREAD_PX)
        return self._cars

    def create_replay_car(self, driver: DriverInterface, manual_mode: bool = False) -> Car:
        """Create a single car for replay or manual mode.

        Args:
            driver: The driver to control the car.
            manual_mode: When ``True`` the car uses the slower manual-mode
                speed profile.

        Returns:
            The created :class:`Car`.
        """
        car = Car(self.config, sprite_surface=self._car_sprite, manual_mode=manual_mode)
        car.reset(*self.track.spawn_position, self.track.spawn_angle)
        self._cars = [car]
        self._drivers = [driver]
        self._genomes = []
        return car

    # ------------------------------------------------------------------
    # EvaluationEngine protocol (used by TrainingManager)
    # ------------------------------------------------------------------

    def evaluate_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        """Run one full generation and assign fitness to each genome.

        Called by :class:`~ai_car_sim.ai.training_manager.TrainingManager`
        via the ``EvaluationEngine`` protocol.  This is where the simulation
        and the NEAT algorithm meet:

        1. ``create_cars_for_genomes()`` — one Car + NeatDriver per genome
        2. ``run_generation()`` — frame loop until all cars crash or timeout
        3. fitness write-back — ``genome.fitness = car.get_reward()``

        The genome list and car list are kept **index-aligned** throughout:
        ``self._genomes[i]`` always corresponds to ``self._cars[i]``.
        This is the critical invariant that ensures each genome receives
        only its own car's fitness and not another car's.

        Args:
            genomes: ``(genome_id, genome)`` pairs from NEAT.
            neat_config: Active NEAT configuration.
        """
        if not self._initialized:
            self.initialize()

        # Reset per-generation flags — critical: _return_to_menu must NOT
        # carry over from a previous generation or Q press would permanently
        # break all future generations.
        self._skip_generation = False
        # Note: _return_to_menu is intentionally NOT reset here so the
        # training loop in run_training() can detect it and stop.

        self.create_cars_for_genomes(genomes, neat_config)
        self._generation += 1

        # Clear any leftover crash effects from the previous generation
        self._crash_fx.clear()

        # Trigger generation overlay
        if not self.headless:
            self._gen_overlay.show(self._generation)

        self.run_generation()

        # --- Fitness write-back (index-aligned: genome[i] ↔ car[i]) ---
        # This is the critical step where NEAT receives the learning signal.
        # Each genome's fitness is set exclusively from its own car's reward.
        # No cross-contamination between genomes is possible because the lists
        # were built together in create_cars_for_genomes() and never reordered.
        for genome, car in zip(self._genomes, self._cars):
            genome.fitness = car.get_reward()

        fitnesses = [g.fitness for g in self._genomes if g.fitness is not None]
        best_fit = max(fitnesses) if fitnesses else 0.0
        avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        best_dist = max((c.distance_travelled for c in self._cars), default=0.0)
        best_cp = max((c.checkpoints_reached for c in self._cars), default=0)

        logger.info(
            "Gen %d | best_fit=%.1f avg_fit=%.1f best_dist=%.0fpx best_cp=%d",
            self._generation, best_fit, avg_fit, best_dist, best_cp,
        )

        # Store for HUD access
        self._last_best_fitness = best_fit
        self._last_avg_fitness = avg_fit
        self._last_best_distance = best_dist
        self._last_best_checkpoints = best_cp

        # Update and persist all-time best records
        self._best_tracker.update(
            generation=self._generation,
            track_name=self.track.name,
            best_fitness=best_fit,
            best_distance=best_dist,
            best_checkpoints=best_cp,
        )
        self._best_tracker.save(f"{self.config.output_dir}/best_performance.json")

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def run_generation(self) -> None:
        """Run the frame loop for one training generation.

        Exits when all cars have crashed, the step budget is exhausted,
        ESC (skip generation) is pressed, or Q (return to menu) is pressed.
        Respects pause state and simulation speed multiplier.
        """
        step = 0
        budget = self.config.steps_per_generation
        start = time.monotonic()
        last_time = start
        total_spawned = len(self._cars)

        while step < budget:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if not self.headless:
                self._handle_events()
                # Q → stop this generation AND signal caller to return to menu
                # ESC → stop this generation only (next gen will start normally)
                if self._return_to_menu or self._skip_generation:
                    break

                if self._photo.paused:
                    alive = sum(1 for c in self._cars if c.is_alive())
                    self._render_frame(
                        mode=SimMode.TRAINING,
                        alive_count=alive,
                        total_spawned=total_spawned,
                        elapsed=time.monotonic() - start,
                        dt=dt,
                    )
                    continue

            # Run multiple physics ticks per frame based on speed multiplier
            speed = _SPEED_STEPS[self._speed_idx]
            ticks = max(1, int(speed))
            alive = 0
            for _ in range(ticks):
                alive = self._step_all()
                step += 1
                if alive == 0 or step >= budget:
                    break

            if alive == 0:
                break

            if not self.headless:
                self._render_frame(
                    mode=SimMode.TRAINING,
                    alive_count=alive,
                    total_spawned=total_spawned,
                    elapsed=time.monotonic() - start,
                    dt=dt,
                )

        logger.debug(
            "Generation %d finished after %d steps, %d/%d cars alive",
            self._generation, step,
            sum(1 for c in self._cars if c.is_alive()), total_spawned,
        )

    def run_replay(self, driver: DriverInterface) -> None:
        """Run the simulation in replay / demo mode with a single driver.

        Args:
            driver: Pre-loaded driver (e.g. from
                :func:`~ai_car_sim.ai.replay_loader.load_replay_driver`).
        """
        if not self._initialized:
            self.initialize()

        self.create_replay_car(driver)
        self._gen_overlay.show(0)   # show "Generation 0" for replay
        step = 0
        budget = self.config.steps_per_generation
        start = time.monotonic()
        last_time = start
        self._skip_generation = False

        while step < budget:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if not self.headless:
                self._handle_events()
                if self._return_to_menu or self._skip_generation:
                    break

                if self._photo.paused:
                    alive = sum(1 for c in self._cars if c.is_alive())
                    self._render_frame(
                        mode=SimMode.REPLAY,
                        alive_count=alive,
                        elapsed=time.monotonic() - start,
                        dt=dt,
                    )
                    continue

            speed = _SPEED_STEPS[self._speed_idx]
            ticks = max(1, int(speed))
            alive = 0
            for _ in range(ticks):
                alive = self._step_all()
                step += 1
                if alive == 0 or step >= budget:
                    break

            if alive == 0:
                break

            if not self.headless:
                self._render_frame(
                    mode=SimMode.REPLAY,
                    alive_count=alive,
                    elapsed=time.monotonic() - start,
                    dt=dt,
                )

    # ------------------------------------------------------------------
    # Manual mode
    # ------------------------------------------------------------------

    def run_manual(self) -> None:
        """Run the simulation in manual / player-controlled mode.

        Spawns one car driven by a :class:`~ai_car_sim.ai.keyboard_driver.KeyboardDriver`.
        The loop runs until the car crashes, the step budget is exhausted,
        or the user presses Q.  After a crash the player is shown a
        "CRASHED" banner and can press R to restart or Q to return to menu.
        """
        from ai_car_sim.ai.keyboard_driver import KeyboardDriver
        from ai_car_sim.ui.photo_mode import PhotoMode, _CONTROLS_MANUAL

        if not self._initialized:
            self.initialize()

        # Use manual-specific controls overlay
        self._photo = PhotoMode(controls=_CONTROLS_MANUAL)

        driver = KeyboardDriver()
        self._return_to_menu = False

        while not self._return_to_menu:
            # Spawn / respawn — use manual_mode=True for slower speed profile
            self.create_replay_car(driver, manual_mode=True)
            self._crash_fx.clear()
            self._generation += 1
            if not self.headless:
                self._gen_overlay.show(self._generation)

            step = 0
            budget = self.config.steps_per_generation
            start = time.monotonic()
            last_time = start

            while step < budget:
                now = time.monotonic()
                dt = now - last_time
                last_time = now

                if not self.headless:
                    self._handle_events()
                    if self._return_to_menu:
                        break

                    if self._photo.paused:
                        car = self._cars[0]
                        self._render_manual_frame(car, elapsed=time.monotonic() - start, dt=dt)
                        continue

                # Single tick — manual mode always runs at 1x
                alive = self._step_all()
                step += 1

                if not self.headless:
                    car = self._cars[0]
                    self._render_manual_frame(car, elapsed=time.monotonic() - start, dt=dt)

                if alive == 0:
                    # Car crashed — show banner then wait for R or Q
                    if not self.headless:
                        self._show_crash_screen()
                    break

            if self._return_to_menu:
                break

        logger.info("Manual mode ended.")

    def _render_manual_frame(
        self,
        car: "Car",
        elapsed: float = 0.0,
        dt: float = 0.0,
    ) -> None:
        """Render one frame for manual mode."""
        import pygame

        shake = self._crash_fx.shake_offset()
        self._screen.blit(self._game_map, shake)
        car.draw(self._screen)

        # Crash effects on top of car, below HUD
        self._crash_fx.draw(self._screen)
        self._crash_fx.tick()

        metrics = HudMetrics(
            generation=self._generation,
            alive_count=1 if car.is_alive() else 0,
            total_spawned=1,
            best_fitness=car.get_reward(),
            track_name=self.track.name,
            mode=SimMode.MANUAL,
            elapsed_seconds=elapsed,
            best_speed=car.state.speed,
            sim_speed=1.0,
        )

        if self._photo.hud_visible:
            self._hud.draw(self._screen, metrics)

        self._gen_overlay.draw(self._screen, dt)
        self._photo.draw_overlays(self._screen, dt)

        pygame.display.flip()
        self._clock.tick(self.config.fps)

    def _show_crash_screen(self) -> None:
        """Show a CRASHED overlay and wait for R (restart) or Q (menu)."""
        import pygame

        self._ensure_crash_font()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._return_to_menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                    elif event.key == pygame.K_q:
                        self._return_to_menu = True
                        return

            # Keep rendering the last frame with the crash overlay
            sw = self._screen.get_width()
            sh = self._screen.get_height()

            text = self._crash_font.render("CRASHED!", True, (220, 60, 60))
            sub  = self._crash_sub_font.render("R = Restart    Q = Main Menu", True, (200, 200, 200))

            panel_w = max(text.get_width(), sub.get_width()) + 60
            panel_h = text.get_height() + sub.get_height() + 40
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 200))
            px = sw // 2 - panel_w // 2
            py = sh // 2 - panel_h // 2
            self._screen.blit(panel, (px, py))
            self._screen.blit(text, text.get_rect(centerx=sw // 2, y=py + 14))
            self._screen.blit(sub,  sub.get_rect(centerx=sw // 2, y=py + 14 + text.get_height() + 10))

            pygame.display.flip()
            self._clock.tick(30)

    def _ensure_crash_font(self) -> None:
        if not hasattr(self, "_crash_font"):
            import pygame
            pygame.font.init()
            self._crash_font     = pygame.font.SysFont(None, 72)
            self._crash_sub_font = pygame.font.SysFont(None, 32)

    # ------------------------------------------------------------------
    # Single-step helper
    # ------------------------------------------------------------------

    def step(
        self,
        cars: list[Car],
        drivers: list[DriverInterface],
        game_map: Any,
    ) -> int:
        """Advance one tick for the given cars and drivers.

        Pure orchestration — no pygame calls.  Useful for headless testing.

        Args:
            cars: Cars to update.
            drivers: Corresponding drivers (same length as *cars*).
            game_map: Map surface for collision and radar lookup.

        Returns:
            Number of cars still alive after this tick.
        """
        alive = 0
        for car, driver in zip(cars, drivers):
            if not car.is_alive():
                continue
            inputs = car.get_sensor_inputs(game_map)
            action = driver.decide_action(inputs, car.state)
            car.apply_action(action)
            car.update(game_map)
            if car.is_alive():
                alive += 1
        return alive

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_all(self) -> int:
        """Step all internal cars/drivers against the loaded map.

        Detects cars that crash this tick and registers crash effects for them.
        """
        # Snapshot alive state before the step so we can detect new crashes
        was_alive = [c.is_alive() for c in self._cars]
        alive = self.step(self._cars, self._drivers, self._game_map)

        # Register crash effects for any car that just died this tick
        if not self.headless:
            for car, previously_alive in zip(self._cars, was_alive):
                if previously_alive and not car.is_alive():
                    pos = car.crash_position
                    if pos is not None:
                        self._crash_fx.register_crash(pos[0], pos[1])

        return alive

    def _render_frame(
        self,
        mode: SimMode,
        alive_count: int,
        total_spawned: int = 0,
        elapsed: float = 0.0,
        dt: float = 0.0,
    ) -> None:
        """Draw map, cars, crash effects, HUD, generation overlay, and photo-mode overlays."""
        import pygame

        # Apply screen shake offset to the map blit position
        shake = self._crash_fx.shake_offset()
        self._screen.blit(self._game_map, shake)

        for car in self._cars:
            if car.is_alive():
                car.draw(self._screen)

        # Crash effects drawn on top of cars, below HUD
        self._crash_fx.draw(self._screen)
        self._crash_fx.tick()

        alive_fitnesses = [c.get_reward() for c in self._cars if c.is_alive()]
        best_fitness = max(alive_fitnesses, default=0.0)
        avg_fitness = (
            sum(alive_fitnesses) / len(alive_fitnesses) if alive_fitnesses else 0.0
        )
        best_dist = max(
            (c.distance_travelled for c in self._cars if c.is_alive()), default=0.0
        )
        best_cp = max(
            (c.checkpoints_reached for c in self._cars if c.is_alive()), default=0
        )

        metrics = HudMetrics(
            generation=self._generation,
            alive_count=alive_count,
            total_spawned=total_spawned,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            track_name=self.track.name,
            mode=mode,
            elapsed_seconds=elapsed,
            sim_speed=_SPEED_STEPS[self._speed_idx],
            species_count=self._last_species_count,
            best_distance=best_dist if best_dist > 0 else None,
            best_checkpoints=best_cp if best_cp > 0 else None,
            # All-time bests from persistent tracker
            all_time_best_fitness=(
                self._best_tracker.best_fitness
                if self._best_tracker.has_any_record() else None
            ),
            all_time_best_fitness_gen=(
                self._best_tracker.best_fitness_generation
                if self._best_tracker.has_any_record() else None
            ),
            all_time_best_distance=(
                self._best_tracker.best_distance
                if self._best_tracker.best_distance > 0 else None
            ),
            all_time_best_checkpoints=(
                self._best_tracker.best_checkpoints
                if self._best_tracker.best_checkpoints > 0 else None
            ),
            total_generations_trained=self._best_tracker.total_generations_trained,
        )

        if self._photo.hud_visible:
            self._hud.draw(self._screen, metrics)

        # Generation overlay (drawn above HUD, below photo overlays)
        self._gen_overlay.draw(self._screen, dt)

        # Photo-mode overlays always on top
        self._photo.draw_overlays(self._screen, dt)

        pygame.display.flip()
        self._clock.tick(self.config.fps)

    def _handle_events(self) -> None:
        """Process all pending pygame events.

        Sets :attr:`_return_to_menu` or :attr:`_skip_generation` flags
        instead of returning a bool, so callers can distinguish the two
        exit reasons.
        """
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._return_to_menu = True
                return

            if event.type != pygame.KEYDOWN:
                continue

            key = event.key

            # Q → return to main menu
            if key == pygame.K_q:
                self._return_to_menu = True
                return

            # ESC → skip to next generation
            if key == pygame.K_ESCAPE:
                self._skip_generation = True
                return

            # Screenshot
            if key == pygame.K_p:
                self._photo.take_screenshot(self._screen)
                continue

            # Speed control
            if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self._speed_idx = min(self._speed_idx + 1, len(_SPEED_STEPS) - 1)
                logger.info("Speed → %.2fx", _SPEED_STEPS[self._speed_idx])
                continue

            if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._speed_idx = max(self._speed_idx - 1, 0)
                logger.info("Speed → %.2fx", _SPEED_STEPS[self._speed_idx])
                continue

            if key in (pygame.K_0, pygame.K_KP0):
                self._speed_idx = _DEFAULT_SPEED_IDX
                logger.info("Speed reset → %.2fx", _SPEED_STEPS[self._speed_idx])
                continue

            # Photo-mode keys (SPACE, H, C)
            self._photo.handle_event(event)

    def _handle_quit_event(self) -> bool:
        """Legacy wrapper — calls :meth:`_handle_events` and returns menu flag."""
        self._handle_events()
        return self._return_to_menu

    def _load_map(self) -> Any:
        """Load and convert the track map image."""
        import pygame
        path = self.track.resolve_map_path(".")
        try:
            surface = pygame.image.load(path).convert()
        except FileNotFoundError:
            raise FileNotFoundError(f"Track map not found: {path}")
        logger.info("Loaded map: %s", path)
        return surface

    def _load_car_sprite(self) -> Any:
        """Load, convert, and scale the car sprite."""
        import pygame
        try:
            sprite = pygame.image.load("assets/cars/car.png").convert()
        except FileNotFoundError:
            logger.warning("Car sprite not found; cars will render as blank.")
            return None
        return pygame.transform.scale(
            sprite, (self.config.car_size_x, self.config.car_size_y)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cars(self) -> list[Car]:
        """Current list of cars (populated by :meth:`create_cars_for_genomes`)."""
        return self._cars

    @property
    def generation(self) -> int:
        """Current generation counter."""
        return self._generation

    @property
    def photo_mode(self) -> PhotoMode:
        """The attached :class:`~ai_car_sim.ui.photo_mode.PhotoMode` controller."""
        return self._photo

    @property
    def sim_speed(self) -> float:
        """Current simulation speed multiplier."""
        return _SPEED_STEPS[self._speed_idx]

    @property
    def return_to_menu(self) -> bool:
        """``True`` if Q was pressed and the engine should return to the menu."""
        return self._return_to_menu

    def speed_up(self) -> float:
        """Increase simulation speed one step and return the new multiplier."""
        self._speed_idx = min(self._speed_idx + 1, len(_SPEED_STEPS) - 1)
        return self.sim_speed

    def speed_down(self) -> float:
        """Decrease simulation speed one step and return the new multiplier."""
        self._speed_idx = max(self._speed_idx - 1, 0)
        return self.sim_speed

    def speed_reset(self) -> float:
        """Reset simulation speed to 1.0x and return the multiplier."""
        self._speed_idx = _DEFAULT_SPEED_IDX
        return self.sim_speed

    def set_species_count(self, count: int) -> None:
        """Update the species count shown in the HUD.

        Called by the training manager after each generation.

        Args:
            count: Number of active NEAT species.
        """
        self._last_species_count = count

    @property
    def best_tracker(self) -> "BestPerformanceTracker":
        """The persistent all-time best performance tracker."""
        return self._best_tracker
