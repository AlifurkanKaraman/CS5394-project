"""Simulation engine for the AI car simulation.

Owns the pygame display, clock, and per-generation frame loop.
Composes all previously built modules — Car, DriverInterface, HudView,
Track, SimulationConfig — without duplicating their logic.

pygame is imported lazily so the module loads cleanly in headless
environments (tests, CI).
"""

from __future__ import annotations

import logging
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
        self._initialized = False

        # Runtime state reset each generation
        self._cars: list[Car] = []
        self._drivers: list[DriverInterface] = []
        self._genomes: list[Any] = []
        self._generation: int = 0

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

        Resets all cars to the track spawn position.

        Args:
            genomes: ``(genome_id, genome)`` pairs from NEAT.
            neat_config: Active NEAT configuration.

        Returns:
            The list of newly created cars (also stored on ``self._cars``).
        """
        self._genomes = [g for _, g in genomes]
        self._cars = []
        self._drivers = []

        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            driver = NeatDriver(network, expected_outputs=len(Action))
            car = Car(self.config, sprite_surface=self._car_sprite)
            car.reset(*self.track.spawn_position, self.track.spawn_angle)
            self._cars.append(car)
            self._drivers.append(driver)

        logger.debug("Created %d cars for generation", len(self._cars))
        return self._cars

    def create_replay_car(self, driver: DriverInterface) -> Car:
        """Create a single car for replay or manual mode.

        Args:
            driver: The driver to control the car.

        Returns:
            The created :class:`Car`.
        """
        car = Car(self.config, sprite_surface=self._car_sprite)
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
        via the ``EvaluationEngine`` protocol.

        Args:
            genomes: ``(genome_id, genome)`` pairs from NEAT.
            neat_config: Active NEAT configuration.
        """
        if not self._initialized:
            self.initialize()

        self.create_cars_for_genomes(genomes, neat_config)
        self._generation += 1
        self.run_generation()

        # Write accumulated fitness back to genomes
        for genome, car in zip(self._genomes, self._cars):
            genome.fitness = car.get_reward()

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def run_generation(self) -> None:
        """Run the frame loop for one training generation.

        Exits when all cars have crashed or the step budget is exhausted.
        Respects pause state from :class:`~ai_car_sim.ui.photo_mode.PhotoMode`.
        """
        step = 0
        budget = self.config.steps_per_generation
        start = time.monotonic()
        last_time = start

        while step < budget:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if not self.headless:
                quit_requested = self._handle_events()
                if quit_requested:
                    break

                if self._photo.paused:
                    # Keep rendering while paused but don't advance physics
                    alive = sum(1 for c in self._cars if c.is_alive())
                    self._render_frame(
                        mode=SimMode.TRAINING,
                        alive_count=alive,
                        elapsed=time.monotonic() - start,
                        dt=dt,
                    )
                    continue

            alive = self._step_all()
            if alive == 0:
                break

            if not self.headless:
                self._render_frame(
                    mode=SimMode.TRAINING,
                    alive_count=alive,
                    elapsed=time.monotonic() - start,
                    dt=dt,
                )

            step += 1

        logger.debug(
            "Generation %d finished after %d steps, %d cars alive",
            self._generation, step, sum(1 for c in self._cars if c.is_alive()),
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
        step = 0
        budget = self.config.steps_per_generation
        start = time.monotonic()
        last_time = start

        while step < budget:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if not self.headless:
                quit_requested = self._handle_events()
                if quit_requested:
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

            alive = self._step_all()
            if alive == 0:
                break

            if not self.headless:
                self._render_frame(
                    mode=SimMode.REPLAY,
                    alive_count=alive,
                    elapsed=time.monotonic() - start,
                    dt=dt,
                )

            step += 1

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
        """Step all internal cars/drivers against the loaded map."""
        return self.step(self._cars, self._drivers, self._game_map)

    def _render_frame(
        self,
        mode: SimMode,
        alive_count: int,
        elapsed: float = 0.0,
        dt: float = 0.0,
    ) -> None:
        """Draw map, cars, HUD, and photo-mode overlays for one frame."""
        import pygame

        self._screen.blit(self._game_map, (0, 0))

        for car in self._cars:
            if car.is_alive():
                car.draw(self._screen)

        best_fitness = max(
            (c.get_reward() for c in self._cars if c.is_alive()),
            default=0.0,
        )
        metrics = HudMetrics(
            generation=self._generation,
            alive_count=alive_count,
            best_fitness=best_fitness,
            track_name=self.track.name,
            mode=mode,
            elapsed_seconds=elapsed,
        )

        # Only draw HUD when photo mode allows it
        if self._photo.hud_visible:
            self._hud.draw(self._screen, metrics)

        # Photo-mode overlays always drawn on top
        self._photo.draw_overlays(self._screen, dt)

        pygame.display.flip()
        self._clock.tick(self.config.fps)

    def _handle_events(self) -> bool:
        """Process all pending pygame events.

        Handles photo-mode keys (P, SPACE, H, C) and quit signals.

        Returns:
            ``True`` if the simulation should exit.
        """
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True

                # Screenshot: P key — photo_mode.handle_event returns True
                # but we also need to call take_screenshot here
                if event.key == pygame.K_p:
                    self._photo.take_screenshot(self._screen)
                    continue

                # All other photo-mode keys (SPACE, H, C)
                self._photo.handle_event(event)

        return False

    def _handle_quit_event(self) -> bool:
        """Legacy wrapper — delegates to :meth:`_handle_events`."""
        return self._handle_events()

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
