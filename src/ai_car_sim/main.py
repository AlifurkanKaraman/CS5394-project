"""Main application bootstrap for the AI car simulation.

Composes all previously built modules into a runnable application:
menu → training / replay / manual.  Replaces the original
``if __name__ == '__main__'`` block with clean, testable composition.

Run via the package entrypoint::

    ai-car-sim                  # uses default config
    ai-car-sim --headless       # headless training (no display)
    ai-car-sim --replay         # skip menu, replay best genome
    ai-car-sim --config path    # custom SimulationConfig JSON

Or directly::

    python -m ai_car_sim.main
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency check (fast-fail before any pygame init)
# ---------------------------------------------------------------------------

def check_dependencies() -> None:
    """Raise :exc:`ImportError` with a helpful message if a required
    package is missing.

    Raises:
        ImportError: If ``neat-python`` or ``pygame`` are not installed.
    """
    missing: list[str] = []
    for pkg, install_name in [("neat", "neat-python"), ("pygame", "pygame")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install_name)
    if missing:
        raise ImportError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + "\nRun: pip install " + " ".join(missing)
        )


# ---------------------------------------------------------------------------
# Config / track helpers
# ---------------------------------------------------------------------------

def build_default_config(config_path: str | None = None):
    """Load or create the :class:`~ai_car_sim.domain.simulation_config.SimulationConfig`.

    Args:
        config_path: Path to a JSON config file.  When ``None`` the
            default :class:`SimulationConfig` is used.

    Returns:
        A :class:`~ai_car_sim.domain.simulation_config.SimulationConfig` instance.
    """
    from ai_car_sim.domain.simulation_config import SimulationConfig

    if config_path and Path(config_path).exists():
        cfg = SimulationConfig.load_from_json(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        cfg = SimulationConfig()
        logger.info("Using default SimulationConfig")
    return cfg


def build_default_tracks(config) -> list:
    """Return a list of :class:`~ai_car_sim.domain.track.Track` objects
    built from the map paths defined in *config*.

    Args:
        config: A :class:`~ai_car_sim.domain.simulation_config.SimulationConfig`.

    Returns:
        Non-empty list of :class:`~ai_car_sim.domain.track.Track` instances.
        Falls back to a single default track if no maps are found.
    """
    from ai_car_sim.domain.track import Track

    # (name, map_path, spawn_position, spawn_angle, difficulty_label, description)
    _TRACK_DEFS = [
        ("Map 1", "assets/maps/map.png",  (830.0, 920.0), 0.0,
         "Easy",   "Wide lanes, gentle curves — good for early training"),
        ("Map 2", "assets/maps/map2.png", (830.0, 920.0), 0.0,
         "Easy",   "Slightly tighter corners than Map 1"),
        ("Map 3", "assets/maps/map3.png", (830.0, 920.0), 0.0,
         "Medium", "Mixed straights and hairpins"),
        ("Map 4", "assets/maps/map4.png", (830.0, 920.0), 0.0,
         "Medium", "Narrow sections require precise steering"),
        ("Map 5", "assets/maps/map5.png", (830.0, 920.0), 0.0,
         "Hard",   "Complex layout — challenging for early generations"),
    ]

    tracks = [
        Track(
            name=name,
            map_image_path=path,
            spawn_position=spawn,
            spawn_angle=angle,
            border_color=config.border_color,
            description=desc,
        )
        for name, path, spawn, angle, _diff, desc in _TRACK_DEFS
    ]
    return tracks


def setup_output_dirs(config) -> None:
    """Ensure the output directory tree exists.

    Args:
        config: A :class:`~ai_car_sim.domain.simulation_config.SimulationConfig`.
    """
    from ai_car_sim.persistence.save_load import ensure_output_dirs

    ensure_output_dirs(
        config.output_dir,
        Path(config.output_dir) / "genomes",
        Path(config.output_dir) / "metrics",
    )


# ---------------------------------------------------------------------------
# Workflow runners
# ---------------------------------------------------------------------------

def run_training(config, track, headless: bool = False) -> bool:
    """Wire up and run a full NEAT training session.

    Args:
        config: Simulation configuration.
        track: Track to train on.
        headless: Skip pygame display when ``True``.

    Returns:
        ``True`` if the user pressed Q (return to menu), ``False`` otherwise.
    """
    from ai_car_sim.simulation.engine import SimulationEngine
    from ai_car_sim.ai.training_manager import TrainingManager
    from ai_car_sim.analytics.run_metrics import RunMetricsCollector
    from ai_car_sim.persistence.save_load import (
        save_best_genome, save_metrics_json, save_metrics_csv,
    )

    engine = SimulationEngine(config, track, headless=headless)
    engine.initialize()

    collector = RunMetricsCollector(track_name=track.name)

    def on_generation(gen: int, best_fitness: float) -> None:
        logger.info("Generation %d — best fitness: %.4f", gen, best_fitness)
        # Stop NEAT loop if user pressed Q
        if engine.return_to_menu:
            raise _ReturnToMenu()

    manager = TrainingManager(config, engine, on_generation=on_generation)
    manager.load_neat_config()
    manager.create_population()

    returned_to_menu = False
    try:
        winner = manager.run_training()
    except _ReturnToMenu:
        logger.info("User returned to menu during training.")
        returned_to_menu = True
        winner = None
    finally:
        engine.shutdown()

    if winner is not None:
        genome_path = Path(config.output_dir) / "genomes" / "best_genome.pkl"
        save_best_genome(winner, genome_path)

        summary = collector.record_run_summary(
            neat_config_path=config.neat_config_path,
            genome_path=str(genome_path),
        ) if collector.generation_count > 0 else None

        if summary:
            save_metrics_json(
                summary.to_dict(),
                Path(config.output_dir) / "metrics" / "run_summary.json",
            )
        save_metrics_csv(
            collector.to_rows(),
            Path(config.output_dir) / "metrics" / "generations.csv",
        )
        logger.info("Training complete. Best genome saved to %s", genome_path)

    return returned_to_menu


class _ReturnToMenu(Exception):
    """Internal signal: user pressed Q during training."""


def run_manual(config, track) -> None:
    """Launch manual / player-controlled mode.

    Args:
        config: Simulation configuration.
        track: Track to drive on.
    """
    from ai_car_sim.simulation.engine import SimulationEngine

    engine = SimulationEngine(config, track, headless=False)
    engine.initialize()
    try:
        engine.run_manual()
    finally:
        engine.shutdown()


def run_replay(config, track, genome_path: str | None = None) -> bool:
    """Load the best saved genome and replay it.

    Args:
        config: Simulation configuration.
        track: Track to replay on.
        genome_path: Path to the genome pickle.  Defaults to
            ``<output_dir>/genomes/best_genome.pkl``.

    Returns:
        ``True`` if the user pressed Q (return to menu).
    """
    from ai_car_sim.simulation.engine import SimulationEngine
    from ai_car_sim.ai.replay_loader import load_replay_driver

    gpath = genome_path or str(Path(config.output_dir) / "genomes" / "best_genome.pkl")
    driver = load_replay_driver(gpath, config.neat_config_path)

    engine = SimulationEngine(config, track, headless=False)
    engine.initialize()
    try:
        engine.run_replay(driver)
    finally:
        engine.shutdown()
    return engine.return_to_menu


def run_menu(config, tracks: list) -> None:
    """Show the interactive menu and launch the selected workflow.

    Loops: after training/replay returns (including via Q), the menu
    is shown again.  The loop only exits when the user selects Quit
    from the menu itself or closes the window.

    Args:
        config: Simulation configuration.
        tracks: Available tracks to choose from.
    """
    import pygame
    from ai_car_sim.ui.menu_controller import MenuController, MenuAction, TrackInfo, Difficulty
    from ai_car_sim.ui.hud_view import SimMode

    # One pygame init for the whole application lifetime
    pygame.init()
    flags = pygame.FULLSCREEN if config.fullscreen else 0
    screen = pygame.display.set_mode(
        (config.screen_width, config.screen_height), flags
    )
    pygame.display.set_caption("AI Car Simulation")
    clock = pygame.time.Clock()

    # Difficulty mapping — must match _TRACK_DEFS order in build_default_tracks()
    _DIFFICULTIES = [
        Difficulty.EASY,
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    ]

    track_infos = [
        TrackInfo(
            name=t.name,
            map_image_path=t.map_image_path,
            difficulty=_DIFFICULTIES[i] if i < len(_DIFFICULTIES) else Difficulty.MEDIUM,
            description=t.description,
        )
        for i, t in enumerate(tracks)
    ]

    while True:
        # ---- Menu loop ----
        menu = MenuController(tracks=track_infos)
        menu_running = True
        while menu_running:
            for event in pygame.event.get():
                action = menu.handle_event(event)
                if action in (MenuAction.QUIT, MenuAction.CONFIRM):
                    menu_running = False

            if not menu_running:
                break

            menu.draw(screen)
            pygame.display.flip()
            clock.tick(30)

        if menu.is_quit_requested():
            logger.info("User quit from menu.")
            break

        # Map selected track name back to Track object
        selected_name = menu.get_selected_track()
        track = next((t for t in tracks if t.name == selected_name), tracks[0])
        mode = menu.get_selected_mode()

        # ---- Run selected mode (engine manages its own display) ----
        # IMPORTANT: do NOT call pygame.quit() here — the engine reuses
        # the existing pygame context via its own initialize().
        pygame.quit()   # engine will re-init; this avoids display conflicts

        if mode == SimMode.TRAINING:
            run_training(config, track)
        elif mode == SimMode.REPLAY:
            run_replay(config, track)
        elif mode == SimMode.MANUAL:
            run_manual(config, track)
        else:
            logger.warning("Unknown mode: %s", mode)

        # Re-init pygame for the menu after the engine shuts down
        pygame.init()
        screen = pygame.display.set_mode(
            (config.screen_width, config.screen_height), flags
        )
        pygame.display.set_caption("AI Car Simulation")
        clock = pygame.time.Clock()

    pygame.quit()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="ai-car-sim",
        description="AI Car Simulation powered by NEAT neuroevolution.",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to a SimulationConfig JSON file.",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Skip the menu and immediately replay the best saved genome.",
    )
    parser.add_argument(
        "--genome", metavar="PATH",
        help="Path to a genome pickle file (used with --replay).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run training without a pygame display (server / CI mode).",
    )
    parser.add_argument(
        "--track", type=int, default=0, metavar="N",
        help="Zero-based index of the track to use (default: 0).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Application entrypoint.

    Args:
        argv: Optional argument list for testing.  Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        check_dependencies()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    config = build_default_config(args.config)
    tracks = build_default_tracks(config)
    setup_output_dirs(config)

    track = tracks[min(args.track, len(tracks) - 1)]

    try:
        if args.headless:
            run_training(config, track, headless=True)
        elif args.replay:
            run_replay(config, track, genome_path=args.genome)
        else:
            run_menu(config, tracks)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
