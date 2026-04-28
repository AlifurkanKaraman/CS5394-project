# AI Car Simulation

A NEAT-based neuroevolution self-driving car simulation using pygame,
refactored from the [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation)
prototype into a clean, modular Python package.

> Full architecture documentation, class diagram, and design decisions:
> **[docs/architecture.md](docs/architecture.md)**
>
> How the NEAT learning algorithm works, fitness design, and configuration guide:
> **[docs/how-learning-works.md](docs/how-learning-works.md)**

## Features

- NEAT neuroevolution training with a configurable population
- Replay mode — load and watch the best saved genome
- Manual mode — drive a car yourself using the keyboard
- Photo mode — screenshots, pause, cinematic letterbox, HUD toggle
- Simulation speed control (0.25x – 8x)
- Generation transition overlay
- Per-generation metrics saved as JSON and CSV
- 477 headless pytest tests across all layers

## Architecture

Layered package structure under `src/ai_car_sim/`:

- `domain/` — pure data models (Track, VehicleState, SensorReading, SimulationConfig)
- `core/` — physics, collision detection, and radar sensing
- `ai/` — driver abstraction, NEAT integration, keyboard driver
- `ui/` — pygame rendering, menu, HUD, photo mode, generation overlay
- `simulation/` — main simulation loop engine
- `analytics/` — per-run metrics collection
- `persistence/` — save/load for genomes and metrics

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+, `neat-python`, and `pygame`.

## Running

**Original prototype:**
```bash
python newcar.py
```

**Refactored package:**
```bash
python -m ai_car_sim.main
# or via the installed entry point:
ai-car-sim
```

**CLI options:**
```
--config PATH     Load a SimulationConfig JSON file
--replay          Skip menu and replay the best saved genome
--genome PATH     Genome pickle to use with --replay
--headless        Train without a display (server / CI mode)
--track N         Zero-based track index (default: 0)
--log-level LEVEL DEBUG / INFO / WARNING / ERROR (default: INFO)
```

## Modes

### Training
NEAT evolves a population of neural-network drivers over many generations.
Genomes and metrics are saved to `outputs/` after training completes or
when you return to the menu with Q.

### Replay
Loads the best saved genome from `outputs/genomes/best_genome.pkl` and
runs it on the selected track. Use `--genome PATH` to specify a different file.

### Manual
Drive a car yourself using the arrow keys. After a crash, press R to
restart on the same track or Q to return to the menu.

## Controls

### Training / Replay

| Key | Action |
|---|---|
| Q | Return to main menu |
| ESC | Skip to next generation (training only) |
| + / = | Speed up simulation |
| - | Slow down simulation |
| 0 | Reset speed to 1x |
| SPACE | Pause / resume |
| P | Save screenshot to `outputs/screenshots/` |
| H | Toggle HUD visibility |
| C | Toggle cinematic mode (letterbox + no HUD) |

### Manual Mode

| Key | Action |
|---|---|
| LEFT | Steer left |
| RIGHT | Steer right |
| UP | Accelerate |
| DOWN | Brake |
| R | Restart after crash |
| Q | Return to main menu |
| SPACE | Pause / resume |
| P | Screenshot |
| H | Toggle HUD |
| C | Cinematic mode |

### Menu

| Key | Action |
|---|---|
| UP / DOWN | Navigate items |
| LEFT / RIGHT | Cycle tracks |
| ENTER / SPACE | Confirm selection |
| ESC | Quit |

## NEAT Configuration

The NEAT config lives at `configs/config.txt`. Key parameters:

| Parameter | Value | Notes |
|---|---|---|
| `pop_size` | 50 | Population per generation |
| `compatibility_threshold` | 3.0 | Species separation threshold |
| `min_species_size` | 2 | Minimum genomes per species |
| `max_stagnation` | 20 | Generations before a species is removed |
| `num_inputs` | 5 | One per radar angle |
| `num_outputs` | 4 | TURN_LEFT, TURN_RIGHT, SLOW_DOWN, SPEED_UP |

## Outputs

After training, the following files are written to `outputs/`:

```
outputs/
  genomes/best_genome.pkl      Best genome (pickle)
  metrics/run_summary.json     Aggregate run statistics
  metrics/generations.csv      Per-generation fitness data
  screenshots/                 PNG screenshots (P key)
```

## Testing

The project ships with a comprehensive pytest suite covering all layers.
All tests are headless — no pygame display is required.

```bash
# Run the full suite
pytest

# With coverage report
pytest --cov=ai_car_sim --cov-report=term-missing

# Single module
pytest tests/test_vector_utils.py -v

# Linter
ruff check src/

# Type checker
mypy src/
```

### Test organisation

| File | What it covers |
|---|---|
| `test_track.py` | Track domain model — construction, validation, path resolution, serialisation |
| `test_vehicle_state.py` | VehicleState — state transitions, crash/advance helpers, round-trip serialisation |
| `test_vector_utils.py` | Pure math helpers — distance, rotate_point, clamp, heading_to_vector, angle normalisation |
| `test_neat_driver.py` | NeatDriver — argmax action mapping, input forwarding, output validation |
| `test_persistence.py` | save/load helpers — genome pickle, JSON, CSV, integration with RunMetricsCollector |
| `test_domain_*.py` | Additional domain model coverage |
| `test_core_*.py` | Collision service, radar sensor, car entity |
| `test_ai_*.py` | Driver interface, training manager, replay loader, keyboard driver |
| `test_simulation_engine.py` | Headless engine step/lifecycle orchestration |
| `test_simulation_controls.py` | Speed control, spawn spread, generation overlay |
| `test_analytics_run_metrics.py` | Metrics collection and aggregation |
| `test_ui_*.py` | HUD, menu navigation, photo mode, generation overlay |
| `test_main_bootstrap.py` | CLI argument parsing and bootstrap helpers |
| `test_properties.py` | Hypothesis property tests across modules |

## Project Goals

- Refactor the monolithic `newcar.py` prototype into a clean, modular Python package
- Separate concerns across domain, core, AI, UI, simulation, analytics, and persistence layers
- Establish a comprehensive test suite (477 tests, all headless)
- Support training, replay, and manual-drive modes via a shared `DriverInterface`
- Provide photo mode and presentation tools for academic demos and reports
