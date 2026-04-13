# AI Car Simulation

A NEAT-based neuroevolution self-driving car simulation using pygame,
refactored from the [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation)
prototype into a clean, modular Python package.

> Full architecture documentation, class diagram, and design decisions:
> **[docs/architecture.md](docs/architecture.md)**

## Architecture

Layered package structure under `src/ai_car_sim/`:

- `domain/` — pure data models (Track, VehicleState, SensorReading, SimulationConfig)
- `core/` — physics, collision detection, and radar sensing
- `ai/` — driver abstraction and NEAT integration
- `ui/` — pygame rendering and menu logic
- `simulation/` — main simulation loop engine
- `analytics/` — per-run metrics collection
- `persistence/` — save/load for genomes and metrics

## Running

**Original prototype (fully working):**
```bash
python newcar.py
```

**Refactored package (once logic is ported):**
```bash
pip install -e ".[dev]"
python -m ai_car_sim.main
# or use the entry point:
ai-car-sim
```

## Testing

The project ships with a comprehensive pytest suite covering all layers.
All tests are headless — no pygame display is required.

```bash
# Install dev dependencies (once)
pip install -e ".[dev]"

# Run the full suite
pytest

# Run with coverage report
pytest --cov=ai_car_sim --cov-report=term-missing

# Run a specific focused module
pytest tests/test_vector_utils.py -v
pytest tests/test_vehicle_state.py -v
pytest tests/test_track.py -v
pytest tests/test_neat_driver.py -v
pytest tests/test_persistence.py -v

# Run linter
ruff check src/

# Run type checker
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
| `test_ai_*.py` | Driver interface, training manager, replay loader |
| `test_simulation_engine.py` | Headless engine step/lifecycle orchestration |
| `test_analytics_run_metrics.py` | Metrics collection and aggregation |
| `test_ui_*.py` | HUD layout helpers and menu navigation state |
| `test_main_bootstrap.py` | CLI argument parsing and bootstrap helpers |
| `test_properties.py` | Hypothesis property tests across modules |

## Project Goals

- Refactor the monolithic `newcar.py` prototype into a clean, modular Python package
- Separate concerns across domain, core, AI, UI, simulation, analytics, and persistence layers
- Establish a comprehensive test suite (428 tests, all headless)
- Enable independent development and testing of each layer
- Support training, replay, and future manual-drive modes via a shared `DriverInterface`
