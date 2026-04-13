# AI Car Simulation

A NEAT-based neuroevolution self-driving car simulation using pygame.

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

```bash
# Run all tests
pytest

# With coverage
pytest --cov=ai_car_sim

# Run linter
ruff check src/

# Run type checker
mypy src/
```

## Project Goals

- Refactor the monolithic `newcar.py` prototype into a clean, modular Python package
- Separate concerns across domain, core, AI, UI, simulation, analytics, and persistence layers
- Establish a test scaffold for incremental implementation and validation
- Enable independent development and testing of each layer
