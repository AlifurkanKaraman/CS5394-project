# AI Car Simulation

A NEAT-based neuroevolution self-driving car simulation using pygame.

## Architecture

The project follows a layered architecture under `src/ai_car_sim/`:

- `domain/` — pure data models (Track, VehicleState, SensorReading, SimulationConfig)
- `core/` — physics, collision detection, and radar sensing
- `ai/` — driver abstraction and NEAT integration
- `ui/` — pygame rendering and menu logic
- `simulation/` — main simulation loop engine
- `analytics/` — per-run metrics collection
- `persistence/` — save/load for genomes and metrics

## Running

Install dependencies and run the simulation:

```bash
pip install -e .
python -m ai_car_sim.main
```

## Testing

Run the test suite with pytest:

```bash
pytest
```

## Project Goals

- Refactor the monolithic `newcar.py` prototype into a clean, modular Python package
- Separate concerns across domain, core, AI, UI, simulation, analytics, and persistence layers
- Establish a test scaffold for incremental implementation and validation
- Enable independent development and testing of each layer
