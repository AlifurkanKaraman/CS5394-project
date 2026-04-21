# CRITICAL REQUIREMENTS - Kiro Prompt for Simulation engine

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/simulation/engine.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Simulation engine` component.

### MANDATORY PURPOSE
Implement the main simulation loop and orchestration for cars, drivers, maps, HUD, collisions, and runtime modes.

### CRITICAL RESPONSIBILITIES
- This is the heart of the system
- Centralize the frame loop without recreating the original monolithic script
- Support training and replay modes
- Coordinate cars, drivers, map loading, drawing, timing, and result collection

### MANDATORY IMPLEMENTATION DETAILS
Implement a `SimulationEngine` class responsible for:
- pygame initialization
- loading the selected track map
- creating cars
- stepping the simulation
- applying driver decisions
- updating HUD
- terminating a run on timeout or zero alive cars

### MANDATORY METHODS
Include methods such as:
- `initialize()`
- `create_cars_for_genomes(...)`
- `run_generation(...)`
- `run_replay(...)`
- `step(...)`
- `shutdown()`

### CRITICAL DEPENDENCIES
Compose previously created modules instead of duplicating their logic.

### ENGINEERING RULES
- Use Python 3.11+ syntax
- Use type hints
- Write clean docstrings for public classes and functions
- Keep the module focused on its own responsibility
- Do not embed unrelated UI or training logic here
- Prefer small helper methods over long procedural blocks

### MANDATORY VALIDATION
- File imports cleanly
- Public API is clear and stable
- Logic is testable
- Add unit tests for non-rendering orchestration helpers where practical.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/simulation/engine.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
