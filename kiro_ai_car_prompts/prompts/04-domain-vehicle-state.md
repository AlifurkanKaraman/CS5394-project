# CRITICAL REQUIREMENTS - Kiro Prompt for VehicleState domain model

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/domain/vehicle_state.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `VehicleState domain model` component.

### MANDATORY PURPOSE
Represent the mutable physical and gameplay state of a car in the simulation.

### CRITICAL RESPONSIBILITIES
- Track position, angle, speed, alive/crashed state, distance traveled, elapsed time, and reward-related metrics
- Keep this as a lightweight domain object
- Make the state easy to inspect, serialize, and test

### MANDATORY IMPLEMENTATION DETAILS
Implement a dataclass `VehicleState` containing at minimum:
- `x: float`
- `y: float`
- `angle: float`
- `speed: float`
- `alive: bool = True`
- `distance_travelled: float = 0.0`
- `time_steps: int = 0`
- optional `lap_progress`, `checkpoint_index`, or `fitness` fields

### MANDATORY METHODS
Include helpers such as:
- `position_tuple()`
- `mark_crashed()`
- `advance_time()`
- optional `to_dict()` / `from_dict()`

### CRITICAL DEPENDENCIES
Use only light standard-library dependencies.

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
- Add pytest coverage for state mutation helpers.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/domain/vehicle_state.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
