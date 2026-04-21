# CRITICAL REQUIREMENTS - Kiro Prompt for Car entity

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/core/car.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Car entity` component.

### MANDATORY PURPOSE
Implement the main car entity by composing vehicle state, sensors, collision checks, and sprite handling.

### CRITICAL RESPONSIBILITIES
- Replace the giant original Car class with a cleaner design
- Keep movement, state updates, reward calculation, and action application together
- Delegate sensing and collisions to dedicated modules
- Support AI and replay/manual control modes

### MANDATORY IMPLEMENTATION DETAILS
Implement a `Car` class that owns:
- `VehicleState`
- sprite / rotated sprite references or asset path metadata
- `RadarSensorSystem`
- references to config and track context as needed

### MANDATORY METHODS
Implement methods such as:
- `reset(track, config)`
- `apply_action(action)`
- `update(game_map)`
- `get_sensor_inputs() -> list[float]`
- `get_reward() -> float`
- `draw(screen)` (drawing may remain here)
- `is_alive() -> bool`

### CRITICAL DEPENDENCIES
Use `VehicleState`, `SimulationConfig`, `RadarSensorSystem`, and collision service.

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
- Add tests for action application and reward behavior where practical.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/core/car.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
