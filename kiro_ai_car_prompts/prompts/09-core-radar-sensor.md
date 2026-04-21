# CRITICAL REQUIREMENTS - Kiro Prompt for Radar sensor system

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/core/radar_sensor.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Radar sensor system` component.

### MANDATORY PURPOSE
Encapsulate the radar logic that scans outward from the car at configured angle offsets.

### CRITICAL RESPONSIBILITIES
- Replace radar code embedded in the original Car class
- Support configurable angle offsets and max distance
- Return structured `SensorReading` objects
- Keep drawing separate from sensing if possible

### MANDATORY IMPLEMENTATION DETAILS
Implement a `RadarSensorSystem` or similar class with configuration for:
- angle offsets
- max distance
- border color

### MANDATORY METHODS
Include methods similar to:
- `scan(vehicle_state, game_map) -> list[SensorReading]`
- internal raycast helper
- optional `normalized_inputs(...) -> list[float]`

### CRITICAL DEPENDENCIES
Depend on `SensorReading`, math helpers, and pygame surface access where needed.

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
- Add at least one test for input formatting or angle offset handling.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/core/radar_sensor.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
