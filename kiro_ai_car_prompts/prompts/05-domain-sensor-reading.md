# CRITICAL REQUIREMENTS - Kiro Prompt for SensorReading domain model

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/domain/sensor_reading.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `SensorReading domain model` component.

### MANDATORY PURPOSE
Represent one radar/sensor reading from the car to surrounding borders or obstacles.

### CRITICAL RESPONSIBILITIES
- Capture endpoint, distance, angle offset, and optionally normalized distance
- Keep the representation serializable and easy to feed into AI drivers

### MANDATORY IMPLEMENTATION DETAILS
Implement a dataclass `SensorReading` with fields like:
- `angle_offset: float`
- `hit_x: int`
- `hit_y: int`
- `distance: float`
- optional `normalized_distance: float | None = None`

### MANDATORY METHODS
Include helpers such as:
- `as_input_value(max_distance: float) -> float`
- `hit_point() -> tuple[int, int]`

### CRITICAL DEPENDENCIES
No pygame rendering logic inside the data model.

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
- Add pytest for normalization behavior.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/domain/sensor_reading.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
