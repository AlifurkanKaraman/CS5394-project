# CRITICAL REQUIREMENTS - Kiro Prompt for Vector utility module

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/core/vector_utils.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Vector utility module` component.

### MANDATORY PURPOSE
Provide geometry helpers used by car motion, radar calculations, angle conversion, and distance measurements.

### CRITICAL RESPONSIBILITIES
- Encapsulate repeated math from the original `newcar.py`
- Keep helpers pure and testable
- Reduce duplicated formulas across modules

### MANDATORY IMPLEMENTATION DETAILS
Implement focused pure functions such as:
- `distance(x1, y1, x2, y2) -> float`
- `rotate_point(cx, cy, angle_deg, length) -> tuple[float, float]`
- `clamp(value, min_value, max_value) -> float`
- `heading_to_vector(angle_deg) -> tuple[float, float]`

### MANDATORY METHODS
Implement the functions listed above and any small helper required.

### CRITICAL DEPENDENCIES
Use only `math` and typing.

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
- Add pytest for distance, clamp, and angle helper behavior.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/core/vector_utils.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
