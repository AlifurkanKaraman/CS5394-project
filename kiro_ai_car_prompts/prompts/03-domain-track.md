# CRITICAL REQUIREMENTS - Kiro Prompt for Track domain model

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/domain/track.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Track domain model` component.

### MANDATORY PURPOSE
Represent a drivable map/track, its asset paths, spawn point, optional checkpoints, and metadata needed by the simulation.

### CRITICAL RESPONSIBILITIES
- Store map image path and human-readable track name
- Store spawn position and initial heading
- Optionally store checkpoints, finish line, border color, and bounds
- Provide helpers needed by the simulation layer without depending on pygame-heavy rendering code

### MANDATORY IMPLEMENTATION DETAILS
Implement a `Track` dataclass with fields similar to:
- `name: str`
- `map_image_path: str`
- `spawn_position: tuple[float, float]`
- `spawn_angle: float`
- `border_color: tuple[int, int, int] | tuple[int, int, int, int]`
- `checkpoints: list[tuple[int, int]] | list[tuple[tuple[int, int], tuple[int, int]]] | None = None`
- `description: str = ""`

### MANDATORY METHODS
Add methods such as:
- `from_dict(...)`
- `to_dict(...)`
- optional helper validation methods
- optional helper to normalize asset paths

### CRITICAL DEPENDENCIES
This module may depend on standard library and typing/dataclasses only. Avoid direct pygame rendering in the domain layer.

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
- Add at least one simple pytest validating dataclass creation or serialization.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/domain/track.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
