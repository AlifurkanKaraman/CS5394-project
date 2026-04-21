# CRITICAL REQUIREMENTS - Kiro Prompt for Collision service

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/core/collision_service.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Collision service` component.

### MANDATORY PURPOSE
Handle car corner calculation and collision checks against the map border color.

### CRITICAL RESPONSIBILITIES
- Move collision logic out of the car entity
- Calculate rotated corners from vehicle state and sprite size
- Detect whether any corner intersects border pixels
- Keep the service reusable and isolated

### MANDATORY IMPLEMENTATION DETAILS
Implement a class or pure-function module that can:
- compute rotated rectangle corners
- inspect the map surface for border collisions
- safely handle out-of-bounds checks

### MANDATORY METHODS
Include methods/functions similar to:
- `compute_corners(state, car_size) -> list[tuple[float, float]]`
- `is_collision(corners, game_map, border_color) -> bool`
- optional `is_out_of_bounds(...) -> bool`

### CRITICAL DEPENDENCIES
This module may use pygame surface pixel lookup, but keep rendering concerns separate.

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
- Add tests for corner calculation. If surface mocking is easier than full pygame setup, do that.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/core/collision_service.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
