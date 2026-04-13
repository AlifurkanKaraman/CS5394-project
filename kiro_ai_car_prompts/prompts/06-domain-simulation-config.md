# CRITICAL REQUIREMENTS - Kiro Prompt for SimulationConfig domain model

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/domain/simulation_config.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `SimulationConfig domain model` component.

### MANDATORY PURPOSE
Centralize configurable parameters for physics, sensors, training loop, screen sizing, and runtime modes.

### CRITICAL RESPONSIBILITIES
- Replace hardcoded constants from the original single-file script
- Keep runtime configuration grouped and explicit
- Support future extension for multiple maps and modes

### MANDATORY IMPLEMENTATION DETAILS
Implement a dataclass `SimulationConfig` with fields for:
- screen width and height
- car sprite size
- max radar distance
- default speed
- max generations
- fps
- training timeout / steps per generation
- fullscreen flag
- selected map name or path
- output directories

### MANDATORY METHODS
Include:
- `from_dict(...)`
- `to_dict(...)`
- `load_from_json(...)` or `load_from_yaml(...)` only if you keep dependencies light

### CRITICAL DEPENDENCIES
Prefer standard library parsing if using JSON.

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
- Add pytest for configuration creation and serialization.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/domain/simulation_config.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
