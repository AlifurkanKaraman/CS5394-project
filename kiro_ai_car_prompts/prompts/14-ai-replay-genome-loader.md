# CRITICAL REQUIREMENTS - Kiro Prompt for Replay genome loader

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ai/replay_loader.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Replay genome loader` component.

### MANDATORY PURPOSE
Support loading a previously saved best genome/model and replaying it in the simulation without training.

### CRITICAL RESPONSIBILITIES
- Create a dedicated replay path for demos and professor evaluation
- Avoid mixing replay with training orchestration
- Support clear errors when files are missing or incompatible

### MANDATORY IMPLEMENTATION DETAILS
Implement functionality to:
- load a serialized best genome/model
- rebuild the NEAT network from saved artifacts
- return a driver usable by the simulation engine

### MANDATORY METHODS
Include methods similar to:
- `load_best_genome(file_path)`
- `build_driver_from_saved_genome(...)`
- optional `validate_saved_run(...)`

### CRITICAL DEPENDENCIES
Integrate with persistence and NEAT config handling.

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
- Add a test for invalid path handling or serialization helpers.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ai/replay_loader.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
