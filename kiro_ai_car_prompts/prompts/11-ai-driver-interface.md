# CRITICAL REQUIREMENTS - Kiro Prompt for AI driver interface

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ai/driver_interface.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `AI driver interface` component.

### MANDATORY PURPOSE
Define the abstraction that produces driving actions from sensor/state inputs.

### CRITICAL RESPONSIBILITIES
- Decouple car control from a specific AI algorithm
- Make it possible to add manual, scripted, or other ML drivers later

### MANDATORY IMPLEMENTATION DETAILS
Implement:
- an `Enum` or constants for possible actions
- a `Protocol`, abstract base class, or interface `DriverInterface`
- a method like `decide_action(sensor_inputs, state) -> Action`

### MANDATORY METHODS
Include action definitions and the abstract decision method.

### CRITICAL DEPENDENCIES
Keep this module lightweight and framework-agnostic.

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
- Add a tiny test validating enum/interface presence if appropriate.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ai/driver_interface.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
