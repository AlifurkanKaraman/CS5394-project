# CRITICAL REQUIREMENTS - Kiro Prompt for Menu controller

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ui/menu_controller.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Menu controller` component.

### MANDATORY PURPOSE
Provide a keyboard-friendly menu or scene selection layer for starting training, replay, and choosing tracks.

### CRITICAL RESPONSIBILITIES
- Make the upgraded project feel like a bigger application
- Keep menu navigation isolated from simulation logic
- Support simple academic demo flows

### MANDATORY IMPLEMENTATION DETAILS
Implement a `MenuController` with support for:
- start training
- replay best model
- select track
- quit application

### MANDATORY METHODS
Include methods similar to:
- `handle_event(event)`
- `draw(screen)`
- `get_selected_mode()`
- `get_selected_track()`

### CRITICAL DEPENDENCIES
Use pygame input/rendering but keep business logic thin.

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
- Add small logic tests for menu selection state if feasible.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ui/menu_controller.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
