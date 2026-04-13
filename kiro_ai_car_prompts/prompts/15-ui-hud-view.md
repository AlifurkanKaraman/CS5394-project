# CRITICAL REQUIREMENTS - Kiro Prompt for HUD view

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ui/hud_view.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `HUD view` component.

### MANDATORY PURPOSE
Render simulation status information on screen in a clean reusable component.

### CRITICAL RESPONSIBILITIES
- Separate status drawing from the engine loop
- Show core run information cleanly
- Make it easy to extend with more metrics later

### MANDATORY IMPLEMENTATION DETAILS
Implement a `HudView` class that can display:
- current generation
- alive cars
- best fitness
- selected track
- current mode (training/replay/manual)
- optional speed or sensor summaries

### MANDATORY METHODS
Include:
- `draw(screen, metrics_or_state)`
- optional helper methods for text rendering and layout

### CRITICAL DEPENDENCIES
Depend on pygame rendering only, not on NEAT internals.

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
- Keep logic testable where practical; pure layout helpers may be unit tested.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ui/hud_view.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
