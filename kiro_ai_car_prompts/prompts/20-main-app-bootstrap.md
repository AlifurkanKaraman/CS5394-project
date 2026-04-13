# CRITICAL REQUIREMENTS - Kiro Prompt for Main application bootstrap

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/main.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Main application bootstrap` component.

### MANDATORY PURPOSE
Compose the app, parse or select mode, and launch menu, training, or replay workflows.

### CRITICAL RESPONSIBILITIES
- Replace the original `if __name__ == '__main__'` block with clean composition
- Keep startup responsibilities centralized
- Make the project runnable as a proper Python package

### MANDATORY IMPLEMENTATION DETAILS
Implement:
- application bootstrap function
- loading of config and default tracks
- wiring of simulation engine, persistence, analytics, and training manager
- entrypoint guard or package entrypoint

### MANDATORY METHODS
Include:
- `main()`
- helper setup functions as needed

### CRITICAL DEPENDENCIES
Depend on the modular services built in previous prompts.

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
- Add at least one smoke-oriented test around config/bootstrap helpers where possible.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/main.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
