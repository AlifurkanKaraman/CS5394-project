# CRITICAL REQUIREMENTS - Kiro Prompt for Training manager

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ai/training_manager.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Training manager` component.

### MANDATORY PURPOSE
Orchestrate NEAT population setup, reporters, generation callbacks, and simulation evaluation.

### CRITICAL RESPONSIBILITIES
- Move training bootstrap out of the original script
- Keep training lifecycle explicit and configurable
- Coordinate with simulation engine rather than doing everything inline

### MANDATORY IMPLEMENTATION DETAILS
Implement a `TrainingManager` class that can:
- load NEAT config
- create population
- register reporters
- launch population training for N generations
- evaluate genomes via the simulation engine

### MANDATORY METHODS
Include methods similar to:
- `load_config(path)`
- `create_population()`
- `run_training()`
- `evaluate_genomes(genomes, neat_config)`

### CRITICAL DEPENDENCIES
Depend on NEAT, config, persistence, analytics, and simulation engine where needed.

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
- Add lightweight tests around configuration loading or collaboration boundaries where feasible.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ai/training_manager.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
