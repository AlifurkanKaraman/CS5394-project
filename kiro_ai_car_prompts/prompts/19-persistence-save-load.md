# CRITICAL REQUIREMENTS - Kiro Prompt for Persistence module

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/persistence/save_load.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Persistence module` component.

### MANDATORY PURPOSE
Save and load best genomes, run metrics, and optional JSON summaries.

### CRITICAL RESPONSIBILITIES
- Make long training outputs reusable
- Provide reproducibility for demos and grading
- Keep persistence concerns out of AI and UI layers

### MANDATORY IMPLEMENTATION DETAILS
Implement helpers or a service class to:
- ensure output directories exist
- save best genome/model artifacts
- save metrics JSON or CSV
- load saved genome/model artifacts
- handle file errors gracefully

### MANDATORY METHODS
Include methods like:
- `ensure_output_dirs(...)`
- `save_best_genome(...)`
- `load_best_genome(...)`
- `save_metrics_json(...)`
- optional `save_metrics_csv(...)`

### CRITICAL DEPENDENCIES
Use standard library serialization unless a tiny extra dependency is clearly justified.

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
- Add tests for output path creation and serialization.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/persistence/save_load.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
