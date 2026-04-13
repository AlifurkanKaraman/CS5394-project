# CRITICAL REQUIREMENTS - Kiro Prompt for Run metrics module

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/analytics/run_metrics.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `Run metrics module` component.

### MANDATORY PURPOSE
Collect and summarize metrics for training generations and replay sessions.

### CRITICAL RESPONSIBILITIES
- Turn the project into a more serious experiment platform
- Track data that can be saved and discussed in a report
- Avoid mixing analytics with rendering logic

### MANDATORY IMPLEMENTATION DETAILS
Implement dataclasses or classes for metrics such as:
- generation number
- alive count
- best fitness
- average fitness
- elapsed seconds
- selected track
- output artifact paths

### MANDATORY METHODS
Include methods such as:
- `record_generation(...)`
- `record_run_summary(...)`
- `to_dict()` or `to_rows()`

### CRITICAL DEPENDENCIES
Keep it storage-agnostic where possible.

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
- Add tests for summary aggregation or serialization.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/analytics/run_metrics.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
