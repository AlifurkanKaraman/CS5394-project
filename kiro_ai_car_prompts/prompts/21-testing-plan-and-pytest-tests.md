# CRITICAL REQUIREMENTS - Kiro Prompt for Testing Plan and Pytest Suite

## MANDATORY DIRECTIVE
You are an expert Python test engineer.

**CRITICAL**: Add a meaningful pytest suite for the upgraded AI car simulation project.

## TASK
Create a practical testing plan and implement tests for the most testable modules.

### MANDATORY TEST TARGETS
Prioritize:
- domain dataclasses and serialization
- math helpers in `vector_utils`
- collision helper corner calculations
- sensor reading normalization
- driver action mapping
- persistence path and serialization helpers

### CRITICAL RULES
- Do not over-focus on fragile full-pygame integration tests
- Prefer deterministic unit tests
- Use stubs/fakes for NEAT networks where possible
- Keep tests fast and isolated

### MANDATORY FILES
Create test files such as:
- `tests/test_track.py`
- `tests/test_vehicle_state.py`
- `tests/test_vector_utils.py`
- `tests/test_neat_driver.py`
- `tests/test_persistence.py`

### MANDATORY DOCUMENTATION
Add a short testing section to the README with the exact command to run tests.

## CRITICAL REQUIREMENT
**MANDATORY**: Provide a realistic test suite that strengthens the project and demonstrates software engineering discipline for academic assessment.
