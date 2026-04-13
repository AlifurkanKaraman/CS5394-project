# CRITICAL REQUIREMENTS - Kiro Prompt for Python Dependencies and Tooling

## MANDATORY DIRECTIVE
You are an expert Python build and tooling engineer.

**CRITICAL**: Configure the upgraded AI car project with proper Python dependencies, test tooling, code quality tooling, and run commands.

## TASK
Create the dependency and tooling configuration for the project.

### MANDATORY IMPLEMENTATION
Use either `pyproject.toml` or `requirements.txt` + tool configs, but prefer `pyproject.toml`.

### CRITICAL DEPENDENCIES
Include at minimum:
- Python 3.11+
- `pygame`
- `neat-python`
- `pytest`
- `pytest-cov`
- optionally `mypy`, `ruff`, or `black`

### MANDATORY TOOLING
Configure:
- installable package from `src/`
- test discovery for `tests/`
- simple command or documented way to run the app
- simple command or documented way to run tests

### MANDATORY FILES
Create or update:
- `pyproject.toml`
- `README.md`
- optionally `.python-version`
- optionally tool sections for pytest, ruff, black, mypy

### CRITICAL RULES
- Do not add heavy unnecessary dependencies
- Keep the stack aligned with the original repo’s Python/Pygame/NEAT focus
- Ensure paths support `src` layout
- Ensure `python -m ai_car_sim.main` or equivalent can be used

### MANDATORY ACCEPTANCE
- Project installs without malformed config
- Test runner can discover tests
- Package imports resolve from `src/`

## CRITICAL REQUIREMENT
**MANDATORY**: Produce correct Python dependency configuration and developer tooling so the remaining prompts can build on a stable foundation.
