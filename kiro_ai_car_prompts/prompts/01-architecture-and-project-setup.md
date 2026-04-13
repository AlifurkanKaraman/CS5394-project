# CRITICAL REQUIREMENTS - Kiro Prompt for Architecture and Project Setup

## MANDATORY DIRECTIVE
You are an expert Python software architect.

**CRITICAL**: Refactor the repository `NeuralNine/ai-car-simulation` into a **large-scale Python project**. The current repository is a minimal prototype with one main Python script and a NEAT config. You must preserve the core self-driving car simulation concept while creating a production-like academic project structure.

## TASK
Create the complete folder structure, package layout, and empty starter files for a modular project.

### MANDATORY OUTPUT
Create this structure or an extremely close equivalent:

```text
assets/
  cars/
  maps/
configs/
outputs/
src/
  ai_car_sim/
    __init__.py
    main.py
    app/__init__.py
    ai/__init__.py
    core/__init__.py
    domain/__init__.py
    ui/__init__.py
    simulation/__init__.py
    analytics/__init__.py
    persistence/__init__.py
tests/
```

### CRITICAL IMPLEMENTATION RULES
- Move original image/config assets into clearly named folders without breaking paths
- Create package `src/ai_car_sim`
- Add placeholder modules for the classes defined in later prompts
- Create `README.md` section headings for architecture, running, testing, and project goals
- Create `.gitignore`
- Create `outputs/` for saved models, metrics, and logs

### MANDATORY FILE PLACEHOLDERS
Create empty or stub files for:
- `domain/track.py`
- `domain/vehicle_state.py`
- `domain/sensor_reading.py`
- `domain/simulation_config.py`
- `core/vector_utils.py`
- `core/collision_service.py`
- `core/radar_sensor.py`
- `core/car.py`
- `ai/driver_interface.py`
- `ai/neat_driver.py`
- `ai/training_manager.py`
- `ai/replay_loader.py`
- `ui/hud_view.py`
- `ui/menu_controller.py`
- `simulation/engine.py`
- `analytics/run_metrics.py`
- `persistence/save_load.py`

## MANDATORY VALIDATION
- All directories and files must exist
- All Python packages must include `__init__.py`
- Do not implement full logic yet
- Ensure the layout is ready for incremental prompt-based development

## CRITICAL REQUIREMENT
**MANDATORY**: Create the modular project structure exactly and prepare the repository for the remaining prompts. Do not leave the project as a single-file script.
