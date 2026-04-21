# CRITICAL REQUIREMENTS - Complete Kiro Implementation Guide for Expanded AI Car Simulation

## MANDATORY DIRECTIVE
You are an expert Python game and AI engineer.

**CRITICAL**: Follow this implementation guide to transform the GitHub repository `NeuralNine/ai-car-simulation` into a **large-scale, modular Python project** with a clean architecture, multiple gameplay and AI features, analytics, persistence, and tests.

**CRITICAL**: The source project is a compact Python/Pygame/NEAT prototype built around:
- `newcar.py`
- `config.txt`
- map and sprite image assets

**MANDATORY**: Refactor this into a professional project structure without losing the core idea of an AI car learning to drive on a track.

## CRITICAL
Access the prompts referenced below in files with corresponding filenames in directory `prompts` located in the root project directory.

## IMPLEMENTATION EXECUTION ORDER

### **CRITICAL**: Execute prompts in this exact sequence:
1. **01-architecture-and-project-setup.md** - Create project structure and migration plan
2. **02-python-dependencies-and-tooling.md** - Configure Python dependencies and developer tooling
3. **03-domain-track.md** - Implement Track domain model
4. **04-domain-vehicle-state.md** - Implement VehicleState domain model
5. **05-domain-sensor-reading.md** - Implement SensorReading domain model
6. **06-domain-simulation-config.md** - Implement SimulationConfig domain model
7. **07-core-vector-utils.md** - Implement math and geometry helpers
8. **08-core-collision-service.md** - Implement collision logic
9. **09-core-radar-sensor.md** - Implement radar sensor system
10. **10-core-car-entity.md** - Implement Car entity
11. **11-ai-driver-interface.md** - Implement AI driver interface
12. **12-ai-neat-driver.md** - Implement NEAT driver adapter
13. **13-ai-training-manager.md** - Implement training orchestration
14. **14-ai-replay-genome-loader.md** - Implement best-genome loading and replay support
15. **15-ui-hud-view.md** - Implement HUD rendering
16. **16-ui-menu-controller.md** - Implement menu and scene navigation
17. **17-simulation-engine.md** - Implement simulation loop and orchestration
18. **18-analytics-run-metrics.md** - Implement metrics collection and reporting
19. **19-persistence-save-load.md** - Implement save/load and experiment output handling
20. **20-main-app-bootstrap.md** - Implement main entrypoint and app composition
21. **21-testing-plan-and-pytest-tests.md** - Add tests and validation scaffolding
22. **22-class-diagram-and-docs.md** - Generate architecture documentation and class diagram

## MANDATORY: Target Expanded Feature Set

**CRITICAL**: The finished upgraded project must include these additional features beyond the original repository:
- Modular package structure instead of one large script
- Support for multiple tracks/maps
- Configurable spawn points and simulation settings
- Sensor subsystem separated from the car entity
- AI abstraction layer so additional algorithms can be added later
- Training mode and replay mode
- HUD with generation, alive cars, best fitness, speed, lap/checkpoint information if available
- Save/load support for best genomes and run summaries
- Analytics output for experiments
- Keyboard-friendly menu/navigation layer
- Unit tests for domain and core systems

## MANDATORY: Verification Between Steps

**CRITICAL**: After each step:
1. Fix any syntax or import errors before proceeding
2. Verify all files are in correct packages
3. Ensure public interfaces match the previous prompts
4. Keep backward compatibility with migrated assets when practical
5. Avoid circular imports

## MANDATORY: Final Project Testing

**CRITICAL**: After completing all steps:
1. Create and activate a Python virtual environment
2. Install dependencies successfully
3. Run the test suite successfully
4. Launch the application in replay/manual validation mode
5. Launch training mode and verify the loop starts correctly
6. Verify maps load correctly
7. Verify radars/sensors update correctly
8. Verify collisions eliminate cars correctly
9. Verify output artifacts are written to disk
10. Verify best model replay works

## MANDATORY: Expected Architecture

**CRITICAL**: Final structure must resemble:

```text
ai_car_simulation/
  assets/
  configs/
  outputs/
  src/
    ai_car_sim/
      __init__.py
      main.py
      app/
      ai/
      core/
      domain/
      ui/
      analytics/
      persistence/
      simulation/
  tests/
```

## MANDATORY: Engineering Standards

**CRITICAL**: All generated code must:
- Use Python 3.11+
- Use type hints consistently
- Use dataclasses where appropriate
- Keep functions focused and testable
- Prefer composition over giant classes
- Document assumptions in docstrings
- Avoid hardcoding values that belong in config files

## MANDATORY: Success Criteria

**CRITICAL**: Project is complete when:
- The original prototype is successfully evolved into a large modular codebase
- The app launches without import/runtime errors
- Training and replay workflows both function
- The code is testable and partially covered by pytest
- The class diagram and docs reflect the implemented design

## CRITICAL REQUIREMENT
**MANDATORY**: Follow each prompt exactly as specified. Do not collapse everything into one file. Do not skip packages, interfaces, tests, or documentation. Preserve the original repository’s purpose while upgrading it into a substantially larger Python project suitable for a university software engineering submission.
