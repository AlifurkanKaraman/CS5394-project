# CRITICAL REQUIREMENTS - Kiro Prompt for NEAT driver adapter

## MANDATORY DIRECTIVE
You are an expert Python engineer.

**CRITICAL**: Implement `src/ai_car_sim/ai/neat_driver.py` as part of the upgraded large-scale AI car simulation project derived from `NeuralNine/ai-car-simulation`.

## TASK
Implement the `NEAT driver adapter` component.

### MANDATORY PURPOSE
Wrap a NEAT neural network so it conforms to the driver interface.

### CRITICAL RESPONSIBILITIES
- Translate sensor inputs into network activations
- Map network outputs to domain actions cleanly
- Isolate NEAT-specific logic from the rest of the app

### MANDATORY IMPLEMENTATION DETAILS
Implement a `NeatDriver` class that:
- accepts a NEAT network object
- implements the driver interface
- maps the max-output neuron to a driving action

### MANDATORY METHODS
Include:
- constructor storing the network
- `decide_action(sensor_inputs, state) -> Action`
- optional validation for output dimensions

### CRITICAL DEPENDENCIES
Depend on `neat-python` only where necessary.

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
- Add at least one unit test using a fake network stub.

## CRITICAL REQUIREMENT
**MANDATORY**: Implement `src/ai_car_sim/ai/neat_driver.py` fully and cleanly, with no placeholder logic beyond what is explicitly necessary.
