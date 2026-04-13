# CRITICAL REQUIREMENTS - Kiro Prompt for Class Diagram and Documentation

## MANDATORY DIRECTIVE
You are an expert software architect and technical writer.

**CRITICAL**: Create architecture documentation and a class diagram for the upgraded Python AI car simulation project.

## TASK
Document the project clearly for a professor or reviewer.

### MANDATORY OUTPUT
Create or update documentation that includes:
- project overview
- original repository baseline and what was expanded
- package responsibilities
- runtime modes: training, replay, optional manual/demo mode
- major classes and how they collaborate
- testing strategy
- future extension ideas

### CRITICAL CLASS DIAGRAM REQUIREMENT
Provide a class diagram in **Mermaid** format inside the README or a dedicated docs file.

### MANDATORY CLASSES TO INCLUDE
At minimum include:
- `SimulationEngine`
- `Car`
- `VehicleState`
- `RadarSensorSystem`
- `Track`
- `SimulationConfig`
- `DriverInterface`
- `NeatDriver`
- `TrainingManager`
- `HudView`
- `MenuController`
- `RunMetrics`
- `SaveLoadService` or equivalent persistence class

### MANDATORY DIAGRAM QUALITY
The diagram must show:
- composition/aggregation where appropriate
- key dependencies
- direction of collaboration
- enough detail to explain the architecture without becoming cluttered

### CRITICAL RULES
- Keep docs aligned with the implemented code
- Do not document nonexistent features as completed
- Use concise, professional language

## CRITICAL REQUIREMENT
**MANDATORY**: Produce polished architecture documentation and a Mermaid class diagram suitable for inclusion in a university project submission.
