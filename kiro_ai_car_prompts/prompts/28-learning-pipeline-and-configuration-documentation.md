### CRITICAL REQUIREMENTS ###

You are an expert Python software engineer and technical documenter working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Analyze the existing project and produce clear in-project documentation and code comments that explain exactly where the learning algorithm is implemented, how the car learning process works, and where configuration values such as speed, acceleration, steering, and NEAT parameters are defined.

**MANDATORY**: Apply changes directly to repository files.

**MANDATORY**: Do not change working behavior unless absolutely necessary for clarity or tiny safe refactors.

**MANDATORY**: Keep the explanation accurate to the actual current codebase.

**MANDATORY**: Use Python 3.11+ syntax and clean technical writing.

**MANDATORY**: Execute this now, not just describe what should be done.

### FEATURE GOAL ###

Make the project easy to understand for a student, professor, or reviewer by clearly documenting:

1. where the NEAT algorithm is integrated

2. how training and learning happen generation by generation

3. how cars receive inputs and produce actions

4. how fitness is calculated

5. where manual speed, AI speed, steering, and other runtime settings are configured

### REQUIRED CHANGES ###

#### 1. Create a Training / Algorithm Explanation Document

**MANDATORY**: Create a new documentation file such as:

`docs/how-learning-works.md`

This file must clearly explain, based on the real codebase:

- which files implement the NEAT/training pipeline

- where genomes are created/evaluated

- where one genome is mapped to one car

- where sensor inputs are collected

- where neural network outputs are converted into actions

- where car fitness/reward is calculated

- where new generations are created

- where manual mode differs from training mode

- where replay mode differs from training mode

**CRITICAL**: Include exact file paths and important class/function names from the current project.

#### 2. Add a “Code Map” Section

**MANDATORY**: In the new documentation file, add a section named something like:

`Code Map of the Learning Pipeline`

It must list the key files and their responsibilities, for example:

- `src/ai_car_sim/ai/training_manager.py`

- `src/ai_car_sim/ai/neat_driver.py`

- `src/ai_car_sim/simulation/engine.py`

- `src/ai_car_sim/core/car.py`

- `src/ai_car_sim/domain/simulation_config.py`

For each file, explain what part of the algorithm it controls.

#### 3. Explain the Learning Flow Step by Step

**MANDATORY**: In the documentation, include a plain-English step-by-step explanation of the learning loop, such as:

- population creation

- generation start

- spawn cars

- collect sensor data

- run neural network

- choose actions

- update car state

- compute fitness

- end generation

- evolve next generation

**CRITICAL**: Base this on the real project implementation, not generic NEAT theory only.

#### 4. Document Configuration for Speed and Controls

**MANDATORY**: Explain clearly where the following are configured in code:

- default car speed

- acceleration

- braking

- steering / rotation speed

- radar distance

- FPS

- generation timeout

- manual mode speed differences

- simulation speed multiplier if implemented

- NEAT config file location and purpose

**CRITICAL**:

- Include exact file paths

- Include exact class names / variables / methods where possible

- If configuration is split across multiple files, explain that clearly

#### 5. Add Helpful Inline Comments in Code

**MANDATORY**: Add concise, high-value comments/docstrings in the most important implementation points, such as:

- where genome fitness is assigned

- where driver decisions are made

- where car actions are applied

- where a new generation begins

- where config values are loaded

**CRITICAL**:

- Do not spam comments everywhere

- Add comments only where they improve understanding

- Keep comments accurate and professional

#### 6. Add a “How to Explain This to a Professor” Section

**MANDATORY**: In the new documentation file, add a short section that explains the training approach in simple academic language.

It should answer:

- what algorithm is used

- why NEAT was chosen

- how the car learns over generations

- what inputs and outputs the car uses

- how better behavior is selected

#### 7. Add a “How to Tune the Project” Section

**MANDATORY**: Include a section showing where someone can tune:

- car speed

- steering sensitivity

- radar angles / distance

- number of generations

- population size

- fitness behavior

- map/track settings

**CRITICAL**: Use exact file references.

### FILE TARGETS ###

Modify the existing architecture appropriately. Use the current project structure and adapt as needed.

Likely files to modify/create include:

`docs/how-learning-works.md`

`README.md`

`src/ai_car_sim/ai/training_manager.py`

`src/ai_car_sim/ai/neat_driver.py`

`src/ai_car_sim/simulation/engine.py`

`src/ai_car_sim/core/car.py`

`src/ai_car_sim/domain/simulation_config.py`

You may update other files if necessary for clarity, but avoid unnecessary behavioral changes.

### ENGINEERING RULES ###

**MANDATORY**:

- Keep all explanations aligned to the actual code

- Do not invent nonexistent classes or functions

- If some behavior is incomplete or buggy, document that honestly

- Keep documentation readable for a student audience

- Use structured markdown headings

- Use clear type hints and docstrings where code changes are made

- Preserve runtime behavior unless a tiny safe clarification refactor is needed

### TESTING REQUIREMENTS ###

**CRITICAL**: Verify all of the following after implementation:

1. The new documentation file exists and is readable

2. It accurately names the files/classes/functions responsible for learning

3. It explains where speed and behavior configs are stored

4. Added comments/docstrings do not break the code

5. The application still runs normally

6. The documentation helps a reviewer understand how training actually works

### OUTPUT REQUIREMENTS ###

**MANDATORY**: Modify repository files directly.

**MANDATORY**: Do not only acknowledge the request.

**MANDATORY**: Do not only explain what should be done.

**MANDATORY**: Do not output a plan.

**MANDATORY**: After implementing, provide a concise summary of:

- created/changed files

- where the algorithm is implemented

- where speed/config values are defined

- what documentation was added

### CRITICAL REQUIREMENTS ###

**MANDATORY**: Fully implement this documentation and code-comment improvement now so the project clearly shows where the learning algorithm lives, how the car learns, and where speed/configuration values are controlled.