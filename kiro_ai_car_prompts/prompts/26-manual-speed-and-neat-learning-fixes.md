### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Modify the existing project to improve Manual Mode driving usability and fix training/evolution issues that prevent cars from improving across generations.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve all working menu, replay, HUD, screenshot, and simulation features unless explicitly changed below.
**MANDATORY**: Keep implementation modular, clean, and production quality.
**MANDATORY**: Use Python 3.11+ syntax.
**MANDATORY**: Execute the implementation now, not just describe a plan.

### FEATURE GOAL ###
1. Make manual driving slower and easier to control.
2. Fix training logic so NEAT cars actually improve over generations instead of behaving as if they are not learning.

### REQUIRED CHANGES ###

#### 1. Slow Down Manual Mode
**MANDATORY**: Reduce the effective driving speed and control sensitivity in Manual Mode only.

Required behavior:
- Manual Mode must feel slower, more controllable, and more playable than training mode.
- The player car should not accelerate too aggressively.
- Steering in Manual Mode should be easier to manage at low speed.
- Keep AI training speed behavior separate unless explicitly needed.

**CRITICAL**:
- Do not globally slow down the entire project unless necessary.
- Prefer a dedicated Manual Mode speed profile, such as:
  - lower default speed
  - lower acceleration
  - lower rotation step
  - optional cap on max speed
- Reuse the same car physics system if possible, but allow mode-specific tuning.

#### 2. Fix Learning / Evolution Problems
The current training appears to produce poor improvement across generations, as if cars are not learning from past failures.

**MANDATORY**: Review and fix the training and evaluation flow so NEAT can improve meaningfully over generations.

You must inspect and correct any issues in:
- genome fitness assignment
- per-generation reset behavior
- car creation per genome
- driver/genome association
- sensor input consistency
- action output mapping
- generation termination conditions
- mutation/evolution integration path
- accidental reuse of stale state between generations
- fitness values being overwritten, not accumulated correctly, or assigned to wrong genomes

**CRITICAL**:
- Ensure each genome gets its own car and its own driver/network.
- Ensure the correct genome fitness is updated from the corresponding car reward.
- Ensure cars are recreated/reset cleanly at the start of each generation.
- Ensure NEAT receives meaningful fitness signals.
- Ensure one bad control-flow bug is not causing all genomes to behave similarly.
- Ensure training mode still runs multiple cars per generation.

#### 3. Improve Fitness Logic
**MANDATORY**: Improve or correct the reward/fitness system so better driving behavior is actually favored.

Fitness should reward things like:
- staying alive
- moving forward productively
- distance traveled
- optionally checkpoint progress if available

Fitness should penalize or at least stop rewarding:
- crashing
- spinning in place
- getting stuck
- meaningless survival without useful movement
- exploiting loops that inflate score without learning to drive well

**CRITICAL**:
- Do not let one broken reward exploit dominate generations.
- If checkpoints already exist in `Track`, integrate them cleanly if helpful.
- Keep the reward understandable and maintainable.

#### 4. Add Training Debug Visibility
**MANDATORY**: When HUD/debug mode is visible during training, show enough information to verify evolution is working.

Display useful values such as:
- generation number
- total cars spawned
- currently alive cars
- best fitness this generation
- average fitness this generation
- current species count if available

**CRITICAL**: This is needed so the user can verify that training is actually progressing.

#### 5. Validate Manual vs Training Separation
**MANDATORY**: Ensure Manual Mode tuning does not break AI training behavior.

Required separation:
- Manual Mode may use slower speed/control values.
- Training Mode should still use its own simulation parameters.
- Replay Mode should continue working correctly.

### FILE TARGETS ###
Modify the existing architecture appropriately. Use the current project structure and adapt as needed.

Likely files to modify/create include:
`src/ai_car_sim/main.py`
`src/ai_car_sim/simulation/engine.py`
`src/ai_car_sim/core/car.py`
`src/ai_car_sim/domain/simulation_config.py`
`src/ai_car_sim/ai/neat_driver.py`
`src/ai_car_sim/ai/training_manager.py`
`src/ai_car_sim/ui/hud_view.py`

You may create helper modules or methods if useful, but keep the design clean.

### ENGINEERING RULES ###
**MANDATORY**:
- Keep Manual Mode logic distinct from training logic where appropriate.
- Reuse existing abstractions instead of duplicating systems.
- Use clear type hints.
- Add clean docstrings.
- Keep fitness logic easy to understand.
- Keep genome-to-car-to-driver mapping explicit and correct.
- Do not break replay mode.
- Do not break menu mode.
- Do not break screenshot/photo mode features.

### TESTING REQUIREMENTS ###
**CRITICAL**: Verify all of the following after implementation:
1. Manual Mode car is noticeably slower and easier to control
2. Manual Mode still remains responsive and playable
3. Training mode still spawns multiple cars per generation
4. Each genome is evaluated independently
5. Fitness values are assigned correctly to the corresponding genomes
6. Cars show at least some improvement over generations on the same track
7. HUD/debug info shows generation, alive count, and fitness stats
8. Replay mode still works
9. Menu flow still works

### OUTPUT REQUIREMENTS ###
**MANDATORY**: Modify repository files directly.
**MANDATORY**: Do not only acknowledge the request.
**MANDATORY**: Do not only explain what should be done.
**MANDATORY**: Do not output a plan.
**MANDATORY**: After implementing, provide a concise summary of:
- changed files
- manual mode speed/control changes
- training/evolution bugs fixed
- fitness logic changes
- new debug information shown on screen

### CRITICAL REQUIREMENTS ###
**MANDATORY**: Fully implement the Manual Mode speed improvements and the NEAT learning/evolution fixes now, in working code, while preserving the rest of the application behavior.