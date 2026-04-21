### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Fix the NEAT training pipeline so cars show meaningful improvement across generations.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve existing menu, replay, manual mode, HUD, and screenshot systems unless explicitly adjusted below.
**MANDATORY**: Keep implementation modular, clean, and production quality.
**MANDATORY**: Use Python 3.11+ syntax.
**MANDATORY**: Execute the implementation now, not just explain it.

### PROBLEMS TO FIX ###
The current NEAT behavior suggests cars are not improving meaningfully. Review and fix the full evaluation pipeline.

### REQUIRED CHANGES ###

#### 1. Verify Correct Genome-to-Car Fitness Mapping
**MANDATORY**:
- Ensure each genome gets exactly one car and one driver/network.
- Ensure each genome's fitness is updated only from its corresponding car.
- Ensure no stale car state, driver state, or genome state leaks across generations.

#### 2. Improve Fitness Function
**MANDATORY**: Replace weak or exploitable reward logic with a stronger driving-oriented fitness design.

Fitness should reward:
- forward progress
- distance traveled
- checkpoint progress if checkpoints are available
- staying alive only in combination with useful motion

Fitness should penalize or stop rewarding:
- crashing
- spinning in place
- standing still
- oscillating without progress
- reward exploits that inflate score without real driving success

#### 3. Add Stuck Detection
**MANDATORY**:
- Detect when a car makes no meaningful forward progress for too many frames.
- Mark that car inactive or end its reward accumulation.
- Prevent genomes from surviving uselessly without learning to drive.

#### 4. Use Checkpoints If Available
**MANDATORY**:
- If `Track.checkpoints` exists, integrate checkpoint-based reward.
- Give a clear fitness bonus when a car reaches a new checkpoint.
- Prevent repeated farming of the same checkpoint.

#### 5. Improve Debug Visibility
**MANDATORY**: Show training debug info on screen or logs:
- generation number
- total spawned cars
- alive cars
- best fitness
- average fitness
- best distance/checkpoint progress
- species count if available

#### 6. Keep Training Stable
**MANDATORY**:
- Recreate all cars cleanly at the start of each generation.
- Reset per-car reward and progress state correctly.
- Keep replay and manual mode behavior intact.

### FILE TARGETS ###
Likely files include:
`src/ai_car_sim/ai/training_manager.py`
`src/ai_car_sim/simulation/engine.py`
`src/ai_car_sim/core/car.py`
`src/ai_car_sim/domain/track.py`
`src/ai_car_sim/ui/hud_view.py`

Adapt to the current architecture as needed.

### TESTING REQUIREMENTS ###
Verify:
1. Each genome is evaluated independently
2. Fitness values differ meaningfully across cars
3. Cars that crash early get lower fitness
4. Cars that move usefully get higher fitness
5. Cars that get stuck are terminated
6. Checkpoints improve learning if available
7. Best and average fitness trend upward across generations on an easy map

### OUTPUT REQUIREMENTS ###
**MANDATORY**: Modify files directly and then summarize:
- changed files
- fitness logic changes
- stuck detection behavior
- checkpoint reward behavior
- debug metrics added

### CRITICAL REQUIREMENTS ###
**MANDATORY**: Fully fix the NEAT learning pipeline now so evolution has a meaningful signal and cars can improve over generations.