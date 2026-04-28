### PROMPT 29 — BEST PERFORMANCE TRACKER ###

### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Add a persistent best performance tracker that records the best AI results across generations and displays them in the HUD.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve all existing functionality.
**MANDATORY**: Keep implementation clean and modular.
**MANDATORY**: Execute now.

### REQUIRED FEATURES ###
Add tracking for:
- best fitness ever
- best generation ever
- longest survival time
- best current run on selected map

Display in HUD when visible.

Save results between runs if persistence system already exists.

Likely files:
- src/ai_car_sim/ui/hud_view.py
- src/ai_car_sim/analytics/run_metrics.py
- src/ai_car_sim/persistence/save_load.py
- src/ai_car_sim/simulation/engine.py

### OUTPUT REQUIREMENTS ###
Implement directly and summarize changed files.

### CRITICAL REQUIREMENTS ###
Fully implement now.