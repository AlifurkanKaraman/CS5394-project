### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Modify the existing project to add a professional **Photo Mode / Screenshot System** for academic presentation screenshots.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve all existing simulation behavior.
**MANDATORY**: Keep implementation modular, clean, and production quality.
**MANDATORY**: Use Python 3.11+ syntax.
**MANDATORY**: Execute the implementation now, not just a description or plan.

### FEATURE GOAL ###
Add a polished screenshot and presentation mode so the user can capture clean images of the simulation for reports, demos, SRS documentation, and presentations.

### REQUIRED FEATURES ###

#### 1. Screenshot Key
**MANDATORY**: Pressing `P` must save the current rendered frame as a PNG into `outputs/screenshots/`.

**CRITICAL**:
- Automatically create the folder if it does not exist.
- Use filename format: `screenshot_YYYYMMDD_HHMMSS.png`
- Example: `screenshot_20260413_194522.png`
- Use `pygame.image.save(...)`.

#### 2. Pause Mode
**MANDATORY**: Pressing `SPACE` must toggle pause/resume.

When paused:
- cars stop updating
- training frame updates stop
- timers stop progressing
- rendering remains active

**CRITICAL**: Show the text `PAUSED` near the top-center of the screen while paused.

#### 3. Hide HUD Toggle
**MANDATORY**: Pressing `H` must toggle HUD visibility on/off.

When HUD is hidden, remove:
- generation number
- alive count
- metrics
- debug overlays
- non-essential labels

**CRITICAL**: This is required for clean screenshots.

#### 4. Screenshot Confirmation
**MANDATORY**: After pressing `P`, show a temporary on-screen confirmation for approximately 2 seconds: `Screenshot Saved`

#### 5. Cinematic Mode
**MANDATORY**: Pressing `C` must toggle cinematic mode.

When cinematic mode is enabled:
- automatically hide HUD
- keep only the clean simulation view
- optionally draw black cinematic bars at the top and bottom

**CRITICAL**: Cinematic mode must be useful for academic screenshots and demo captures.

#### 6. Controls Overlay
**MANDATORY**: Display a small controls guide on screen showing:
`P = Screenshot`
`SPACE = Pause`
`H = Hide HUD`
`C = Cinematic`
`ESC = Quit`

This overlay should remain visible unless hidden by HUD-off or cinematic mode.

### FILE TARGETS ###
Modify the existing architecture appropriately. Use the current project structure and adapt as needed.

Likely files to modify/create include:
`src/ai_car_sim/main.py`
`src/ai_car_sim/simulation/engine.py`
`src/ai_car_sim/ui/hud_view.py`
`src/ai_car_sim/ui/photo_mode.py`

**CRITICAL**: If the current architecture differs, integrate the feature cleanly into the correct existing modules.

### ENGINEERING RULES ###
**MANDATORY**:
- Do not break the NEAT training loop.
- If paused, generation timing and fitness progression must also pause correctly.
- Keep rendering responsive while paused.
- Use `pathlib` for filesystem paths.
- Use clear type hints.
- Add clean docstrings.
- Separate input handling, rendering, and screenshot logic.
- Keep the codebase maintainable and modular.

### TESTING REQUIREMENTS ###
**CRITICAL**: Verify all of the following after implementation:
1. Pressing `P` saves a PNG successfully in `outputs/screenshots/`
2. Pressing `SPACE` pauses and resumes correctly
3. Pressing `H` hides and restores the HUD
4. Pressing `C` toggles cinematic mode correctly
5. The “Screenshot Saved” message appears temporarily
6. The simulation still runs normally after using all toggles

### OUTPUT REQUIREMENTS ###
**MANDATORY**: Modify repository files directly.
**MANDATORY**: Do not only acknowledge the request.
**MANDATORY**: Do not only explain what should be done.
**MANDATORY**: Do not output a plan.
**MANDATORY**: After implementing, provide a concise summary of:
- changed files
- key additions
- how to use the new controls

### CRITICAL REQUIREMENTS ###
**MANDATORY**: Fully implement the Photo Mode / Screenshot System now, in working code, while preserving the existing simulation and training behavior.