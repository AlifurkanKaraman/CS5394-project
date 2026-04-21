### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Modify the existing project to add improved simulation controls, generation feedback, and better car spawning behavior.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve all existing simulation and training behavior unless explicitly changed below.
**MANDATORY**: Keep implementation modular, clean, and production quality.
**MANDATORY**: Use Python 3.11+ syntax.
**MANDATORY**: Execute the implementation now, not just a description or plan.

### FEATURE GOAL ###
Improve usability and presentation of the simulation by updating quit/reset controls, adding simulation speed control, spreading starting cars so they do not overlap visually, and showing a clear generation transition popup/overlay.

### REQUIRED FEATURES ###

#### 1. Update Keybindings
**MANDATORY**: Change control behavior as follows:
- Pressing `Q` must quit the active simulation and return to the main menu.
- Pressing `ESC` must force-end the current generation and immediately begin the next generation.

**CRITICAL**:
- Remove any old conflicting behavior for these keys.
- Make key handling consistent across simulation mode.
- If the project currently uses `ESC` to quit, move that quit behavior to `Q`.

#### 2. Simulation Speed Control
**MANDATORY**: Add controls to change the simulation speed during runtime.

Required controls:
- Press `=` or `+` to increase simulation speed
- Press `-` to decrease simulation speed
- Press `0` to reset simulation speed to normal

**CRITICAL**:
- Speed control must affect training/update speed, not just rendering text.
- Use a configurable simulation speed multiplier such as `0.5x`, `1.0x`, `2.0x`, `4.0x`, `8.0x`.
- Display the current speed on screen when HUD is visible.
- Speed changes must remain stable and must not break NEAT generation logic.
- If some maps are difficult and cars crash fast, higher speed should let the user move through generations faster.

#### 3. Spread Cars at Spawn
**MANDATORY**: Adjust spawning so cars do not appear directly stacked inside each other.

**CRITICAL**:
- Spawn cars using small offsets around the track spawn position.
- Preserve fairness and consistency as much as possible.
- Spread should be visually noticeable but not large enough to cause immediate collisions or unfair advantage.
- Maintain the same intended starting direction/angle.
- Use a deterministic and clean spawning pattern if possible, such as slight lane-style offsets or a small fan/grid pattern.

#### 4. Generation Transition Popup / Overlay
**MANDATORY**: After each generation transition, show a clear temporary popup/overlay indicating the current generation.

Examples:
- `New Generation`
- `Generation 1`
- `Generation 2`
- `Generation 3`

**CRITICAL**:
- Show this overlay for a short duration such as 1 to 2 seconds.
- It must be clearly visible.
- It must appear whenever a new generation starts, including after pressing `ESC`.
- It should not block or break the simulation loop.
- It may be rendered as centered text, a banner, or a small popup overlay.

#### 5. HUD / Controls Update
**MANDATORY**: Update on-screen controls/help text to reflect the new bindings.

The controls overlay must now show:
- `Q = Main Menu`
- `ESC = Next Generation`
- `+ / - = Speed Up / Slow Down`
- `0 = Reset Speed`

If photo mode controls already exist, keep them and integrate the new controls cleanly.

### FILE TARGETS ###
Modify the existing architecture appropriately. Use the current project structure and adapt as needed.

Likely files to modify/create include:
`src/ai_car_sim/main.py`
`src/ai_car_sim/simulation/engine.py`
`src/ai_car_sim/ui/hud_view.py`
`src/ai_car_sim/ui/menu_controller.py`

You may also create a helper module if useful, such as:
`src/ai_car_sim/ui/generation_overlay.py`

**CRITICAL**: If the current architecture differs, integrate the feature cleanly into the correct existing modules.

### ENGINEERING RULES ###
**MANDATORY**:
- Do not break the NEAT training loop.
- Do not break existing pause/photo/cinematic functionality if already implemented.
- Keep key handling centralized and maintainable.
- Keep simulation speed logic separate from rendering logic.
- Use clear type hints.
- Add clean docstrings.
- Separate spawn logic, generation overlay logic, and speed-control logic into focused helpers or methods where appropriate.
- Keep the codebase maintainable and modular.

### TESTING REQUIREMENTS ###
**CRITICAL**: Verify all of the following after implementation:
1. Pressing `Q` returns to main menu
2. Pressing `ESC` skips to the next generation
3. Pressing `+` or `=` increases simulation speed
4. Pressing `-` decreases simulation speed
5. Pressing `0` resets speed to normal
6. Current speed is visible on screen when HUD is shown
7. Cars spawn visually separated instead of directly overlapping
8. A temporary generation popup/overlay appears whenever a new generation starts
9. All existing simulation behavior still works correctly

### OUTPUT REQUIREMENTS ###
**MANDATORY**: Modify repository files directly.
**MANDATORY**: Do not only acknowledge the request.
**MANDATORY**: Do not only explain what should be done.
**MANDATORY**: Do not output a plan.
**MANDATORY**: After implementing, provide a concise summary of:
- changed files
- key additions
- new controls

### CRITICAL REQUIREMENTS ###
**MANDATORY**: Fully implement these usability and generation-control improvements now, in working code, while preserving the existing simulation and training behavior except for the explicit control changes requested above.