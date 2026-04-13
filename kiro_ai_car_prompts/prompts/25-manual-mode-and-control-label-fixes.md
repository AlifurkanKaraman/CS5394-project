### CRITICAL REQUIREMENTS ###
You are an expert Python game engineer working on a Pygame + NEAT self-driving car simulation.

**CRITICAL**: Implement the missing Manual Mode and fix broken control label rendering in the UI.
**MANDATORY**: Apply changes directly to repository files.
**MANDATORY**: Preserve all working simulation, training, replay, menu, and screenshot features.
**MANDATORY**: Keep implementation modular, clean, and production quality.
**MANDATORY**: Use Python 3.11+ syntax.
**MANDATORY**: Execute the implementation now, not just describe it.

### BUGS / MISSING FEATURES TO FIX ###

#### 1. Manual Mode is listed but not implemented
The application currently logs: `Manual mode not yet implemented.`

**MANDATORY**: Fully implement Manual Mode so the user can drive a car directly in the simulation.

Required behavior:
- Selecting Manual Mode from the menu must start a playable simulation.
- Spawn one player-controlled car on the selected track.
- The car must use the same underlying physics and collision system as AI cars.
- The simulation should run in a dedicated manual gameplay loop or an adapted engine path.
- Crashing should end the current manual run or reset appropriately.
- The user must be able to return to the main menu with `Q`.

#### 2. Manual Controls
**MANDATORY**: Support keyboard controls for the player car.

Required controls:
- `LEFT` = steer left
- `RIGHT` = steer right
- `UP` = accelerate / forward
- `DOWN` = brake / slow down

If the current action model is discrete, map these inputs cleanly to the existing action system.
If needed, create a `KeyboardDriver` that conforms to the same driver abstraction used by AI drivers.

#### 3. Broken Arrow Symbols in UI
The current UI shows blank symbols instead of left/right arrows.

**MANDATORY**: Fix this by replacing unreliable unicode arrow glyphs with labels that always render correctly.

Use safe labels such as:
- `LEFT = Steer Left`
- `RIGHT = Steer Right`
- `UP = Accelerate`
- `DOWN = Brake`

**CRITICAL**:
- Do not rely on unicode arrows unless the selected font fully supports them.
- Prefer robust plain-text labels that render correctly in pygame fonts.

#### 4. Update Menu / HUD / Help Text
**MANDATORY**: Update all on-screen controls/help text so manual mode displays the correct controls.

Examples:
- `LEFT = Steer Left`
- `RIGHT = Steer Right`
- `UP = Accelerate`
- `DOWN = Brake`
- `Q = Main Menu`

If training mode and replay mode have separate controls, ensure each mode shows the correct help text.

#### 5. Manual Mode HUD
**MANDATORY**: In manual mode, show useful HUD information such as:
- current mode = Manual
- speed
- alive/crashed state
- selected track name

This should integrate with the existing HUD if possible.

### FILE TARGETS ###
Modify the existing architecture appropriately. Use the current project structure and adapt as needed.

Likely files to modify/create include:
`src/ai_car_sim/main.py`
`src/ai_car_sim/simulation/engine.py`
`src/ai_car_sim/ui/hud_view.py`
`src/ai_car_sim/ui/menu_controller.py`
`src/ai_car_sim/ai/driver_interface.py`

You may also create:
`src/ai_car_sim/ai/keyboard_driver.py`

**CRITICAL**: If the current architecture differs, integrate the feature cleanly into the correct existing modules.

### ENGINEERING RULES ###
**MANDATORY**:
- Keep Manual Mode separate from NEAT training mode logic where appropriate.
- Reuse the existing Car, VehicleState, collision, and sensor systems as much as possible.
- Do not break training mode.
- Do not break replay mode.
- Do not break menu navigation.
- Use clear type hints.
- Add clean docstrings.
- Keep control handling centralized and maintainable.

### TESTING REQUIREMENTS ###
**CRITICAL**: Verify all of the following after implementation:
1. Selecting Manual Mode no longer logs “Manual mode not yet implemented.”
2. Manual Mode launches a playable simulation
3. LEFT and RIGHT steer the car
4. UP accelerates
5. DOWN brakes or slows the car
6. Q returns to main menu from manual mode
7. Control labels render correctly with no blank symbols
8. Training mode still works
9. Replay mode still works

### OUTPUT REQUIREMENTS ###
**MANDATORY**: Modify repository files directly.
**MANDATORY**: Do not only acknowledge the request.
**MANDATORY**: Do not only explain what should be done.
**MANDATORY**: Do not output a plan.
**MANDATORY**: After implementing, provide a concise summary of:
- changed files
- manual mode behavior
- updated controls
- UI text fixes

### CRITICAL REQUIREMENTS ###
**MANDATORY**: Fully implement Manual Mode and fix the broken control label rendering now, in working code, while preserving the rest of the application behavior.