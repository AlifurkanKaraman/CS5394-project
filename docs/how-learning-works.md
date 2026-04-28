# How Learning Works — AI Car Simulation

This document explains how the NEAT neuroevolution algorithm is integrated
into this project, how cars learn to drive over generations, and where every
tunable parameter lives in the codebase.

Intended audience: students, professors, and reviewers who want to understand
the implementation without reading every source file.

---

## Table of Contents

1. [Algorithm Overview](#1-algorithm-overview)
2. [Code Map of the Learning Pipeline](#2-code-map-of-the-learning-pipeline)
3. [Step-by-Step Learning Loop](#3-step-by-step-learning-loop)
4. [How Sensor Inputs Work](#4-how-sensor-inputs-work)
5. [How Neural Network Outputs Become Actions](#5-how-neural-network-outputs-become-actions)
6. [How Fitness Is Calculated](#6-how-fitness-is-calculated)
7. [Stuck Detection](#7-stuck-detection)
8. [Checkpoint Rewards](#8-checkpoint-rewards)
9. [Training vs Replay vs Manual Mode](#9-training-vs-replay-vs-manual-mode)
10. [Where Speed and Control Values Are Configured](#10-where-speed-and-control-values-are-configured)
11. [How to Tune the Project](#11-how-to-tune-the-project)
12. [How to Explain This to a Professor](#12-how-to-explain-this-to-a-professor)

---

## 1. Algorithm Overview

This project uses **NEAT** (NeuroEvolution of Augmenting Topologies), implemented
by the [`neat-python`](https://neat-python.readthedocs.io/) library.

NEAT is an evolutionary algorithm that simultaneously evolves both the **weights**
and the **topology** (structure) of neural networks. It starts with minimal networks
and adds nodes and connections over generations as needed — so the network grows in
complexity only when complexity helps.

Each car in the simulation is controlled by one neural network (genome). Cars that
drive further before crashing receive higher fitness scores. NEAT uses those scores
to select which genomes reproduce, mutate, and form the next generation's population.

---

## 2. Code Map of the Learning Pipeline

| File | Responsibility |
|---|---|
| `configs/config.txt` | NEAT hyperparameters: population size, mutation rates, speciation thresholds |
| `src/ai_car_sim/domain/simulation_config.py` | All Python-side tunable values: car speed, radar, FPS, generation budget |
| `src/ai_car_sim/ai/training_manager.py` | Owns the NEAT `Population` object; calls `pop.run()` which drives the entire evolutionary loop |
| `src/ai_car_sim/ai/neat_driver.py` | Wraps one NEAT `FeedForwardNetwork`; converts sensor inputs → network outputs → `Action` |
| `src/ai_car_sim/ai/driver_interface.py` | Defines the `DriverInterface` ABC and the `Action` enum (TURN_LEFT, TURN_RIGHT, SLOW_DOWN, SPEED_UP) |
| `src/ai_car_sim/simulation/engine.py` | Creates one `Car` + one `NeatDriver` per genome; runs the per-generation frame loop; assigns fitness back to genomes |
| `src/ai_car_sim/core/car.py` | Physics, collision, stuck detection, checkpoint detection, and `get_reward()` |
| `src/ai_car_sim/core/radar_sensor.py` | Casts 5 radar rays from the car; returns normalised distances [0, 1] |
| `src/ai_car_sim/domain/track.py` | Track data: spawn position, border colour, optional checkpoint line segments |
| `src/ai_car_sim/domain/vehicle_state.py` | Mutable per-car state: position, angle, speed, distance, checkpoint index |
| `src/ai_car_sim/ai/replay_loader.py` | Loads a saved genome pickle and reconstructs a `NeatDriver` for replay |
| `src/ai_car_sim/ai/keyboard_driver.py` | Human keyboard input mapped to the same `Action` enum — used in manual mode |
| `src/ai_car_sim/main.py` | CLI entry point; wires config → tracks → engine → training manager |

---

## 3. Step-by-Step Learning Loop

Below is the complete learning cycle as it actually executes in this codebase.

### Step 1 — Load configuration

`main.py → build_default_config()` loads `SimulationConfig` (Python physics values)
and `TrainingManager.load_neat_config()` loads `configs/config.txt` (NEAT parameters).

### Step 2 — Create the population

`TrainingManager.create_population()` calls `neat.Population(neat_config)`.
NEAT creates `pop_size` (default: 50) genomes, each representing a minimal
feed-forward network with 5 inputs and 4 outputs and no hidden nodes.

### Step 3 — Start a generation (NEAT calls `_eval_genomes`)

`neat.Population.run()` calls `TrainingManager._eval_genomes(genomes, neat_config)`
once per generation. This is the bridge between NEAT and the simulation.

```
TrainingManager._eval_genomes()          # neat_driver.py calls this each generation
  → resets genome.fitness = 0.0 for all genomes
  → calls engine.evaluate_genomes(genomes, neat_config)
```

### Step 4 — Spawn one car per genome

`SimulationEngine.create_cars_for_genomes()` iterates over every `(genome_id, genome)`
pair and creates:
- one `neat.nn.FeedForwardNetwork` from the genome
- one `NeatDriver` wrapping that network
- one `Car` instance with a fresh `VehicleState`

All cars are placed at the track's spawn position with a small perpendicular
spread (2 px apart) so they don't visually stack.

```python
# engine.py — one network, one driver, one car per genome
network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
driver  = NeatDriver(network, expected_outputs=len(Action))
car     = Car(config, sprite_surface=sprite, track=track)
car.reset(spawn_x, spawn_y, spawn_angle)
```

### Step 5 — Run the generation frame loop

`SimulationEngine.run_generation()` runs until all cars have crashed, the
step budget (`steps_per_generation`, default 1200 ticks = 20 seconds) is
exhausted, or the user presses ESC/Q.

Each tick:

```
engine._step_all()
  for each (car, driver) pair:
    inputs  = car.get_sensor_inputs(game_map)   # 5 radar distances [0,1]
    action  = driver.decide_action(inputs, car.state)  # network forward pass
    car.apply_action(action)                    # update speed / angle
    car.update(game_map)                        # move, collide, stuck-check
```

### Step 6 — Collect sensor inputs

`Car.get_sensor_inputs()` calls `RadarSensorSystem.normalized_inputs()`, which
casts 5 rays at angles `[-90, -45, 0, 45, 90]` degrees relative to the car's
heading. Each ray marches outward until it hits a white border pixel or reaches
`max_radar_distance` (300 px). The distance is normalised to `[0.0, 1.0]`.

### Step 7 — Run the neural network

`NeatDriver.decide_action()` calls `network.activate(sensor_inputs)`, which
runs a forward pass through the NEAT network. The output is a list of 4
activation values. The index of the highest value selects the action:

```
output index 0 → Action.TURN_LEFT
output index 1 → Action.TURN_RIGHT
output index 2 → Action.SLOW_DOWN
output index 3 → Action.SPEED_UP
```

### Step 8 — Apply the action to the car

`Car.apply_action()` updates the car's speed or heading angle:

```
TURN_LEFT   → angle += 10°
TURN_RIGHT  → angle -= 10°
SPEED_UP    → speed += 2 px/tick  (clamped to max_speed = 40)
SLOW_DOWN   → speed -= 2 px/tick  (clamped to min_speed = 12)
```

`Car.update()` then moves the car by `speed` pixels in the heading direction,
checks for border collision, and runs stuck detection.

### Step 9 — Compute fitness

After `run_generation()` returns, `evaluate_genomes()` reads each car's reward:

```python
# engine.py — index-aligned: genome[i] gets car[i]'s reward
for genome, car in zip(self._genomes, self._cars):
    genome.fitness = car.get_reward()
```

`Car.get_reward()` returns:

```
fitness = (distance_travelled / 30) + checkpoint_bonus
```

where `30` is half the car's 60 px width (normalisation constant), and
`checkpoint_bonus` is `200 × checkpoints_reached` if the track has checkpoints.

### Step 10 — Evolve the next generation

After `_eval_genomes` returns, NEAT's internal reproduction logic runs
automatically. It:
- groups genomes into species by structural similarity
- selects the top `survival_threshold` (20%) of each species to reproduce
- applies mutation (weight changes, node/connection add/remove)
- preserves the top `elitism` (2) genomes unchanged per species
- removes species that haven't improved in `max_stagnation` (20) generations

The cycle then repeats from Step 3 with the new population.

---

## 4. How Sensor Inputs Work

**File:** `src/ai_car_sim/core/radar_sensor.py` — `RadarSensorSystem.scan()`

The car has 5 radar rays cast at fixed angles relative to its heading:

```
-90°  (hard left)
-45°  (soft left)
  0°  (straight ahead)
+45°  (soft right)
+90°  (hard right)
```

Each ray marches pixel by pixel from the car's centre outward. When it hits
a white border pixel `(255, 255, 255, 255)` or reaches 300 px, it stops.
The distance is divided by `max_radar_distance` to produce a value in `[0, 1]`.

A value near `0.0` means a wall is very close. A value near `1.0` means the
path is clear for at least 300 px.

These 5 values are the **only inputs** the neural network receives. The network
has no knowledge of its position, speed, or heading — it must infer everything
from the radar distances alone.

---

## 5. How Neural Network Outputs Become Actions

**File:** `src/ai_car_sim/ai/neat_driver.py` — `NeatDriver.decide_action()`

The NEAT network produces 4 output activations (one per action). The action
with the highest activation wins (argmax). This is a discrete classification
approach — the car always takes exactly one action per tick.

```python
outputs = network.activate(sensor_inputs)   # [left_score, right_score, slow_score, fast_score]
action  = Action(outputs.index(max(outputs)))  # pick the winner
```

The `Action` enum is defined in `src/ai_car_sim/ai/driver_interface.py`.

---

## 6. How Fitness Is Calculated

**File:** `src/ai_car_sim/core/car.py` — `Car.get_reward()`

```python
fitness = (distance_travelled / car_half_size) + checkpoint_bonus
```

- `distance_travelled`: total pixels driven before crashing or being stuck-killed
- `car_half_size`: 30 px (half of the 60 px car width) — normalisation constant
- `checkpoint_bonus`: `200 × checkpoints_reached` (only if the track defines checkpoints)

**Why this formula?**

Distance is the primary signal. A car that drives further before crashing is
strictly better than one that crashes sooner. The normalisation by `car_half_size`
keeps fitness values in a human-readable range (roughly 0–800 for a 20-second run).

Checkpoint bonuses (200 pts each) strongly dominate the distance score, so if
checkpoints are defined, reaching them is the dominant objective.

**What is intentionally excluded:**

- Survival time bonus (`time_steps × k`): this was previously present but caused
  a critical exploit — a car that spins in place for the full 1200 ticks scored
  higher than a car that drove 600 px and crashed. It has been removed.
- Efficiency ratio: distance already captures this; adding a ratio term creates
  non-linear interactions that confuse NEAT in early generations.

---

## 7. Stuck Detection

**File:** `src/ai_car_sim/core/car.py` — `Car._check_stuck()`

Every 60 ticks, the car checks whether it has moved at least 30 px since the
last snapshot. If not, it is marked as crashed and removed from the generation.

```python
_STUCK_CHECK_INTERVAL = 60    # ticks between checks
_STUCK_MIN_DISTANCE   = 30.0  # minimum px of progress required
```

This prevents cars from surviving the full generation budget by oscillating,
spinning, or hugging a wall — behaviours that would otherwise accumulate
distance_travelled without actually learning to drive.

---

## 8. Checkpoint Rewards

**File:** `src/ai_car_sim/core/car.py` — `Car._check_checkpoints()`

If `Track.checkpoints` is defined (a list of line segments across the track),
the car earns `+200` fitness each time it crosses the next checkpoint in order.

Checkpoints must be reached sequentially — the car cannot farm the same
checkpoint twice. `VehicleState.checkpoint_index` tracks the next checkpoint
to reach, and `VehicleState.lap_progress` records the fraction of checkpoints
completed.

The current map assets (`assets/maps/map*.png`) do not have checkpoints defined
by default. To add them, set `Track.checkpoints` in `main.py → build_default_tracks()`.

---

## 9. Training vs Replay vs Manual Mode

All three modes use the same `Car` physics and the same `DriverInterface` contract.
The only difference is which driver is plugged in.

| Mode | Driver | Cars | Fitness assigned? |
|---|---|---|---|
| Training | `NeatDriver` (one per genome) | 50 per generation | Yes — written back to `genome.fitness` |
| Replay | `NeatDriver` (loaded from pickle) | 1 | No |
| Manual | `KeyboardDriver` (arrow keys) | 1 | No |

**Training** is orchestrated by `TrainingManager` → `SimulationEngine.evaluate_genomes()`.

**Replay** loads a saved genome via `src/ai_car_sim/ai/replay_loader.py →
load_replay_driver()`, reconstructs the network, and runs `engine.run_replay()`.

**Manual** creates a `KeyboardDriver` in `engine.run_manual()` and uses the
slower speed profile (`manual_default_speed = 8.0` vs AI `default_speed = 20.0`).

---

## 10. Where Speed and Control Values Are Configured

### Python physics values — `src/ai_car_sim/domain/simulation_config.py`

All values are fields on the `SimulationConfig` dataclass with documented defaults.

| Parameter | Field | Default | Notes |
|---|---|---|---|
| AI default speed | `default_speed` | `20.0` px/tick | Starting speed for training cars |
| AI max speed | `max_speed` | `40.0` px/tick | Hard cap for training cars |
| AI min speed | `min_speed` | `12.0` px/tick | Floor for training cars |
| Manual default speed | `manual_default_speed` | `8.0` px/tick | Slower for human playability |
| Manual max speed | `manual_max_speed` | `16.0` px/tick | |
| Manual min speed | `manual_min_speed` | `4.0` px/tick | |
| Manual turn step | `manual_turn_step` | `5.0°/tick` | AI uses hardcoded `10.0°/tick` |
| Radar max distance | `max_radar_distance` | `300` px | Length of each radar ray |
| Radar angles | `radar_angles` | `[-90,-45,0,45,90]` | Degrees relative to heading |
| FPS | `fps` | `60` | Target frames per second |
| Generation budget | `steps_per_generation` | `1200` ticks | 60 fps × 20 s |
| Max generations | `max_generations` | `1000` | NEAT stops after this many |
| NEAT config path | `neat_config_path` | `"configs/config.txt"` | |
| Output directory | `output_dir` | `"outputs"` | Genomes, metrics, screenshots |

### Speed delta and turn step — `src/ai_car_sim/core/car.py`

Inside `Car.__init__()`, the mode-specific profile is selected:

```python
self._turn_step   = config.manual_turn_step if manual_mode else 10.0
self._speed_delta = 1.0 if manual_mode else 2.0   # px/tick per SPEED_UP/SLOW_DOWN
```

The AI turn step (`10.0°`) and speed delta (`2.0 px/tick`) are hardcoded constants
in `Car.__init__()` — they are not currently exposed in `SimulationConfig`.

### Simulation speed multiplier — `src/ai_car_sim/simulation/engine.py`

```python
_SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # runtime speed multiplier steps
```

Controlled at runtime with `+` / `-` keys. This multiplies the number of physics
ticks per rendered frame — it does not change the car's in-simulation speed.

### NEAT hyperparameters — `configs/config.txt`

| Parameter | Value | Effect |
|---|---|---|
| `pop_size` | 50 | Genomes evaluated per generation |
| `num_inputs` | 5 | Must match `len(radar_angles)` in SimulationConfig |
| `num_outputs` | 4 | Must match `len(Action)` in driver_interface.py |
| `weight_mutate_rate` | 0.8 | 80% chance a connection weight mutates each generation |
| `conn_add_prob` | 0.5 | 50% chance a new connection is added |
| `node_add_prob` | 0.2 | 20% chance a new hidden node is added |
| `compatibility_threshold` | 3.0 | Genomes more different than this form separate species |
| `max_stagnation` | 20 | Generations a species can go without improvement before removal |
| `survival_threshold` | 0.2 | Top 20% of each species survive to reproduce |
| `elitism` | 2 | Top 2 genomes per species are copied unchanged |

---

## 11. How to Tune the Project

### Make cars faster or slower (AI training)

Edit `src/ai_car_sim/domain/simulation_config.py`:

```python
default_speed: float = 20.0   # increase for faster cars
max_speed:     float = 40.0   # hard cap
min_speed:     float = 12.0   # floor (cars never stop completely)
```

### Make steering more or less sensitive (AI)

Edit `Car.__init__()` in `src/ai_car_sim/core/car.py`:

```python
self._turn_step = 10.0   # degrees per tick — increase for sharper turns
```

### Change radar range or angles

Edit `src/ai_car_sim/domain/simulation_config.py`:

```python
max_radar_distance: int = 300          # pixels — increase for longer sight
radar_angles: list[int] = [-90, -45, 0, 45, 90]  # add more angles for richer input
```

> **Important:** if you change `len(radar_angles)`, you must also update
> `num_inputs` in `configs/config.txt` to match.

### Change generation length

```python
steps_per_generation: int = 1200   # 60 fps × 20 s — increase for longer runs
```

### Change population size

Edit `configs/config.txt`:

```ini
pop_size = 50   # increase for more diversity, decrease for faster generations
```

### Change how quickly stuck cars are killed

Edit `src/ai_car_sim/core/car.py`:

```python
_STUCK_CHECK_INTERVAL = 60    # ticks between progress checks (lower = faster kill)
_STUCK_MIN_DISTANCE   = 30.0  # minimum px of progress required per interval
```

### Add checkpoints to a track

Edit `main.py → build_default_tracks()`:

```python
Track(
    name="Map 1",
    map_image_path="assets/maps/map.png",
    spawn_position=(830.0, 920.0),
    spawn_angle=0.0,
    border_color=config.border_color,
    checkpoints=[
        ((500, 800), (500, 1000)),   # line segment across the track
        ((900, 600), (1100, 600)),
    ],
)
```

Each checkpoint is a `((x1, y1), (x2, y2))` line segment drawn across the
drivable lane. Cars earn `+200` fitness per checkpoint reached in order.

### Change the fitness formula

Edit `Car.get_reward()` in `src/ai_car_sim/core/car.py`. The current formula is:

```python
distance_score = self.state.distance_travelled / self.config.car_half_size()
return max(0.0, distance_score + self._checkpoint_bonus)
```

### Tune NEAT evolution speed

Edit `configs/config.txt`. Key levers:

- Increase `weight_mutate_rate` (currently 0.8) to explore weight space faster
- Decrease `compatibility_threshold` (currently 3.0) to create more species
- Decrease `max_stagnation` (currently 20) to kill underperforming species sooner
- Increase `survival_threshold` (currently 0.2) to keep more genomes per generation

---

## 12. How to Explain This to a Professor

### What algorithm is used?

**NEAT — NeuroEvolution of Augmenting Topologies** (Stanley & Miikkulainen, 2002).
NEAT is an evolutionary algorithm that evolves both the weights and the topology
of neural networks simultaneously. It uses speciation to protect structural
innovation and historical markings to solve the competing conventions problem.

### Why NEAT?

NEAT is well-suited to this problem because:
- The optimal network topology is unknown in advance
- The task (driving around a track) has a clear scalar fitness signal
- NEAT's speciation mechanism prevents premature convergence
- The `neat-python` library provides a clean, well-tested implementation

### How does the car learn over generations?

Each generation, 50 cars are spawned simultaneously on the same track. Each car
is controlled by a different neural network (genome). Cars that drive further
before crashing receive higher fitness scores. NEAT selects the fittest genomes,
applies mutation (weight changes, adding/removing nodes and connections), and
produces the next generation's population. Over many generations, networks that
produce useful steering behaviour are selected and refined.

### What inputs and outputs does the car use?

**Inputs (5 values):** normalised distances from 5 radar rays cast at
-90°, -45°, 0°, +45°, +90° relative to the car's heading. Each value is in
[0, 1] where 0 means a wall is immediately adjacent and 1 means the path is
clear for 300 px.

**Outputs (4 values):** one activation per action. The action with the highest
activation is selected each tick: TURN_LEFT, TURN_RIGHT, SLOW_DOWN, SPEED_UP.

### How is better behaviour selected?

Fitness = `distance_travelled / 30` + `200 × checkpoints_reached`.

Cars that drive further and reach more checkpoints score higher. Cars that crash
immediately score near zero. Cars that get stuck (< 30 px progress in 60 ticks)
are killed early. NEAT's selection pressure ensures that genomes producing
higher fitness are more likely to reproduce.

### Where is the implementation?

The core learning pipeline spans four files:

1. `configs/config.txt` — NEAT hyperparameters
2. `src/ai_car_sim/ai/training_manager.py` — population lifecycle
3. `src/ai_car_sim/simulation/engine.py` — per-generation evaluation
4. `src/ai_car_sim/core/car.py` — physics, fitness, stuck detection
