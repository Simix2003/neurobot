# NeuroBot

A brain-inspired AI / robotics simulation built entirely in Python.

## Overview

NeuroBot is a 2D simulation where a small robot learns to find food inside a simple world.

The robot has:

* sensors
* a small neural "brain"
* movement outputs
* a goal: survive by finding food

This project is designed as an approachable entry point into:

* artificial life
* brain-inspired AI
* robotics control loops
* simulation design
* neural network experimentation

The first goal is **not** to build a perfect scientific model.
The first goal is to build a clean MVP that works, is visual, and can later evolve into more advanced experiments.

---

# Core Concept

The simulation loop is:

```text
environment -> sensors -> brain -> actions -> environment update
```

At every frame:

1. the robot reads the world through simple sensors
2. the brain processes those inputs
3. the robot moves
4. the world updates
5. if the robot reaches food, it scores and food respawns

---

# MVP Goal

Build a **fully working Python simulation** where:

* a 2D world is rendered on screen
* a robot exists inside it
* food spawns in the world
* the robot can move and detect food
* the robot is driven by code, not by keyboard
* the project structure is clean enough to support future learning methods

For the MVP, the robot does **not** need to learn yet.
The main goal is to create the full simulation pipeline so learning can be added later.

---

# Tech Stack

## Language

* Python

## Main libraries

* `pygame` for the 2D window, rendering, loop, and input/debug support
* `numpy` for math and vector operations
* `torch` for the neural brain structure

## Optional later

* `stable-baselines3` for reinforcement learning
* `matplotlib` for training charts
* `pymunk` for more advanced 2D physics

---

# Project Structure

```text
neurobot/
├─ main.py
├─ config.py
├─ world/
│  ├─ environment.py
│  ├─ food.py
│  └─ obstacles.py
├─ agents/
│  ├─ robot.py
│  ├─ sensors.py
│  └─ brain.py
├─ render/
│  ├─ renderer.py
│  └─ hud.py
├─ training/
│  ├─ rewards.py
│  ├─ evolution.py
│  └─ rl_env.py
├─ utils/
│  ├─ math_utils.py
│  └─ seed.py
├─ assets/
└─ runs/
```

For the MVP, not all files need to be fully implemented.
But the structure should already suggest the future direction.

---

# MVP in 3 Steps

## STEP 1 — Build the Simulation Sandbox

### Objective

Create a minimal 2D world where the robot and food exist and the simulation runs visually.

### What must be implemented

* project folder structure
* Python environment and dependencies
* Pygame window
* world boundaries
* robot entity with:

  * position
  * angle
  * radius
  * movement speed
* food entity with:

  * position
  * radius
* food random spawn logic
* collision check between robot and food
* score counter
* simple manual control for debugging

### Deliverable

A runnable prototype where:

* the robot is visible
* food is visible
* the robot can be moved
* touching food increases score
* new food respawns correctly

### Why this step matters

This step proves that the simulation foundation works.
Without a stable world loop and rendering, the AI part will be much harder to debug.

### Definition of done

* simulation opens in a window
* robot moves correctly
* food respawns correctly
* score updates correctly
* codebase is organized, not a single messy file

---

## STEP 2 — Add Sensors and the Brain Interface

### Objective

Make the robot perceive the environment and drive its movement through a small neural brain API.

### What must be implemented

* sensor system for the robot
* at minimum these inputs:

  * normalized distance to nearest food
  * relative angle to nearest food
  * distance to wall in front
  * distance to wall left
  * distance to wall right
* sensor visualization on screen
* `brain.py` with a small PyTorch model
* brain input/output interface
* action decoding from neural outputs to robot movement
* switch from manual control to brain-driven control

### Brain suggestion for MVP

Input example:

```text
[food_distance, food_angle, wall_front, wall_left, wall_right]
```

Output example:

```text
[forward_value, turn_value]
```

### Deliverable

A runnable prototype where:

* the robot reads sensor values
* the brain receives those values
* the brain outputs movement commands
* the robot moves because of the brain interface

At this stage, the behavior can still be random or poor.
That is fine.

### Why this step matters

This step completes the core loop:

```text
world -> sensors -> brain -> actions
```

Once this works, the project becomes a real brain/robot simulation.

### Definition of done

* sensor values update correctly every frame
* sensor rays/debug lines can be visualized
* brain receives valid numeric input
* brain outputs drive robot movement
* manual control is no longer required for the normal run mode

---

## STEP 3 — Create the MVP Autonomous Behavior Layer

### Objective

Make the robot behave in a meaningful way and prepare the project for future learning.

### What must be implemented

* one baseline autonomous behavior path
* choose one of these two approaches:

#### Option A — Simple heuristic controller

A non-learning controller that uses the same sensor API as the brain.
For example:

* turn toward food
* avoid walls
* move forward when safe

#### Option B — Simple neural weights test controller

Use fixed or hand-tuned neural weights and verify that the robot can react to inputs.

### Additional required tasks

* episode reset logic
* score reset logic
* optional energy or timer per episode
* simple metrics logging:

  * food collected
  * distance traveled
  * lifetime
* save basic run results inside `runs/`

### Deliverable

A working MVP where:

* the robot acts autonomously
* the simulation can run without user control
* behavior is understandable on screen
* the project is ready for the next phase: evolution or reinforcement learning

### Why this step matters

This step turns the project from a visual toy into a real autonomous-agent prototype.
It also gives you a benchmark before adding learning.

### Definition of done

* robot can run full episodes automatically
* robot can sometimes reach food intentionally or semi-intentionally
* metrics are printed or logged
* the codebase is ready for future training experiments

---

# Recommended MVP Scope

Keep the MVP intentionally small.

## Include

* one robot
* one food target at a time
* rectangular world
* simple wall sensing
* visible debug rendering
* score
* reset/restart logic

## Do not include yet

* multiple robots
* predators
* reproduction
* advanced physics
* reinforcement learning training
* evolution
* obstacle-heavy maps
* fancy UI dashboards
* Flutter frontend

---

# Suggested Development Order

1. Build the world first
2. Make the robot move manually
3. Add food collection and scoring
4. Add sensors
5. Add brain input/output API
6. Let the brain move the robot
7. Add autonomous baseline behavior
8. Only after that, explore learning

---

# Future Phases After MVP

Once the MVP is done, the next upgrades can be:

## Phase 2

* obstacles
* energy system
* more foods
* better sensing
* episode replay

## Phase 3

* evolutionary training
* reinforcement learning with PPO
* compare multiple brains
* save/load best policy

## Phase 4

* multiple robots
* competition
* predator-prey setup
* ecosystem dynamics

## Phase 5

* spiking neural networks
* more biologically inspired control
* connection to a physical robot or webcam-based environment

---

# Success Criteria for MVP

The MVP is successful if:

* it runs smoothly in Python
* the robot exists in a visible 2D world
* food spawning and collection work
* sensors are implemented and visible/debuggable
* the brain interface is connected
* the robot can act autonomously without keyboard control
* the codebase is clean enough for learning experiments later

---

# Notes for the First Agent Task

The first implementation task should focus only on:

## STEP 1 — Build the Simulation Sandbox

That means the agent should ignore learning and neural training for now.
It should only create the visual and structural foundation of the simulation.

The output of Step 1 should be a solid sandbox that you can run, inspect, and debug before moving to sensors and autonomous control.

---

# First Command to the Agent

A good instruction for the first agent run is:

> Build Step 1 of the NeuroBot MVP in Python using Pygame. Create the project structure, simulation window, robot entity, food spawning, collision detection, score tracking, and manual debug controls. Keep the code modular and prepare it for Step 2, where sensors and the brain interface will be added.

---

# Step 1A — Running the Simulation Sandbox

Step 1A focuses only on creating the visual sandbox: window, world boundaries, and an empty simulation loop. The robot, food, collisions, score, and manual controls belong to Step 1B.

## Requirements for Step 1A

Create and activate a Python virtual environment, then install the dependencies.

### 1. Create the virtual environment (once)

From the `neurobot` directory:

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

- **PowerShell (Windows)**:

  ```bash
  .venv\Scripts\Activate.ps1
  ```

- **Command Prompt (Windows)**:

  ```bash
  .venv\Scripts\activate.bat
  ```

- **macOS / Linux (bash/zsh)**:

  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies inside the venv

```bash
pip install -r requirements.txt
```

This installs:

* `pygame` — used directly in Step 1A for the window and loop
* `numpy` — reserved for later math utilities
* `torch` — reserved for the future brain interface

## How to run Step 1A

From the `neurobot` directory:

```bash
python -m main
```

You should see:

* a window titled “NeuroBot - Simulation Sandbox (Step 1A)”
* a dark background
* a rectangle showing the world boundaries

Closing the window cleanly exits the simulation. No robot, food, or score is present yet—that will be added in Step 1B.
