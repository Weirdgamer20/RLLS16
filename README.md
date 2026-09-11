# RLLS 16 — 3D Artificial-World Simulation System (v0.3)

RLLS 16 is a resource-efficient 3D artificial-world simulation desktop system built with a hardware-accelerated OpenGL 3.3 / ModernGL rendering pipeline, an immutable canonical Earth foundation, deterministic astronomical mechanics, and a futuristic simulation workstation interface.

---

## Architecture Overview

```text
                           +---------------------------+
                           |       main.py Entry       |
                           +-------------+-------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
           +---------v---------+                   +---------v---------+
           |    Home Screen    |                   |  Main Simulation  |
           | (Create/Load Wld) |                   |    Workstation    |
           +-------------------+                   +---------+---------+
                                                             |
                               +-----------------------------+-----------------------------+
                               |                                                           |
                     +---------v---------+                                       +---------v---------+
                     |  3D OpenGL Scene  |                                       | Simulation Engine |
                     | (ModernGL Native) |                                       |  (Deterministic)  |
                     +---------+---------+                                       +---------+---------+
                               |                                                           |
          +--------------------+--------------------+                                      |
          |           |        |          |         |                                      |
     +----v----+ +----v----+ +-v--+   +---v---+ +---v---+                                  |
     |Starfield| |   Sun   | |Eart|   | Moon  | |Orbits/|                                  |
     | Skybox  | | (Corona,| |h 3D|   | (Tidal| | Grid  |                                  |
     | 3K Pts  | | Light)  | |Mesh|   | Orbit)| | Lines |                                  |
     +---------+ +---------+ +----+   +-------+ +-------+                                  |
                                                                                           |
                               +-----------------------------------------------------------+
                               |
                     +---------v---------+
                     | 2D Workstation UI |
                     | - Top Time Bar    |
                     | - Left Accordions |
                     | - Right Nav Tools |
                     | - Bottom Telemetry|
                     +-------------------+
```

---

## Key Features

1. **Hardware-Accelerated 3D Space (ModernGL + OpenGL 3.3 Core)**:
   - Deep-space background with 3,000 deterministic stars and cosmos atmosphere.
   - 3D Emissive Sun at the origin emitting directional light and corona limb darkening.
   - 3D Canonical Earth with terrain relief, ocean specular reflections, day/night dynamic terminator, biomes, and translucent rotating cloud layer.
   - 3D Moon orbiting Earth with tidal locking and realistic lunar regolith shading.
   - Anti-aliased 3D orbital rings (Earth around Sun, Moon around Earth) and spatial reference grid.

2. **Decoupled Architecture**:
   - **Canonical World**: `worlds/canonical_world.npz` is immutable and generated once.
   - **Simulation Instances**: Saved under `simulations/<world_name>/world_state.json` containing only instance state (simulation clock, human population, optimal settlement site, mutable environment parameters).

3. **Camera & Navigation**:
   - **Mouse (Primary)**:
     - Left-click drag: Orbit camera around focus target.
     - Right/Middle-click drag: Pan camera.
     - Mouse wheel: Smooth zoom.
     - Double-click: Ray-cast and smoothly focus Sun, Earth, or Moon.
   - **Keyboard**:
     - `W`/`A`/`S`/`D`: Move/pan camera in view plane.
     - `Q`/`E`: Vertical camera elevation.
     - `+`/`-`: Zoom in/out.
     - `F`: Focus Earth/active target.
     - `Space`: Pause/resume simulation.
     - `ESC`: Close overlay / reset focus.

4. **Right Navigation Toolbar**:
   - `FOCUS`: Cycle focus between Solar System, Earth, Moon, and Sun.
   - `EARTH`: Instantly focus Earth.
   - `ORBIT`: Toggle orbital paths on/off.
   - `GRID`: Toggle spatial coordinate grid.
   - `INFO`: Toggle astronomical telemetry HUD card.
   - `MEASURE`: Distance inspection mode.
   - `LOCATE`: Vector GPS target crosshair tool that smoothly glides camera to human settlement coordinates.

5. **Top Bar & Time Controls**:
   - Deterministic calendar: `Year 000,001 | Day 001 | 00:00:00`.
   - Speed multipliers: `[▶ PLAY]`, `[⏸ PAUSE]`, `[1×]`, `[10×]`, `[100×]`, `[1000×]`.

6. **Left Hierarchical Panel**:
   - Collapsible sections: `WORLD`, `WORLD / SPACE`, `EARTH`, `LIFE`, `REINFORCEMENT LEARNING`, `VISUALIZATION`, `DATA & ANALYTICS`.
   - Interactive environmental sliders for solar irradiance, global temperature offset, cloud cover, and CO2.
   - Interactive RL goal toggle (`Survive`, `Growth`, `Exploration`, `Culture`).

7. **Coupled Earth-System Simulation Engine**:
   - **Multi-Rate Scheduler**: Decouples 60 FPS graphics from simulation physics (1 tick = 1 simulated hour; hourly weather, 6-hourly pressure, daily hydrology/demographics, monthly climate aggregation).
   - **Reduced-Order Physics Solver**: Surface energy balance ($Q_{solar} - Q_{lw} - Q_{latent} - Q_{sensible}$), pressure gradients, Coriolis-deflected horizontal winds, Clausius-Clapeyron moisture advection, and orographic precipitation.
   - **Human Cognition & Cultural Lore ($H_t = [C, M, K, E, S, B]$)**: Low-initial-knowledge agents, physiological homeostasis, empirical discoveries, social lore transmission across generations, and catastrophic knowledge decay.
   - **Hierarchical RL Interface**: Standardized 12-dim normalized observation vector, 14 hierarchical discrete actions with dynamic action masking, and multi-objective reward formulations.

---

## How to Run

Activate your virtual environment and execute:

```powershell
# Run the 3D Desktop Application
python main.py
```
