# RLLS 16 — Canonical 2D Earth Observation & Life Simulation Platform (v1.0)

RLLS 16 is a deterministic, data-driven 2D rectangular Earth observation and life-simulation desktop platform viewed from above ("a watcher observing the world"). It features real geographic baselines from Natural Earth, 13 independent simulation layers, a 120 FPS CPU-based watcher renderer, an orthographic camera with cursor-pinned zoom, coupled Earth-system physics, intelligent beings with homeostasis, and reinforcement learning.

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
                         | 2D Map Renderer   |                                       | Simulation Engine |
                         | (120 FPS Target)  |                                       |  (Deterministic)  |
                         +---------+---------+                                       +---------+---------+
                                   |                                                           |
              +--------------------+--------------------+                                      |
              |           |        |          |         |                                      |
         +----v----+ +----v----+ +-v--+   +---v---+ +---v---+                                  |
         | Spatial | | 13 Map  | |16b |   |Lat/Lon| |Dynamic|                                  |
         | Chunks  | | Layers  | |Icon|   | Grid  | |Weather|                                  |
         | Cache   | | (Layers)| |Pack|   | Lines | | Drift |                                  |
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

1. **Deterministic 2D Canonical Earth**:
   - Derived directly from real Earth datasets (Natural Earth vector layers: land, ocean, coastlines, rivers, lakes).
   - Equirectangular 2:1 rectangular world format ($512 \times 256$ standard baseline).
   - Generates offline in ~1.1 seconds with deterministic checksum:
     `sha256:25d9eaeefbf8d50be9175d3bc07e5d961d4ffeec30cc84202067b486029b41f0`.

2. **13 Independent Queryable Simulation Layers**:
   - `Layer 0`: Ocean / Land mask
   - `Layer 1`: Elevation (hypsometric relief, sea level = 0.50)
   - `Layer 2`: Coastlines
   - `Layer 3`: Rivers & Lakes (Hydrology)
   - `Layer 4`: Climate
   - `Layer 5`: Temperature
   - `Layer 6`: Precipitation
   - `Layer 7`: Soil Moisture
   - `Layer 8`: Biomes (14 WWF terrestrial biomes)
   - `Layer 9`: Vegetation Biomass
   - `Layer 10`: Wildlife Density
   - `Layer 11`: Resources
   - `Layer 12`: Agents Density

3. **2D Orthographic Watcher Camera**:
   - **Continuous Zoom**: Smooth zoom range across 6 semantic tiers:
     `WORLD` (1.0x - 2.5x) → `CONTINENT` (2.5x - 7.0x) → `REGION` (7.0x - 20.0x) → `LOCAL` (20.0x - 50.0x) → `SETTLEMENT` (50.0x - 120.0x) → `AGENT` (>120.0x).
   - **Cursor-Anchored Stability**: Geographic coordinate directly under cursor remains geographically pinned during mouse wheel zoom (zero drift verified).
   - Smooth `fly_to` cubic ease-out animation for locating agents.

4. **120 FPS Target CPU Map Renderer**:
   - Partitions Earth into 128 spatial tiles with LRU surface cache and viewport culling.
   - 6 Visual Layer Modes: `NATURAL`, `ELEVATION`, `TEMPERATURE`, `PRECIPITATION`, `BIOMES`, `WATER`.
   - 16-Bit ecological icons scaled with nearest-neighbor interpolation.
   - Dynamic cloud drift and lat/lon coordinate grid.

5. **Intelligent Beings & Multi-Rate Simulation**:
   - Spawns intelligent beings with vitals (health, energy, hydration, hunger, comfort), memory, and actions.
   - Deploys populations onto real Earth land coordinates matching chosen habitat.
   - Gym-compatible 12-dim observation vector and 14 discrete actions with action masking.

6. **UI & Controls**:
   - Snapping speed multipliers: `1x`, `2x`, `4x`, `6x`, `8x`, `16x`, `32x`, `64x`, `100x`.
   - Strict scroll isolation: mouse wheel over UI panels scrolls panel content only; mouse wheel over map zooms camera; scrolling never toggles dropdowns.
   - Navigation toolbar: `RESET`, `LAYERS`, `LOCATE`, `WEATHER`, `ICONS`, `GRID`, `INFO`.

---

## Quick Start

### Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the desktop simulation workstation
python main.py

# Launch direct 2D map preview
python main.py --preview

# Run 10-stage diagnostic suite headlessly
python main.py --headless-diagnostic

# Run comprehensive 2D test suite
python tests/test_2d_earth_migration_suite.py
```
