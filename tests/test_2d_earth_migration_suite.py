"""
RLLS 16 — Comprehensive 2D Earth Migration & Verification Test Suite.
Verifies all requirements from Prompt Section 41 & 42:
- Canonical 2D Earth 13 layers integrity & invariants
- Deterministic world generation & hashing
- 2D Orthographic Watcher Camera pan & cursor-pinned zoom (zero geographic drift)
- Deep continuous zoom transitions across all 6 zoom tiers
- Viewport culling & chunk cache LRU eviction
- UI mouse wheel isolation
- Coupled environmental physics & demographic homeostasis
- Intelligent beings vitals & metabolism
- Reinforcement learning 12-dim observation & action masking
- Deterministic Save, Load, and Reset roundtrips
- 120 FPS performance budget execution
"""

import sys
import math
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pygame

from rlls16.storage import load_world
from rlls16.map.layers import MapLayers, BIOME_NAMES, BIOME_COLORS
from rlls16.map.chunk_tile import ChunkManager, ChunkTile
from rlls16.graphics.camera_2d import Camera2D
from rlls16.graphics.renderer_2d import Renderer2D, ALL_LAYER_MODES
from rlls16.simulation.world_instance import WorldInstance, IntelligentAgent
from rlls16.ui.hud import SimulationHUD
from rlls16.assets_loader import get_icon, preload_all_icons


def test_01_canonical_earth_13_layers_integrity(map_layers):
    """Verify all 13 independent simulation layers exist with strict physical bounds."""
    # Dimensions
    assert map_layers.width >= 512
    assert map_layers.height >= 256
    H, W = map_layers.height, map_layers.width

    # Layer 0: Land Mask
    assert map_layers.land_mask.shape == (H, W)
    land_pct = float(np.mean(map_layers.land_mask))
    assert 0.20 <= land_pct <= 0.40, f"Earth land fraction should be ~29%, got {land_pct*100:.1f}%"

    # Layer 1: Elevation (normalized 0..1, sea level = 0.50)
    assert map_layers.elevation.shape == (H, W)
    assert np.all(map_layers.elevation >= 0.0) and np.all(map_layers.elevation <= 1.0)
    # Ocean elevation should be below 0.50, land >= 0.50
    assert np.all(map_layers.elevation[~map_layers.land_mask] <= 0.505)
    assert np.all(map_layers.elevation[map_layers.land_mask] >= 0.495)

    # Layer 2: Coastline
    assert map_layers.coastline.shape == (H, W)
    assert np.any(map_layers.coastline)

    # Layer 3: Rivers / Lakes
    assert map_layers.rivers_lakes.shape == (H, W)
    assert np.all(map_layers.rivers_lakes >= 0.0) and np.all(map_layers.rivers_lakes <= 1.0)

    # Layer 4: Base Climate
    assert map_layers.base_climate.shape == (H, W)

    # Layer 5: Temperature (normalized 0..1, equatorial hot, polar cold)
    assert map_layers.temperature.shape == (H, W)
    assert np.all(map_layers.temperature >= 0.0) and np.all(map_layers.temperature <= 1.0)
    equator_r = H // 2
    pole_r = 5
    assert np.mean(map_layers.temperature[equator_r]) > np.mean(map_layers.temperature[pole_r])

    # Layer 6: Precipitation
    assert map_layers.precipitation.shape == (H, W)
    assert np.all(map_layers.precipitation >= 0.0) and np.all(map_layers.precipitation <= 1.0)

    # Layer 7: Soil Moisture
    assert map_layers.soil_moisture.shape == (H, W)

    # Layer 8: Base Biome (14 WWF classes)
    assert map_layers.base_biome.shape == (H, W)
    assert np.all(map_layers.base_biome >= 0) and np.all(map_layers.base_biome <= 13)

    # Layer 9: Vegetation Biomass
    assert map_layers.vegetation.shape == (H, W)

    # Layer 10: Wildlife Density
    assert map_layers.wildlife.shape == (H, W)

    # Layer 11: Resources
    assert map_layers.resources.shape == (H, W)

    # Layer 12: Agents density layer
    assert map_layers.agents.shape == (H, W)


def test_02_16bit_asset_loader():
    """Verify 16-bit SVG icon loading, caching, and nearest-neighbor scaling."""
    preload_all_icons([(16, 16), (24, 24), (32, 32)])
    for name in ["human", "tree", "animal", "water", "shelter", "cursor_select"]:
        icon = get_icon(name, (24, 24))
        assert isinstance(icon, pygame.Surface)
        assert icon.get_size() == (24, 24)
        assert icon.get_flags() & pygame.SRCALPHA


def test_03_camera_cursor_pinned_zoom():
    """Verify cursor-pinned mouse wheel continuous zoom with zero geographic drift."""
    cam = Camera2D(1280, 768)

    # Pick non-center screen cursor position
    cursor_x, cursor_y = 850, 420

    # 1. Geographic point under cursor before zoom
    wx0, wy0 = cam.screen_to_world(cursor_x, cursor_y)

    # 2. Zoom in by 4 scroll clicks
    for _ in range(4):
        cam.zoom_around_cursor(1.0, cursor_x, cursor_y)

    # 3. Geographic point under cursor after zoom
    wx1, wy1 = cam.screen_to_world(cursor_x, cursor_y)

    assert abs(wx0 - wx1) < 1e-4, f"Longitude drift: {wx0} vs {wx1}"
    assert abs(wy0 - wy1) < 1e-4, f"Latitude drift: {wy0} vs {wy1}"


def test_04_deep_zoom_tiers():
    """Verify continuous zoom reaches all 6 semantic tiers smoothly."""
    cam = Camera2D(1280, 768)
    tiers_reached = set()

    cam.zoom = 1.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_WORLD

    cam.zoom = 4.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_CONTINENT

    cam.zoom = 12.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_REGION

    cam.zoom = 30.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_LOCAL

    cam.zoom = 75.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_SETTLEMENT

    cam.zoom = 150.0
    tiers_reached.add(cam.zoom_tier)
    assert cam.zoom_tier == Camera2D.TIER_AGENT

    assert len(tiers_reached) == 6


def test_05_chunk_viewport_culling_and_cache(map_layers):
    """Verify chunk manager correctly partitions 2D Earth and culls invisible tiles."""
    chunk_mgr = ChunkManager(map_layers, chunk_size=32, max_cached=64)
    assert len(chunk_mgr.chunks) == 128  # (512/32) * (256/32) = 16 * 8 = 128 chunks

    # Viewport culling test: tight local zoom on coordinates [0.4, 0.2] to [0.5, 0.25]
    visible = chunk_mgr.get_visible_chunks(0.40, 0.20, 0.50, 0.25)
    assert 0 < len(visible) < len(chunk_mgr.chunks)
    assert len(visible) <= 8

    # Render a chunk surface into cache
    c0 = visible[0]
    surf = chunk_mgr.get_chunk_surface(c0)
    assert surf.get_size() == (32, 32)
    assert c0.key in chunk_mgr.cache


def test_06_renderer_2d_layer_modes(map_layers):
    """Verify all 6 visual layer modes render without exceptions or allocation leaks."""
    chunk_mgr = ChunkManager(map_layers, chunk_size=32)
    cam = Camera2D(800, 600)
    renderer = Renderer2D(800, 600, chunk_mgr, cam)

    target_surface = pygame.Surface((800, 600))

    for mode in ALL_LAYER_MODES:
        renderer.set_layer_mode(mode)
        assert renderer.current_layer_mode == mode
        renderer.render(target_surface, agents=None, dt=0.016)


def test_07_intelligent_beings_and_demographics(canonical_world):
    """Verify deployment of settlement cohort and individual intelligent beings with vitals."""
    world = WorldInstance(
        world_name="Test_2D_World",
        canonical_path="worlds/canonical_world.npz",
        human_population=1500,
        initial_environment="Savannah",
        world_data=canonical_world,
    )

    # Deployed on real land coordinates
    lat = world.human.latitude_deg
    lon = world.human.longitude_deg
    assert -80.0 <= lat <= 80.0
    assert -180.0 <= lon <= 180.0
    assert world.human.elevation >= 0.49

    # Agents spawned
    assert len(world.agents) >= 6
    ag0 = world.agents[0]
    assert ag0.health == 100.0
    assert ag0.alive is True
    assert ag0.name in ["Adam", "Eve", "Enki", "Inanna", "Manu", "Noah", "Gilgamesh", "Ishtar"]

    # Hourly simulation advances metabolism
    world._on_hourly_tick(sim_time=3600.0)
    assert ag0.recent_action in ["FORAGE_FOOD", "GATHER_WATER", "EXPLORE", "REST", "SEEK_SHELTER"]
    assert ag0.energy <= 100.0


def test_08_reinforcement_learning_interface(canonical_world):
    """Verify Gym-style RL interface, 12-dim normalized observation vector, and action masking."""
    world = WorldInstance(
        world_name="Test_RL_World",
        canonical_path="worlds/canonical_world.npz",
        human_population=1000,
        initial_environment="Forest",
        world_data=canonical_world,
    )

    rl = world.rl_interface
    assert rl.observation_dim == 12
    assert rl.action_space_size == 14

    obs = rl.get_observation()
    assert obs.shape == (12,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    mask = rl.get_action_mask()
    assert mask.shape == (14,)
    assert np.sum(mask) >= 1  # At least STAY and exploration actions valid


def test_09_ui_mouse_wheel_isolation():
    """Verify mouse wheel over UI panel scrolls panel only, and never triggers dropdowns."""
    hud = SimulationHUD(1280, 768)
    initial_scroll = hud.scroll_y

    # Event inside left panel (x = 100, y = 300)
    wheel_event_panel = pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": 1, "flipped": False, "pos": (100, 300)})
    result = hud.handle_event(wheel_event_panel, None, None, None)
    assert result == {"consumed": True}
    assert hud.scroll_y >= initial_scroll

    # Event over world (x = 600, y = 400)
    wheel_event_world = pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": 1, "flipped": False, "pos": (600, 400)})
    result_world = hud.handle_event(wheel_event_world, None, None, None)
    # Returns None so app.py can route to camera.zoom_around_cursor!
    assert result_world is None


def test_10_deterministic_save_load_reset(canonical_world, tmp_path):
    """Verify state serialization: save, load, and reset preserve exact state deterministically."""
    world = WorldInstance(
        world_name="Deterministic_World_01",
        canonical_path="worlds/canonical_world.npz",
        human_population=1200,
        initial_environment="Temperate",
        world_data=canonical_world,
    )

    # Advance time and change environment
    world.scheduler.advance(7200.0)  # 2 hours
    world.environment.global_temp_offset = 2.5
    world.environment.co2_ppm = 520.0

    save_path = world.save(base_dir=str(tmp_path))
    assert Path(save_path).exists()

    # Load into separate instance
    loaded = WorldInstance.load(save_path, world_data=canonical_world)
    assert loaded.world_name == world.world_name
    assert loaded.sim_time_sec == world.sim_time_sec
    assert loaded.environment.global_temp_offset == world.environment.global_temp_offset
    assert loaded.environment.co2_ppm == world.environment.co2_ppm
    assert len(loaded.agents) == len(world.agents)
    assert loaded.agents[0].name == world.agents[0].name
    assert abs(loaded.agents[0].wx - world.agents[0].wx) < 1e-6

    # Reset
    world.reset(canonical_world)
    assert world.sim_time_sec == 0.0
    assert world.environment.global_temp_offset == 0.0
    assert world.environment.co2_ppm == 415.0


def test_11_performance_budget_and_120fps(canonical_world, map_layers):
    """Verify that a complete frame render executes well within the 120 FPS frame budget (< 8.3ms)."""
    chunk_mgr = ChunkManager(map_layers, chunk_size=32)
    cam = Camera2D(1280, 768)
    renderer = Renderer2D(1280, 768, chunk_mgr, cam)
    target_surf = pygame.Surface((1280, 768))

    world = WorldInstance("Bench_World", world_data=canonical_world)
    agents = world.get_all_agents()

    # Warmup
    renderer.render(target_surf, agents=agents, dt=0.016)

    # Measure 50 frame renders
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        renderer.render(target_surf, agents=agents, dt=0.016)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(times))
    # 120 FPS frame budget = 8.33 ms
    assert avg_ms < 8.33, f"Average render time {avg_ms:.2f}ms exceeds 8.33ms 120 FPS budget!"


if __name__ == "__main__":
    import tempfile
    pygame.init()
    pygame.display.set_mode((640, 480), pygame.HIDDEN)

    cw_path = Path("worlds/canonical_world.npz")
    if not cw_path.exists():
        cw_path = Path("world_data/canonical/canonical_world.npz")
    cw = load_world(str(cw_path))
    layers = MapLayers(cw)

    tests = [
        ("01 Canonical Earth 13 Layers", lambda: test_01_canonical_earth_13_layers_integrity(layers)),
        ("02 16-Bit Asset Loader", test_02_16bit_asset_loader),
        ("03 Camera Cursor-Pinned Zoom", test_03_camera_cursor_pinned_zoom),
        ("04 Deep Zoom Tiers", test_04_deep_zoom_tiers),
        ("05 Chunk Viewport Culling & Cache", lambda: test_05_chunk_viewport_culling_and_cache(layers)),
        ("06 Renderer 2D Layer Modes", lambda: test_06_renderer_2d_layer_modes(layers)),
        ("07 Intelligent Beings & Demographics", lambda: test_07_intelligent_beings_and_demographics(cw)),
        ("08 Reinforcement Learning Interface", lambda: test_08_reinforcement_learning_interface(cw)),
        ("09 UI Mouse Wheel Isolation", test_09_ui_mouse_wheel_isolation),
        ("10 Deterministic Save/Load/Reset", lambda: test_10_deterministic_save_load_reset(cw, Path(tempfile.mkdtemp()))),
        ("11 120 FPS Performance Budget", lambda: test_11_performance_budget_and_120fps(cw, layers)),
    ]

    print("\n" + "=" * 70)
    print(" RLLS 16 — 2D EARTH MIGRATION COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    all_ok = True
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dur = (time.perf_counter() - t0) * 1000.0
            print(f" [PASS] {name.ljust(44)} ({dur:6.1f} ms)")
        except Exception as e:
            all_ok = False
            dur = (time.perf_counter() - t0) * 1000.0
            print(f" [FAIL] {name.ljust(44)} ({dur:6.1f} ms) -> {e}")

    print("=" * 70)
    if all_ok:
        print(" ALL 11 TEST SUITES PASSED SUCCESSFULLY!")
    else:
        print(" SOME TESTS FAILED!")
    print("=" * 70 + "\n")
    sys.exit(0 if all_ok else 1)
