"""
RLLS 16 — Comprehensive Planetary Navigation & Advanced Rendering Verification Suite.

Validates:
1. Wheel normalization, inertial zoom impulse, and exponential damping.
2. Zoom-to-cursor raycasting, unprojecting to terrain, and focus shifting.
3. Surface-tangent navigation (N, F_tan, R_tan) and keyboard acceleration/damping.
4. Dynamic near clip and horizon-tracking far plane calculations.
5. Astronomical Sun vector calculation and shader uniform passing.
6. Deterministic per-tile vegetation instancing and hardware-instanced rendering.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pygame
import moderngl

pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))

from rlls16.storage import load_world
from rlls16.graphics.math3d import (
    screen_to_ray, ray_sphere_intersect, unproject_terrain_hit,
    compute_surface_tangent_basis, sample_terrain_altitude
)
from rlls16.graphics.camera import OrbitCamera, PlanetCamera
from rlls16.graphics.quadtree import QuadtreeNode, LODManager
from rlls16.graphics.terrain_tile import (
    build_tile_mesh_data, build_tile_vegetation_instances,
    create_base_tree_mesh, TerrainTile, TerrainTilePool
)
from rlls16.graphics.renderer import EarthRenderer, EarthTier
from rlls16.graphics.shaders import (
    CUBESPHERE_TERRAIN_VS, CUBESPHERE_TERRAIN_FS,
    PLANET_VS, EARTH_FS, VEGETATION_VS, VEGETATION_FS
)


def run_planetary_navigation_suite():
    print("\n" + "=" * 70)
    print(" RLLS 16 — PLANETARY NAVIGATION & ADVANCED RENDERING TEST SUITE")
    print("=" * 70)

    # 1. Setup headless OpenGL context
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((1024, 768), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    ctx = moderngl.create_context()

    canonical_file = pkg_root / "worlds" / "canonical_world.npz"
    assert canonical_file.exists(), f"Canonical world missing: {canonical_file}"
    world_data = load_world(str(canonical_file))
    earth_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # -----------------------------------------------------------------
    # TEST 1: Inertial Mouse Wheel & Logarithmic Zoom
    # -----------------------------------------------------------------
    print("\n[TEST 1/6] Inertial Mouse Wheel & Logarithmic Altitude Response:")
    cam = PlanetCamera(aspect=1024.0 / 768.0)
    cam.set_world_data(world_data)
    cam.focus_object("EARTH", earth_pos, distance=22.0, snap=True)

    initial_dist = cam.distance
    # Zoom in impulse with delta 1.0
    cam.handle_mouse_wheel(1.0)
    assert cam.zoom_velocity < 0.0, "Zoom in impulse should produce negative zoom velocity (dolly forward)"
    init_vel = cam.zoom_velocity
    print(f"  -> Orbit scale wheel impulse zoom velocity: {init_vel:.4f}")

    # Integrate 10 frames of exponential damping
    for _ in range(10):
        cam.update(earth_pos, dt=0.016)
    assert abs(cam.zoom_velocity) < abs(init_vel), "Zoom velocity must exponentially decay"
    assert cam.distance < initial_dist, "Camera distance must have decreased towards Earth"
    print(f"  -> Inertial decay verified: velocity {init_vel:.4f} -> {cam.zoom_velocity:.4f}, distance {initial_dist:.2f} -> {cam.distance:.2f}")

    # Test high-resolution wheel ticks (e.g. precise_y = 120 raw ticks)
    cam.distance = 20.0
    cam.target_distance = 20.0
    cam.zoom_velocity = 0.0
    cam.handle_mouse_wheel(120.0) # Raw high-res mouse wheel event
    assert abs(cam.zoom_velocity) < 100.0, "High-res wheel ticks must be normalized"
    print(f"  -> High-res wheel normalization verified (raw 120.0 -> normalized velocity {cam.zoom_velocity:.4f})")

    # -----------------------------------------------------------------
    # TEST 2: Zoom-to-Cursor & Surface Unprojection
    # -----------------------------------------------------------------
    print("\n[TEST 2/6] Planetary Zoom-to-Cursor (zoomToCursor) & Surface Raycasting:")
    cam.focus_object("EARTH", earth_pos, distance=12.0, snap=True)
    view_mat = cam.get_view_matrix()
    proj_mat = cam.get_projection_matrix()

    # Cursor at screen center (512, 384)
    hit_center = unproject_terrain_hit(512, 384, 1024, 768, view_mat, proj_mat, earth_pos, 5.0, world_data)
    assert hit_center is not None, "Ray through screen center must hit Earth"
    hit_world, lat_c, lon_c = hit_center
    assert np.isclose(np.linalg.norm(hit_world - earth_pos), 5.0, atol=0.25)
    print(f"  -> Raycast hit center at lat={math.degrees(lat_c):.1f}°, lon={math.degrees(lon_c):.1f}°, radius={np.linalg.norm(hit_world):.3f}")

    # Off-center cursor (e.g. 650, 250)
    hit_off = unproject_terrain_hit(650, 250, 1024, 768, view_mat, proj_mat, earth_pos, 5.0, world_data)
    assert hit_off is not None, "Ray through off-center coordinate on disk must hit Earth"
    off_world, lat_o, lon_o = hit_off

    # Test zoom_to_cursor shifting target towards cursor point
    prev_target = cam.target_pos.copy()
    cam.handle_mouse_wheel(1.0, mouse_pos=(650, 250), width=1024, height=768)
    shift_vector = cam.target_pos - prev_target
    dot_shift = np.dot(shift_vector, off_world - prev_target)
    assert dot_shift > 0.0, "Camera target must shift in the direction of the cursor surface point"
    print("  -> Zoom-to-cursor surface anchor shift verified: target moved toward cursor point")

    # -----------------------------------------------------------------
    # TEST 3: Surface-Tangent Navigation Basis (WASD/QE)
    # -----------------------------------------------------------------
    print("\n[TEST 3/6] Surface-Tangent Orthonormal Basis & Keyboard Navigation:")
    # Focus point at positive Y (North Pole)
    focus_np = np.array([0.0, 5.0, 0.0], dtype=np.float32)
    cam_fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    N, f_tan, r_tan = compute_surface_tangent_basis(focus_np, earth_pos.astype(np.float32), cam_fwd)

    # Verify orthonormality
    assert np.isclose(np.linalg.norm(N), 1.0, atol=1e-4), "N must be unit length"
    assert np.isclose(np.linalg.norm(f_tan), 1.0, atol=1e-4), "f_tan must be unit length"
    assert np.isclose(np.linalg.norm(r_tan), 1.0, atol=1e-4), "r_tan must be unit length"
    assert abs(np.dot(N, f_tan)) < 1e-4, "f_tan must be orthogonal to surface normal N"
    assert abs(np.dot(N, r_tan)) < 1e-4, "r_tan must be orthogonal to surface normal N"
    assert abs(np.dot(f_tan, r_tan)) < 1e-4, "f_tan and r_tan must be orthogonal"
    print("  -> Surface tangent basis orthonormality verified (N, F_tan, R_tan strictly orthogonal)")

    # Test velocity integration with simulated keypress
    cam.focus_object("EARTH", earth_pos, distance=5.10, snap=True)
    fake_keys = {k: False for k in range(1000)}
    fake_keys[pygame.K_w] = True
    cam.handle_keyboard(fake_keys, dt=0.016)
    assert np.linalg.norm(cam.cam_velocity) > 0.0, "Pressing W must impart surface-tangent forward velocity"
    initial_cam_vel = np.linalg.norm(cam.cam_velocity)

    # Damping test
    fake_keys[pygame.K_w] = False
    cam.update(earth_pos, dt=0.016)
    assert np.linalg.norm(cam.cam_velocity) < initial_cam_vel, "Velocity must damp exponentially when key released"
    print(f"  -> Keyboard velocity and acceleration verified: initial {initial_cam_vel:.4f} -> damped {np.linalg.norm(cam.cam_velocity):.4f}")

    # -----------------------------------------------------------------
    # TEST 4: Horizon-Tracking Frustum & Near Plane
    # -----------------------------------------------------------------
    print("\n[TEST 4/6] Dynamic Near Plane & Visible Horizon Far Plane:")
    # In orbit: distance = 25.0
    cam.distance = 25.0
    cam.target_distance = 25.0
    cam.update(earth_pos, dt=0.016)
    # Visible horizon: sqrt(25^2 - 5^2) = sqrt(600) ~= 24.5 units
    assert cam.far > 24.5, "Far plane must encompass the visible horizon in orbit"
    assert cam.near <= 0.5, "Near plane at orbit must be safely bounded"
    orbit_near = cam.near
    orbit_far = cam.far
    print(f"  -> Orbit (d=25.0): near={orbit_near:.4f}, far={orbit_far:.1f}")

    # Near ground: distance = 5.02 (altitude = 0.02)
    cam.distance = 5.02
    cam.target_distance = 5.02
    cam.update(earth_pos, dt=0.016)
    assert cam.near < 0.005, "Near plane must scale down to sub-millimeter scale near ground"
    ground_near = cam.near
    ground_far = cam.far
    print(f"  -> Near-ground (d=5.02): near={ground_near:.6f}, far={ground_far:.1f}")
    assert ground_near < orbit_near, "Near plane must decrease as altitude decreases"
    print("  -> Dynamic near/far frustum scaling verified: PASSED")

    # -----------------------------------------------------------------
    # TEST 5: Astronomical Lighting Flow
    # -----------------------------------------------------------------
    print("\n[TEST 5/6] Astronomical Sun Vector & Multi-Body Lighting Architecture:")
    prog_cubesphere = ctx.program(vertex_shader=CUBESPHERE_TERRAIN_VS, fragment_shader=CUBESPHERE_TERRAIN_FS)
    prog_earth = ctx.program(vertex_shader=PLANET_VS, fragment_shader=EARTH_FS)
    prog_veg = ctx.program(vertex_shader=VEGETATION_VS, fragment_shader=VEGETATION_FS)

    assert 'u_sun_dir' in prog_cubesphere, "CUBESPHERE shader must accept u_sun_dir uniform"
    assert 'u_sun_dir' in prog_earth, "EARTH fallback shader must accept u_sun_dir uniform"
    assert 'u_sun_dir' in prog_veg, "VEGETATION shader must accept u_sun_dir uniform"

    renderer = EarthRenderer(ctx, world_data, prog_cubesphere, prog_earth, {"is_dedicated": True})
    assert renderer.prog_vegetation is not None, "Renderer must initialize prog_vegetation"

    # Test lighting direction at an orbital point
    sun_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    earth_orbit_pos = np.array([120.0, 0.0, 45.0], dtype=np.float32)
    diff = sun_pos - earth_orbit_pos
    expected_sun_dir = diff / np.linalg.norm(diff)

    # Render pass with astronomical position
    tex_albedo = ctx.texture((32, 32), 4)
    tex_clouds = ctx.texture((32, 32), 4)
    renderer.render(
        camera=cam,
        sim_time_sec=100.0,
        earth_pos=earth_orbit_pos.astype(np.float64),
        earth_rot_angle=0.45,
        solar_irradiance=1.0,
        show_clouds=True,
        show_atmo=True,
        tex_albedo=tex_albedo,
        tex_clouds=tex_clouds,
        width=1024,
        height=768,
    )
    # Check that u_sun_dir in shader was updated with expected direction
    actual_sun_dir = np.array(prog_cubesphere['u_sun_dir'].value, dtype=np.float32)
    assert np.allclose(actual_sun_dir, expected_sun_dir, atol=1e-4), "Shader u_sun_dir must match true astronomical sun direction"
    print(f"  -> Astronomical Sun direction verified in shader: {actual_sun_dir} (dot with expected = {np.dot(actual_sun_dir, expected_sun_dir):.4f})")

    # -----------------------------------------------------------------
    # TEST 6: Deterministic Instanced Vegetation System
    # -----------------------------------------------------------------
    print("\n[TEST 6/6] Deterministic Per-Tile Vegetation Instancing & Hardware Draw:")
    # Build tree mesh
    tree_vbo, tree_ibo = create_base_tree_mesh(ctx)
    assert tree_vbo is not None and tree_ibo is not None
    print(f"  -> Shared low-poly 3D tree mesh created ({tree_vbo.size} bytes)")

    # Find a land tile at LOD 5
    lod_mgr = LODManager(max_lod=6)
    # Place camera directly over land at low altitude
    cam_pos_land = np.array([0.0, 0.0, 5.08], dtype=np.float64)
    leaves = lod_mgr.update(cam_pos_land, earth_pos, radius=5.0, world_data=world_data)
    deep_leaves = [n for n in leaves if n.level >= 4]
    assert len(deep_leaves) > 0, "Should have deep quadtree leaves near ground"

    # Test vegetation instance generation
    veg_found = False
    for node in deep_leaves:
        instances = build_tile_vegetation_instances(node, world_data, radius=5.0)
        if instances is not None and len(instances) > 0:
            veg_found = True
            # Check instance buffer shape and properties
            assert instances.shape[1] == 8, "Each vegetation instance must have 8 components [x,y,z,scale,type,nx,ny,nz]"
            assert np.all(instances[:, 3] > 0.0), "All tree instances must have positive scale"
            assert np.all((instances[:, 4] >= 0.0) & (instances[:, 4] <= 2.5)), "Species type must be in valid range [0, 2]"
            print(f"  -> Tile {node.key} generated {len(instances)} deterministic tree instances")

            # Test hardware-instanced VAO setup and rendering
            verts, idxs = build_tile_mesh_data(node, world_data, grid_size=16, radius=5.0)
            tile = TerrainTile(ctx, prog_cubesphere, verts, idxs, node.key)
            tile.setup_vegetation(ctx, prog_veg, tree_vbo, tree_ibo, instances)
            assert tile.veg_vao is not None
            assert tile.veg_count == len(instances)

            # Test rendering instanced trees in frame buffer
            fbo = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
            fbo.use()
            ctx.enable(moderngl.DEPTH_TEST)
            tile.render_vegetation()
            ctx.screen.use()
            fbo.release()
            print("  -> Hardware-instanced draw call (vao.render(instances=N)) executed successfully: PASSED")
            tile.release()
            break

    assert veg_found, "At least one land tile at deep LOD must contain instanced vegetation"

    print("\n" + "=" * 70)
    print(" RESULT: ALL PLANETARY NAVIGATION & ADVANCED RENDERING TESTS PASSED 100%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_planetary_navigation_suite()
