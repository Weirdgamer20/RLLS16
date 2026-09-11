"""
RLLS 16 — Standalone 3D Planetary World Preview.
Unified wrapper using the authoritative SceneRenderer and Cube-Sphere Quadtree engine.
"""

import sys
import math
import time
from pathlib import Path
import pygame
import moderngl
import numpy as np

from .storage import load_world
from .graphics.renderer import SceneRenderer
from .graphics.camera import OrbitCamera
from .graphics.gpu_config import apply_gpu_environment_hints, configure_pygame_gl_attributes
from .simulation.astronomy import compute_astronomical_state, SUN_POSITION


def preview(target_dir: str = "worlds/canonical_world"):
    path = Path(target_dir)
    npz_path = path / "canonical_world.npz" if path.is_dir() else path

    if not npz_path.exists():
        npz_path = Path("worlds/canonical_world.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find canonical world dataset at {npz_path}. Run python generate_world.py first.")

    print(f"[RLLS 16] Launching 3D Planetary Preview for: {npz_path.resolve()}")
    world_data = load_world(str(npz_path))

    apply_gpu_environment_hints()
    pygame.init()
    width, height = 1200, 800
    pygame.display.set_caption(f"RLLS 16 — 3D Planetary World Preview [{npz_path.name}]")

    configure_pygame_gl_attributes()

    window = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    clock = pygame.time.Clock()

    ctx = moderngl.create_context()
    renderer = SceneRenderer(ctx, width, height, world_data)
    camera = OrbitCamera(aspect=width / height)

    # Focus Earth initially
    earth_pos = np.array([120.0, 0.0, 0.0], dtype=np.float32)
    camera.focus_object("EARTH", earth_pos, distance=22.0)

    print("\n[Controls]")
    print("  * Left Mouse Drag   : Orbit camera around planet")
    print("  * Right Mouse Drag  : Pan camera")
    print("  * Mouse Wheel       : Logarithmic Zoom (Space -> Orbital -> Regional -> Surface)")
    print("  * W/A/S/D           : Move view position")
    print("  * F                 : Focus Earth")
    print("  * C                 : Toggle CubeSphere LOD / Classic Sphere")
    print("  * Space             : Pause/Resume dynamic rotation")
    print("  * ESC / Q           : Exit preview\n")

    sim_time = 0.0
    auto_rotate = True
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        if auto_rotate:
            sim_time += dt * 3600.0 * 0.5 # half simulated hour per second

        astro_state = compute_astronomical_state(sim_time)
        earth_pos = astro_state["earth_pos"]
        moon_pos = astro_state["moon_pos"]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(400, event.w), max(300, event.h)
                renderer.resize(width, height)
                camera.set_aspect(width / height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                camera.handle_mouse_down(event.button, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                camera.handle_mouse_up(event.button)
            elif event.type == pygame.MOUSEMOTION:
                camera.handle_mouse_motion(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEWHEEL:
                camera.handle_mouse_wheel(event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_rotate = not auto_rotate
                elif event.key == pygame.K_f:
                    camera.focus_object("EARTH", earth_pos, distance=22.0)
                elif event.key == pygame.K_c:
                    renderer.use_cubesphere = not renderer.use_cubesphere
                    mode_name = "CubeSphere Quadtree LOD" if renderer.use_cubesphere else "Classic UV Sphere"
                    print(f"Renderer Mode Toggled: {mode_name}")

        key_state = pygame.key.get_pressed()
        camera.handle_keyboard(key_state, dt)

        # Track Earth if camera target is Earth
        target = earth_pos if camera.focus_target_name == "EARTH" else None
        camera.update(target, dt)

        renderer.render(
            camera=camera,
            sim_time_sec=sim_time,
            earth_pos=earth_pos,
            earth_rot_angle=astro_state["earth_rot_angle"],
            moon_pos=moon_pos,
            moon_rot_angle=astro_state["moon_rot_angle"],
            solar_irradiance=1.0,
            ui_surface=None
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "worlds/canonical_world"
    preview(t)
