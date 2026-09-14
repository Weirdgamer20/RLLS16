"""
RLLS 16 — Standalone 2D Canonical Earth Preview.
Allows fast inspection of the 2D rectangular Earth, layers, pan, and cursor-pinned zoom.
"""

import sys
import time
from pathlib import Path
import pygame
import numpy as np

from .storage import load_world
from .map.layers import MapLayers
from .map.chunk_tile import ChunkManager
from .graphics.camera_2d import Camera2D
from .graphics.renderer_2d import Renderer2D, ALL_LAYER_MODES


def preview(target_path: str = "worlds/canonical_world.npz"):
    p = Path(target_path)
    if p.is_dir():
        p = p / "canonical_world.npz"
    if not p.exists():
        p = Path("worlds/canonical_world.npz")
    if not p.exists():
        p = Path("world_data/canonical/canonical_world.npz")
    if not p.exists():
        raise FileNotFoundError(f"Could not find canonical world dataset at {p}. Run generate_world.py first.")

    print(f"[RLLS 16] Launching 2D Canonical Earth Preview for: {p.resolve()}")
    world_data = load_world(str(p))

    pygame.init()
    width, height = 1280, 768
    window = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption(f"RLLS 16 — 2D Canonical Earth Preview [{p.name}]")
    clock = pygame.time.Clock()

    layers = MapLayers(world_data)
    chunk_mgr = ChunkManager(layers, chunk_size=32)
    camera = Camera2D(width, height)
    renderer = Renderer2D(width, height, chunk_mgr, camera)

    print("\n[2D Watcher Controls]")
    print("  * Middle / Right Mouse Drag : Pan camera across Earth")
    print("  * Mouse Wheel              : Continuous zoom pinned around cursor")
    print("  * W / A / S / D            : Pan camera (Hold Shift to accelerate)")
    print("  * L                        : Cycle visual layer modes (Natural, Elevation, Temp, Precip, Biomes, Water)")
    print("  * G                        : Toggle Lat/Lon grid")
    print("  * C                        : Toggle animated weather cloud drift")
    print("  * I                        : Toggle 16-bit ecological icons")
    print("  * R                        : Reset view to entire Earth")
    print("  * ESC / Q                  : Exit preview\n")

    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = min(0.05, now - last_time)
        last_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(400, event.w), max(300, event.h)
                window = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.RESIZABLE)
                camera.resize(width, height)
                renderer.resize(width, height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (2, 3):
                    camera.handle_mouse_down(event.button, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                camera.handle_mouse_up(event.button)
            elif event.type == pygame.MOUSEMOTION:
                camera.handle_mouse_motion(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                camera.zoom_around_cursor(event.y, mx, my)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_l:
                    modes = ALL_LAYER_MODES
                    cur_idx = modes.index(renderer.current_layer_mode) if renderer.current_layer_mode in modes else 0
                    nxt = modes[(cur_idx + 1) % len(modes)]
                    renderer.set_layer_mode(nxt)
                    print(f"Layer: {nxt}")
                elif event.key == pygame.K_g:
                    renderer.toggle_grid()
                elif event.key == pygame.K_c:
                    renderer.toggle_weather()
                elif event.key == pygame.K_i:
                    renderer.toggle_icons()
                elif event.key == pygame.K_r:
                    camera.fly_to(0.50, 0.25, target_zoom=1.0, duration=0.8)

        key_state = pygame.key.get_pressed()
        camera.handle_keyboard(key_state, dt)
        camera.update(dt)

        renderer.render(window, agents=None, dt=dt)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "worlds/canonical_world.npz"
    preview(t)
