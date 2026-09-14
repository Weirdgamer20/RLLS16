"""
RLLS 16 — Canonical 2D Earth Artificial-World Simulation Desktop Application.
Coordinates 2D Watcher Viewport (120 FPS CPU target), multi-rate simulation loop,
and scientific workstation HUD.
"""

import sys
import math
import time
from pathlib import Path
import pygame
import numpy as np

from .storage import load_world
from .map.layers import MapLayers
from .map.chunk_tile import ChunkManager
from .graphics.camera_2d import Camera2D
from .graphics.renderer_2d import Renderer2D
from .simulation.world_instance import WorldInstance
from .ui.home import HomeScreen
from .ui.hud import SimulationHUD


class RLLS16App:
    """
    RLLS 16 — 2D Earth Observation & Life Simulation Desktop Application.
    Observes Earth from above as a watcher.
    """

    def __init__(self, canonical_path: str = "worlds/canonical_world.npz"):
        self.canonical_path = canonical_path
        self.width = 1280
        self.height = 768

        # Initialize Pygame Display
        pygame.init()
        pygame.display.set_caption("RLLS 16 — Canonical 2D Earth Simulation")

        self.window = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        # Load Canonical World (Immutable Baseline)
        p = Path(canonical_path)
        if not p.exists():
            p = Path("worlds/canonical_world.npz")
        if not p.exists():
            p = Path("world_data/canonical/canonical_world.npz")
        if not p.exists():
            raise FileNotFoundError(f"Canonical world not found at '{canonical_path}'. Please run world_acquisition/generate_canonical_world.py first.")

        self.canonical_path = str(p)
        self.canonical_world = load_world(self.canonical_path)

        # Map Layers & Spatial Chunk Cache
        self.layers = MapLayers(self.canonical_world)
        self.chunk_manager = ChunkManager(self.layers, chunk_size=32, max_cached=512)

        # 2D Orthographic Watcher Camera & 2D Renderer
        self.camera = Camera2D(self.width, self.height)
        self.renderer = Renderer2D(self.width, self.height, self.chunk_manager, self.camera)

        # UI Subsystems
        self.home_screen = HomeScreen(self.width, self.height)
        self.hud = SimulationHUD(self.width, self.height)

        # State Machine: 'HOME' or 'SIMULATION'
        self.app_state = "HOME"
        self.world_instance: WorldInstance | None = None

        # Running flag
        self.is_running = True

    def create_new_world(self, name: str, population: int, environment: str, rl_config: dict | None = None):
        self.world_instance = WorldInstance(
            world_name=name,
            canonical_path=self.canonical_path,
            human_population=population,
            initial_environment=environment,
            world_data=self.canonical_world,
        )
        if rl_config and hasattr(self.world_instance, 'rl_interface'):
            if "goal" in rl_config:
                self.world_instance.rl_interface.goal_type = rl_config["goal"]

        self.app_state = "SIMULATION"

        # Fly to deployed settlement coordinate
        lat = self.world_instance.human.latitude_deg
        lon = self.world_instance.human.longitude_deg
        self.camera.fly_to_latlon(lat, lon, target_zoom=35.0, duration=1.2)
        self.hud.show_message(f"World Created: {name} (Habitat: {environment})")

    def load_existing_world(self, path: str):
        self.world_instance = WorldInstance.load(path, world_data=self.canonical_world)
        self.app_state = "SIMULATION"

        lat = self.world_instance.human.latitude_deg
        lon = self.world_instance.human.longitude_deg
        self.camera.fly_to_latlon(lat, lon, target_zoom=35.0, duration=1.2)
        self.hud.show_message(f"World Loaded: {self.world_instance.world_name}")

    def handle_events(self, dt: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = max(400, event.w), max(300, event.h)
                self.window = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.RESIZABLE)
                self.camera.resize(self.width, self.height)
                self.renderer.resize(self.width, self.height)
                self.home_screen.resize(self.width, self.height)
                self.hud.resize(self.width, self.height)

            if self.app_state == "HOME":
                action = self.home_screen.handle_event(event)
                if action:
                    if action["action"] == "CREATE":
                        self.create_new_world(
                            action["world_name"],
                            action["human_population"],
                            action["environment"],
                            action.get("rl_config")
                        )
                    elif action["action"] == "LOAD":
                        self.load_existing_world(action["path"])
                    elif action["action"] == "EXIT":
                        self.is_running = False

            elif self.app_state == "SIMULATION":
                mx, my = pygame.mouse.get_pos()
                is_ui_event = self.hud.is_mouse_over_ui(mx, my)

                # UI Events
                if is_ui_event or event.type == pygame.MOUSEBUTTONUP:
                    action = self.hud.handle_event(event, self.world_instance, self.camera, self.renderer)
                    if action:
                        if action.get("action") == "RESET_WORLD":
                            self.world_instance.reset(self.canonical_world)
                            self.hud.show_message("World State Reset to Initial Configuration")
                        elif action.get("action") == "RETURN_HOME":
                            self.app_state = "HOME"

                # 2D Viewport Mouse Navigation Events (Only when not interacting with UI)
                if not is_ui_event:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            # Left click: select agent or pick coordinate
                            wx, wy = self.camera.screen_to_world(mx, my)
                            selected_id = None
                            if self.world_instance and hasattr(self.world_instance, "agents"):
                                for ag in self.world_instance.agents:
                                    asx, asy = self.camera.world_to_screen(ag.wx, ag.wy)
                                    if math.hypot(mx - asx, my - asy) < 18:
                                        selected_id = ag.id
                                        self.hud.show_message(f"Selected: {ag.name} (Health: {ag.health:.0f}%)")
                                        break
                            self.renderer.selected_agent_id = selected_id
                        elif event.button in (2, 3):
                            # Middle / Right click: pan
                            self.camera.handle_mouse_down(event.button, (mx, my))

                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.camera.handle_mouse_up(event.button)

                    elif event.type == pygame.MOUSEMOTION:
                        self.camera.handle_mouse_motion((mx, my))

                    elif event.type == pygame.MOUSEWHEEL:
                        # Cursor-anchored continuous zoom
                        self.camera.zoom_around_cursor(event.y, mx, my)

                # Keyboard Controls
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.world_instance:
                            self.world_instance.is_paused = not self.world_instance.is_paused
                    elif event.key == pygame.K_ESCAPE:
                        if self.hud.show_info_overlay:
                            self.hud.show_info_overlay = False
                        else:
                            self.camera.fly_to(0.50, 0.25, target_zoom=1.0, duration=0.8)

        # Continuous keyboard navigation (WASD / Arrows)
        if self.app_state == "SIMULATION":
            key_state = pygame.key.get_pressed()
            self.camera.handle_keyboard(key_state, dt)

    def run(self):
        """Main loop with decoupled simulation steps and 120 FPS target render loop."""
        last_time = time.time()

        while self.is_running:
            now = time.time()
            dt = min(0.05, max(0.001, now - last_time))
            last_time = now

            # 1. User & Window Events
            self.handle_events(dt)

            # 2. Simulation Step
            if self.app_state == "HOME":
                loading_result = self.home_screen.check_loading_finished()
                if loading_result and loading_result["action"] == "CREATE":
                    self.create_new_world(
                        loading_result["world_name"],
                        loading_result["human_population"],
                        loading_result["environment"],
                        loading_result.get("rl_config")
                    )

            if self.world_instance is not None:
                self.world_instance.update(dt)

            self.camera.update(dt)
            self.hud.update(dt, world_instance=self.world_instance)

            # 3. Render 2D Frame
            if self.app_state == "HOME":
                self.window.fill((8, 12, 18))
                # Render background macro map
                self.renderer.render(self.window, agents=None, dt=dt)
                self.home_screen.render(self.window)

            elif self.app_state == "SIMULATION":
                agents = self.world_instance.get_all_agents() if self.world_instance else []
                self.renderer.render(self.window, agents=agents, dt=dt)
                fps = self.clock.get_fps()
                self.hud.render(self.window, self.world_instance, self.camera, fps, renderer=self.renderer)

            # 4. Flip Buffers & Target 120 FPS
            pygame.display.flip()
            self.clock.tick(120)

        pygame.quit()


def launch(canonical_path: str = "worlds/canonical_world.npz"):
    app = RLLS16App(canonical_path)
    app.run()


if __name__ == "__main__":
    launch()
