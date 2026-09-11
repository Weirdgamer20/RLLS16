import sys
import math
import time
from pathlib import Path
import pygame
import moderngl
import numpy as np

from .storage import load_world
from .graphics.math3d import screen_to_ray, ray_sphere_intersect, unproject_terrain_hit
from .graphics.camera import OrbitCamera
from .graphics.renderer import SceneRenderer
from .graphics.gpu_config import apply_gpu_environment_hints, configure_pygame_gl_attributes
from .simulation.astronomy import compute_astronomical_state, SUN_POSITION
from .simulation.world_instance import WorldInstance
from .ui.home import HomeScreen
from .ui.hud import SimulationHUD


class RLLS16App:
    """
    RLLS 16 — 3D Artificial-World Simulation Desktop Application.
    Coordinates ModernGL 3D hardware viewport, simulation loop,
    and futuristic workstation UI.
    """

    def __init__(self, canonical_path: str = "worlds/canonical_world.npz"):
        self.canonical_path = canonical_path
        self.width = 1280
        self.height = 768

        # 1. Hardware Optimization & OpenGL Attributes (Optimus / High-Perf GPU)
        apply_gpu_environment_hints()
        pygame.init()
        pygame.display.set_caption("RLLS 16 — Artificial World Simulation")
        configure_pygame_gl_attributes()

        self.window = pygame.display.set_mode(
            (self.width, self.height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        # 2. Initialize ModernGL Context
        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, self.width, self.height)

        # 3. Load Canonical World (Immutable Baseline)
        if not Path(canonical_path).exists():
            raise FileNotFoundError(f"Canonical world not found at '{canonical_path}'. Please run generate_world.py first.")
        self.canonical_world = load_world(canonical_path)

        # 4. Initialize Subsystems
        self.renderer = SceneRenderer(self.ctx, self.width, self.height, self.canonical_world)
        self.camera = OrbitCamera(aspect=self.width / self.height)
        self.camera.focus_object("SOLAR", np.array([0, 0, 0], dtype=np.float32), distance=190.0)

        # UI Subsystems
        self.home_screen = HomeScreen(self.width, self.height)
        self.hud = SimulationHUD(self.width, self.height)
        self.hud.set_gpu_info(self.renderer.gpu_info)
        self.ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # State Machine: 'HOME' or 'SIMULATION'
        self.app_state = "HOME"
        self.world_instance: WorldInstance | None = None

        # Double click timing for 3D body focusing
        self.last_click_time = 0.0
        self.last_click_pos = (0, 0)

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
            if "goal" in rl_config: self.world_instance.rl_interface.goal_type = rl_config["goal"]

        self.app_state = "SIMULATION"
        astro_state = compute_astronomical_state(self.world_instance.sim_time_sec)
        # Point 1: Immediately snap camera focus to Earth
        self.camera.focus_object("EARTH", astro_state["earth_pos"], distance=22.0, snap=True)
        self.hud.show_message(f"World Created: {name} (Habitat: {environment})")

    def load_existing_world(self, path: str):
        self.world_instance = WorldInstance.load(path, world_data=self.canonical_world)
        self.app_state = "SIMULATION"
        astro_state = compute_astronomical_state(self.world_instance.sim_time_sec)
        # Point 1: Immediately snap camera focus to Earth
        self.camera.focus_object("EARTH", astro_state["earth_pos"], distance=22.0, snap=True)
        self.hud.show_message(f"World Loaded: {self.world_instance.world_name}")

    def handle_events(self, dt: float):
        astro_state = compute_astronomical_state(self.world_instance.sim_time_sec if self.world_instance else 0.0)
        earth_pos = astro_state["earth_pos"]
        moon_pos = astro_state["moon_pos"]
        sun_pos = SUN_POSITION

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = max(400, event.w), max(300, event.h)
                self.renderer.resize(self.width, self.height)
                self.camera.set_aspect(self.width / self.height)
                self.home_screen.resize(self.width, self.height)
                self.hud.resize(self.width, self.height)
                self.ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

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

                # Keep camera informed of astronomical planetary state
                if astro_state:
                    self.camera.earth_center = earth_pos
                    self.camera.earth_rot_angle = astro_state.earth_rot_rad
                    self.camera.axial_tilt = astro_state.axial_tilt_rad

                # UI Events
                if is_ui_event or event.type == pygame.MOUSEBUTTONUP:
                    action = self.hud.handle_event(event, self.world_instance, self.camera, self.renderer, astro_state=astro_state)
                    if action:
                        if action.get("action") == "RESET_WORLD":
                            self.world_instance.reset(self.canonical_world)
                            self.hud.show_message("World State Reset to Initial Configuration")
                        elif action.get("action") == "RETURN_HOME":
                            self.app_state = "HOME"

                # 3D Viewport Mouse Navigation Events
                if not is_ui_event:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        now = time.time()
                        # Detect Double Click
                        if now - self.last_click_time < 0.28 and math.hypot(mx - self.last_click_pos[0], my - self.last_click_pos[1]) < 12:
                            self._handle_double_click_focus(mx, my, earth_pos, moon_pos, sun_pos, astro_state)
                        else:
                            self.camera.handle_mouse_down(event.button, (mx, my))
                        self.last_click_time = now
                        self.last_click_pos = (mx, my)

                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.camera.handle_mouse_up(event.button)

                    elif event.type == pygame.MOUSEMOTION:
                        self.camera.handle_mouse_motion((mx, my))

                    elif event.type == pygame.MOUSEWHEEL:
                        precise_y = getattr(event, 'precise_y', float(event.y))
                        self.camera.handle_mouse_wheel(precise_y, mouse_pos=(mx, my), width=self.width, height=self.height)

                # Keyboard Controls
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.world_instance.is_paused = not self.world_instance.is_paused
                    elif event.key == pygame.K_f:
                        # Focus Earth or active target
                        self.camera.focus_object("EARTH", earth_pos, distance=28.0)
                    elif event.key == pygame.K_ESCAPE:
                        if self.hud.show_info_overlay:
                            self.hud.show_info_overlay = False
                        else:
                            self.camera.focus_object("SOLAR", np.array([0, 0, 0], dtype=np.float32), distance=190.0)

        # Continuous keyboard navigation (W/A/S/D/Q/E/+/-)
        if self.app_state == "SIMULATION":
            key_state = pygame.key.get_pressed()
            self.camera.handle_keyboard(key_state, dt)

    def _handle_double_click_focus(self, mx: int, my: int, earth_pos: np.ndarray, moon_pos: np.ndarray, sun_pos: np.ndarray, astro_state=None):
        """Ray cast from screen coordinates to detect double-click on Earth, Moon, or Sun."""
        view_mat = self.camera.get_view_matrix()
        proj_mat = self.camera.get_projection_matrix()

        # Check pinpoint Earth surface hit first for geographic settlement inspection
        rot_rad = astro_state.earth_rot_rad if astro_state else 0.0
        tilt_rad = astro_state.axial_tilt_rad if astro_state else math.radians(23.44)
        earth_hit = unproject_terrain_hit(
            mx, my, self.width, self.height, view_mat, proj_mat,
            earth_center=earth_pos, earth_radius=5.0,
            world_data=self.canonical_world,
            earth_rot_rad=rot_rad,
            axial_tilt_rad=tilt_rad
        )
        if earth_hit is not None:
            hit_world, lat, lon = earth_hit
            lat_deg = math.degrees(lat)
            lon_deg = math.degrees(lon)
            self.camera.fly_to_lat_lon(lat_deg, lon_deg, earth_pos=earth_pos, earth_rot_rad=rot_rad, altitude=1.2, duration=2.0)
            self.hud.show_message(f"Focused Earth: {abs(lat_deg):.1f}°{'N' if lat_deg >= 0 else 'S'}, {abs(lon_deg):.1f}°{'E' if lon_deg >= 0 else 'W'}")
            return

        ray_origin, ray_dir = screen_to_ray(mx, my, self.width, self.height, view_mat, proj_mat)

        hit_sun = ray_sphere_intersect(ray_origin, ray_dir, sun_pos, 13.0)
        hit_earth = ray_sphere_intersect(ray_origin, ray_dir, earth_pos, 6.0)
        hit_moon = ray_sphere_intersect(ray_origin, ray_dir, moon_pos, 2.5)

        hits = []
        if hit_sun is not None: hits.append((hit_sun, "SUN", sun_pos, 65.0))
        if hit_earth is not None: hits.append((hit_earth, "EARTH", earth_pos, 28.0))
        if hit_moon is not None: hits.append((hit_moon, "MOON", moon_pos, 12.0))

        if hits:
            hits.sort(key=lambda x: x[0])
            _, name, pos, dist = hits[0]
            self.camera.focus_object(name, pos, distance=dist)
            self.hud.show_message(f"Focused: {name}")

    def run(self):
        """Main simulation execution loop separating rendering and simulation ticks."""
        last_time = time.time()

        while self.is_running:
            now = time.time()
            dt = min(0.1, now - last_time)
            last_time = now

            # 1. Process Window & User Input
            self.handle_events(dt)

            # 2. Simulation Step (Fixed/Variable Timestep)
            sim_time_sec = 0.0
            solar_irradiance = 1.0
            if self.app_state == "HOME":
                # Check if multi-stage loading progress completed (Point 38)
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
                sim_time_sec = self.world_instance.sim_time_sec
                solar_irradiance = self.world_instance.environment.solar_irradiance

            astro_state = compute_astronomical_state(sim_time_sec)
            earth_pos = astro_state["earth_pos"]
            moon_pos = astro_state["moon_pos"]

            # Dynamic camera tracking if following a moving body
            target_pos = None
            if self.camera.focus_target_name == "EARTH":
                target_pos = earth_pos
            elif self.camera.focus_target_name == "MOON":
                target_pos = moon_pos
            elif self.camera.focus_target_name == "SUN" or self.camera.focus_target_name == "SOLAR":
                target_pos = SUN_POSITION

            self.camera.update(target_pos, dt)
            self.hud.update(dt, world_instance=self.world_instance)

            # 3. Clear 2D UI Surface
            self.ui_surface.fill((0, 0, 0, 0))

            # Render 2D UI into transparent surface
            if self.app_state == "HOME":
                self.home_screen.render(self.ui_surface)
            elif self.app_state == "SIMULATION":
                fps = self.clock.get_fps()
                self.hud.render(self.ui_surface, self.world_instance, astro_state, self.camera, fps, renderer=self.renderer)

            # 4. Render 3D Scene + Composite 2D HUD via ModernGL
            self.renderer.render(
                camera=self.camera,
                sim_time_sec=sim_time_sec,
                earth_pos=earth_pos,
                earth_rot_angle=astro_state["earth_rot_angle"],
                moon_pos=moon_pos,
                moon_rot_angle=astro_state["moon_rot_angle"],
                solar_irradiance=solar_irradiance,
                ui_surface=self.ui_surface,
            )

            # Swap buffers
            pygame.display.flip()
            self.clock.tick(120)  # Point 14: 120 FPS target cap

        pygame.quit()


def launch(canonical_path: str = "worlds/canonical_world.npz"):
    app = RLLS16App(canonical_path)
    app.run()
