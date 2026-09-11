"""
RLLS 16 — Futuristic Scientific Workstation Interface (HUD).
Features:
1. Top Time & Simulation Controls Bar (44x44 menu hitbox, exponential speeds 1x..100x)
2. Left Hierarchical Collapsible Controller Menu (200ms ease-out slide animation,
   strict viewport clipping, isolated wheel scrolling)
3. Right 3D Navigation Toolbar (Focus, Earth, Orbit, Grid, Info, Measure, Locate with fly-to)
4. Bottom High-Value Telemetry & Status Bar (Day, Temp, Humidity, Wind, Rain, Population, FPS, SIM)
5. Visual Debug Modes (Earth Solid, Wireframe, Height, Biome, Land, Ocean, Normals, LOD, Bounds)
6. Data & Analytics with 7 Microcharts / Sparklines (Temp, Pop, Rain, Biomass, Water, Reward, Health)
7. Restructured Life Panel (Individual Vitals vs Population Dynamics)
"""

import math
from collections import deque
from pathlib import Path
import pygame
import numpy as np

from .theme import (
    PANEL_BG, PANEL_BORDER, PANEL_BORDER_FOCUS,
    ACCENT_CYAN, ACCENT_BLUE, ACCENT_AMBER, ACCENT_GREEN,
    TEXT_PRIMARY, TEXT_MUTED, TEXT_DIM,
    BUTTON_BG, BUTTON_HOVER, BUTTON_ACTIVE, BUTTON_BORDER,
    get_fonts, draw_panel, draw_button, draw_gps_icon, draw_chevron
)
from ..simulation.astronomy import SUN_POSITION, compute_astronomical_state

DEBUG_MODE_NAMES = [
    (0, "Normal (PBR)"),
    (1, "Earth Solid"),
    (2, "Wireframe"),
    (3, "Height Elevation"),
    (4, "Biome Map"),
    (5, "Land Mask"),
    (6, "Ocean Mask"),
    (7, "Surface Normals"),
    (8, "Quadtree LOD"),
    (9, "Tile Bounds"),
]

SPEED_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
SPEED_LABELS = ["1×", "2×", "4×", "8×", "16×", "32×", "64×", "100×"]


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def draw_sparkline(
    surface: pygame.Surface,
    rect: pygame.Rect,
    data: list[float],
    color: tuple,
    label: str,
    current_str: str,
    unit: str = "",
    fonts: tuple | None = None
):
    """
    Renders a sleek scientific microchart sparkline with min, max, current value,
    and trend indicator.
    """
    if fonts is None:
        fonts = get_fonts()
    f_title, f_header, f_body, f_small = fonts

    # Background box
    pygame.draw.rect(surface, (12, 24, 42), rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 1)

    if not data or len(data) < 2:
        lbl_surf = f_small.render(f"{label}: {current_str}{unit}", True, TEXT_MUTED)
        surface.blit(lbl_surf, (rect.x + 6, rect.y + 4))
        return

    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val > min_val else 1.0

    # Trend calculation
    recent_delta = data[-1] - data[max(0, len(data) - 5)]
    if recent_delta > 0.01 * val_range:
        trend_icon = "▲"
        trend_col = ACCENT_GREEN
    elif recent_delta < -0.01 * val_range:
        trend_icon = "▼"
        trend_col = ACCENT_AMBER
    else:
        trend_icon = "─"
        trend_col = TEXT_MUTED

    # Header text
    hdr_txt = f"{label}: {current_str}{unit}  "
    hdr_surf = f_small.render(hdr_txt, True, TEXT_PRIMARY)
    surface.blit(hdr_surf, (rect.x + 6, rect.y + 3))

    trend_surf = f_small.render(trend_icon, True, trend_col)
    surface.blit(trend_surf, (rect.x + 6 + hdr_surf.get_width(), rect.y + 3))

    minmax_str = f"[{min_val:.1f} .. {max_val:.1f}]"
    minmax_surf = f_small.render(minmax_str, True, TEXT_DIM)
    surface.blit(minmax_surf, (rect.right - minmax_surf.get_width() - 6, rect.y + 3))

    # Graph plotting area
    gx = rect.x + 8
    gy = rect.y + 18
    gw = rect.width - 16
    gh = rect.height - 22

    pts = []
    n = len(data)
    for i, val in enumerate(data):
        norm_x = gx + int(i * (gw / max(1, n - 1)))
        norm_y = gy + gh - int(((val - min_val) / val_range) * (gh - 4)) - 2
        pts.append((norm_x, norm_y))

    if len(pts) >= 2:
        pygame.draw.lines(surface, color, False, pts, 1)
        # Glow dot at current value
        last_pt = pts[-1]
        pygame.draw.circle(surface, color, last_pt, 2)


class SimulationHUD:
    """
    RLLS 16 Scientific Workstation Interface.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fonts = get_fonts()

        # Left panel state & 200ms slide animation
        self.left_panel_width = 320
        self.left_panel_open = True
        self.panel_anim_t = 1.0       # 1.0 = fully open, 0.0 = fully closed
        self.panel_anim_duration = 0.20

        self.sections_expanded = {
            "WORLD": True,
            "WORLD / SPACE": True,
            "EARTH": True,
            "LIFE": True,
            "REINFORCEMENT LEARNING": False,
            "VISUALIZATION": True,
            "DATA & ANALYTICS": False,
        }
        self.scroll_y = 0
        self.total_content_height = 600

        # Microchart / Sparkline history buffers (Point 33)
        self.history_len = 60
        self.history = {
            "temperature": deque(maxlen=self.history_len),
            "population": deque(maxlen=self.history_len),
            "rainfall": deque(maxlen=self.history_len),
            "biomass": deque(maxlen=self.history_len),
            "water": deque(maxlen=self.history_len),
            "reward": deque(maxlen=self.history_len),
            "health": deque(maxlen=self.history_len),
        }
        self.sample_timer = 0.0

        # Right toolbar active states
        self.show_info_overlay = False
        self.measure_mode = False

        # Slider interaction tracking
        self.active_slider = None

        # GPU hardware telemetry
        self.gpu_info = None

        # Notification banner
        self.notification = ""
        self.notification_timer = 0.0

    def set_gpu_info(self, gpu_info: dict):
        self.gpu_info = gpu_info

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height

    def show_message(self, text: str, duration: float = 3.0):
        self.notification = text
        self.notification_timer = duration

    def _get_panel_x(self) -> int:
        """Returns the current animated X offset of the left panel (ease-out)."""
        offset_ratio = 1.0 - ease_out_cubic(self.panel_anim_t)
        return int(-self.left_panel_width * offset_ratio)

    def is_mouse_over_ui(self, mx: int, my: int) -> bool:
        """Determines if the mouse is over 2D UI elements to prevent raycast pass-through."""
        # Top Bar (height 48)
        if my < 48:
            return True
        # Bottom Status Bar (height 28)
        if my > self.height - 28:
            return True
        # Left Panel (when animating or open)
        panel_x = self._get_panel_x()
        if self.panel_anim_t > 0.01 and mx < panel_x + self.left_panel_width:
            return True
        # Right Toolbar
        if mx > self.width - 70 and 48 <= my <= self.height - 28:
            return True
        # Info overlay card
        if self.show_info_overlay and mx > self.width - 330 and 56 <= my <= 320:
            return True
        return False

    def handle_event(self, event, world_instance, camera, renderer, astro_state: dict | None = None) -> dict | None:
        mx, my = getattr(event, 'pos', pygame.mouse.get_pos())

        if astro_state is None:
            sim_time = world_instance.sim_time_sec if world_instance else 0.0
            astro_state = compute_astronomical_state(sim_time)

        if event.type == pygame.MOUSEBUTTONDOWN:
            # 1. Top Bar clicks (height 48)
            if my < 48:
                return self._handle_top_bar_click(mx, my, world_instance)

            # 2. Right Toolbar clicks
            if mx > self.width - 70 and 48 <= my <= self.height - 28:
                return self._handle_right_toolbar_click(mx, my, world_instance, camera, renderer, astro_state)

            # 3. Left Panel clicks (strictly inside viewport between 48 and height-28)
            panel_x = self._get_panel_x()
            if self.panel_anim_t > 0.01 and mx < panel_x + self.left_panel_width and (48 <= my <= self.height - 28):
                return self._handle_left_panel_click(mx - panel_x, my, world_instance, renderer)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.active_slider = None

        elif event.type == pygame.MOUSEMOTION:
            if self.active_slider is not None:
                panel_x = self._get_panel_x()
                self._update_slider(mx - panel_x, world_instance)

        elif event.type == pygame.MOUSEWHEEL:
            panel_x = self._get_panel_x()
            # Strict event priority: isolate mouse-wheel inside left panel (Point 34, 35)
            if self.panel_anim_t > 0.01 and mx < panel_x + self.left_panel_width:
                viewport_h = self.height - 48 - 28
                min_scroll = min(0, viewport_h - self.total_content_height - 20)
                self.scroll_y = max(min_scroll, min(0, self.scroll_y + event.y * 32))
                return {"consumed": True}

        return None

    def _handle_top_bar_click(self, mx: int, my: int, inst) -> dict | None:
        # Point 40: Menu button with minimum 44x44px hitbox
        btn_toggle = pygame.Rect(4, 2, 44, 44)
        if btn_toggle.collidepoint(mx, my):
            self.left_panel_open = not self.left_panel_open
            return None

        # Play / Pause button
        time_ctrl_x = self.width - 390
        btn_pp = pygame.Rect(time_ctrl_x, 8, 32, 32)
        if btn_pp.collidepoint(mx, my):
            inst.is_paused = not inst.is_paused
            return None

        # Point 16: Exponential speed buttons [1x, 2x, 4x, 8x, 16x, 32x, 64x, 100x]
        for i, (spd, lbl) in enumerate(zip(SPEED_VALUES, SPEED_LABELS)):
            rect = pygame.Rect(time_ctrl_x + 40 + i * 42, 8, 38, 32)
            if rect.collidepoint(mx, my):
                inst.time_speed = spd
                inst.is_paused = False
                return None

        return None

    def _get_right_toolbar_rects(self) -> list[tuple[str, pygame.Rect]]:
        bx = self.width - 64
        btn_y = 56
        btn_h = 44
        gap = 8
        tools = ["FOCUS", "EARTH", "ORBIT", "GRID", "INFO", "MEASURE", "LOCATE"]
        return [(tool, pygame.Rect(bx, btn_y + i * (btn_h + gap), 54, btn_h)) for i, tool in enumerate(tools)]

    def _handle_right_toolbar_click(self, mx: int, my: int, inst, camera, renderer, astro_state: dict) -> dict | None:
        earth_pos = astro_state["earth_pos"]
        moon_pos = astro_state["moon_pos"]
        sun_pos = SUN_POSITION

        for tool, rect in self._get_right_toolbar_rects():
            if rect.collidepoint(mx, my):
                if tool == "FOCUS":
                    order = ["SOLAR", "EARTH", "MOON", "SUN"]
                    curr_idx = order.index(camera.focus_target_name) if camera.focus_target_name in order else 0
                    next_target = order[(curr_idx + 1) % len(order)]
                    if next_target == "EARTH":
                        camera.focus_object("EARTH", earth_pos, distance=24.0)
                    elif next_target == "MOON":
                        camera.focus_object("MOON", moon_pos, distance=8.0)
                    elif next_target == "SUN":
                        camera.focus_object("SUN", sun_pos, distance=65.0)
                    else:
                        camera.focus_object("SOLAR", np.array([0, 0, 0], dtype=np.float64), distance=190.0)
                    self.show_message(f"Focus Target: {next_target}")

                elif tool == "EARTH":
                    camera.focus_object("EARTH", earth_pos, distance=24.0)
                    self.show_message("Focus Earth")

                elif tool == "ORBIT":
                    renderer.show_orbits = not renderer.show_orbits
                    self.show_message(f"Orbital Paths: {'ENABLED' if renderer.show_orbits else 'DISABLED'}")

                elif tool == "GRID":
                    renderer.show_grid = not renderer.show_grid
                    self.show_message(f"Spatial Grid: {'ENABLED' if renderer.show_grid else 'DISABLED'}")

                elif tool == "INFO":
                    self.show_info_overlay = not self.show_info_overlay

                elif tool == "MEASURE":
                    self.measure_mode = not self.measure_mode
                    self.show_message(f"Measure Tool: {'ACTIVE' if self.measure_mode else 'OFF'}")

                elif tool == "LOCATE":
                    # Point 53: Locate deployed settlement smoothly via fly_to animation
                    camera.focus_target_name = "EARTH"
                    camera.target = earth_pos.copy()
                    camera.fly_to_lat_lon(inst.human.latitude_deg, inst.human.longitude_deg, altitude=8.5, duration=2.0)
                    self.show_message(f"LOCATED: Settlement ({inst.human.latitude_deg:.1f}°N, {inst.human.longitude_deg:.1f}°E, {inst.human.environment_type})")

                return None
        return None

    def _handle_left_panel_click(self, panel_mx: int, my: int, inst, renderer) -> dict | None:
        content_y = 56 + self.scroll_y

        for sec_name, is_exp in self.sections_expanded.items():
            hdr_rect = pygame.Rect(8, content_y, self.left_panel_width - 20, 26)
            if hdr_rect.collidepoint(panel_mx, my):
                self.sections_expanded[sec_name] = not is_exp
                return None

            content_y += 28

            if is_exp:
                if sec_name == "WORLD":
                    b_save = pygame.Rect(16, content_y + 18, 80, 24)
                    b_reset = pygame.Rect(102, content_y + 18, 80, 24)
                    b_home = pygame.Rect(188, content_y + 18, 80, 24)

                    if b_save.collidepoint(panel_mx, my):
                        path = inst.save()
                        self.show_message(f"World Saved: {path}")
                        return None
                    elif b_reset.collidepoint(panel_mx, my):
                        return {"action": "RESET_WORLD"}
                    elif b_home.collidepoint(panel_mx, my):
                        return {"action": "RETURN_HOME"}
                    content_y += 48

                elif sec_name == "WORLD / SPACE":
                    content_y += 96

                elif sec_name == "EARTH":
                    slider_names = ["solar", "temp", "clouds", "co2"]
                    for idx, s_name in enumerate(slider_names):
                        track_rect = pygame.Rect(105, content_y + idx * 22 + 4, 110, 10)
                        if track_rect.collidepoint(panel_mx, my):
                            self.active_slider = s_name
                            self._update_slider(panel_mx, inst)
                            return None
                    content_y += 96

                elif sec_name == "LIFE":
                    content_y += 184

                elif sec_name == "REINFORCEMENT LEARNING":
                    goal_btn = pygame.Rect(140, content_y, 120, 22)
                    if goal_btn.collidepoint(panel_mx, my):
                        goals = ["Survive", "Growth", "Exploration", "Culture"]
                        curr = inst.rl_interface.goal_type
                        next_goal = goals[(goals.index(curr) + 1) % len(goals)]
                        inst.rl_interface.goal_type = next_goal
                        self.show_message(f"RL Active Goal: {next_goal}")
                        return None
                    content_y += 86

                elif sec_name == "VISUALIZATION":
                    # Point 44: Debug view cycle buttons
                    prev_btn = pygame.Rect(16, content_y, 24, 20)
                    next_btn = pygame.Rect(self.left_panel_width - 40, content_y, 24, 20)
                    mid_btn = pygame.Rect(44, content_y, self.left_panel_width - 88, 20)

                    if prev_btn.collidepoint(panel_mx, my) or mid_btn.collidepoint(panel_mx, my):
                        if renderer and hasattr(renderer, "earth_renderer"):
                            cur = renderer.earth_renderer.debug_mode
                            nxt = (cur - 1) % len(DEBUG_MODE_NAMES)
                            renderer.earth_renderer.debug_mode = nxt
                            self.show_message(f"Debug View: {DEBUG_MODE_NAMES[nxt][1]}")
                        return None
                    elif next_btn.collidepoint(panel_mx, my):
                        if renderer and hasattr(renderer, "earth_renderer"):
                            cur = renderer.earth_renderer.debug_mode
                            nxt = (cur + 1) % len(DEBUG_MODE_NAMES)
                            renderer.earth_renderer.debug_mode = nxt
                            self.show_message(f"Debug View: {DEBUG_MODE_NAMES[nxt][1]}")
                        return None

                    content_y += 26

                    toggles = ["Atmosphere", "Clouds", "Orbits", "Grid"]
                    for idx, name in enumerate(toggles):
                        chk_rect = pygame.Rect(16, content_y + idx * 22, self.left_panel_width - 32, 20)
                        if chk_rect.collidepoint(panel_mx, my):
                            if name == "Atmosphere": renderer.show_atmo = not renderer.show_atmo
                            elif name == "Clouds": renderer.show_clouds = not renderer.show_clouds
                            elif name == "Orbits": renderer.show_orbits = not renderer.show_orbits
                            elif name == "Grid": renderer.show_grid = not renderer.show_grid
                            return None
                    content_y += 94

                elif sec_name == "DATA & ANALYTICS":
                    content_y += 240

        return None

    def _update_slider(self, panel_mx: int, inst):
        track_x = 105
        track_w = 110
        ratio = max(0.0, min(1.0, (panel_mx - track_x) / track_w))

        if self.active_slider == "solar":
            inst.environment.solar_irradiance = 0.5 + ratio * 1.0
        elif self.active_slider == "temp":
            inst.environment.global_temp_offset = -10.0 + ratio * 20.0
        elif self.active_slider == "clouds":
            inst.environment.cloud_cover_factor = ratio * 2.0
        elif self.active_slider == "co2":
            inst.environment.co2_ppm = 280.0 + ratio * 600.0

    def update(self, dt: float, world_instance=None):
        # 1. Slide animation update (180..220ms ease-out)
        target_t = 1.0 if self.left_panel_open else 0.0
        if self.panel_anim_t != target_t:
            step = dt / self.panel_anim_duration
            if self.panel_anim_t < target_t:
                self.panel_anim_t = min(target_t, self.panel_anim_t + step)
            else:
                self.panel_anim_t = max(target_t, self.panel_anim_t - step)

        # 2. Notification timer
        if self.notification_timer > 0:
            self.notification_timer -= dt
            if self.notification_timer <= 0:
                self.notification = ""

        # 3. Microchart history sampling (every 0.5 sec)
        self.sample_timer += dt
        if self.sample_timer >= 0.5 and world_instance is not None:
            self.sample_timer = 0.0
            obs = world_instance.get_current_observation()
            human = world_instance.human
            reward = world_instance.rl_interface.last_reward if hasattr(world_instance.rl_interface, "last_reward") else 0.0

            self.history["temperature"].append(obs.temperature_c)
            self.history["population"].append(float(human.population))
            self.history["rainfall"].append(obs.precipitation_rate_mm_h)
            self.history["biomass"].append(obs.vegetation_biomass * 100.0)
            self.history["water"].append(obs.soil_moisture * 100.0)
            self.history["reward"].append(float(reward))
            self.history["health"].append(float(human.vitals.health))

    def render(self, surface: pygame.Surface, world_instance, astronomy_data: dict, camera, fps: float, renderer=None):
        """
        Renders HUD with strict layer ordering:
        Layer 1: Left Animated Hierarchical Panel
        Layer 2: Right Navigation Toolbar
        Layer 3: Top Taskbar (Opaque solid overlay, never overlapped by scrolling)
        Layer 4: Bottom Telemetry Bar (Point 42 specification)
        Layer 5: Overlays & Notifications
        """
        f_title, f_header, f_body, f_small = self.fonts
        mx, my = pygame.mouse.get_pos()

        # -------------------------------------------------------------
        # 1. LEFT HIERARCHICAL CONTROL PANEL (Animated Slide)
        # -------------------------------------------------------------
        if self.panel_anim_t > 0.005:
            panel_x = self._get_panel_x()
            panel_y = 48
            panel_h = self.height - 48 - 28
            panel_rect = pygame.Rect(panel_x, panel_y, self.left_panel_width, panel_h)

            # Draw base panel background
            draw_panel(surface, panel_rect, border_color=PANEL_BORDER, fill_color=(10, 20, 36, 245))

            # Strictly clip all scrolling section content to the panel viewport
            surface.set_clip(panel_rect)
            self._render_left_panel_sections(
                surface, world_instance, astronomy_data,
                panel_x=panel_x, renderer=renderer, mx=mx, my=my
            )

            # Draw scrollbar if content exceeds viewport
            if self.total_content_height > panel_h:
                track_rect = pygame.Rect(panel_x + self.left_panel_width - 5, panel_y + 4, 3, panel_h - 8)
                pygame.draw.rect(surface, (20, 40, 65), track_rect)

                thumb_h = max(24, int((panel_h / max(1, self.total_content_height)) * (panel_h - 8)))
                min_scroll = min(0, panel_h - self.total_content_height - 20)
                scroll_ratio = abs(self.scroll_y) / abs(min_scroll) if min_scroll < 0 else 0.0
                thumb_y = panel_y + 4 + int(scroll_ratio * (panel_h - 8 - thumb_h))
                thumb_rect = pygame.Rect(panel_x + self.left_panel_width - 5, thumb_y, 3, thumb_h)
                pygame.draw.rect(surface, ACCENT_CYAN, thumb_rect)

            surface.set_clip(None)

        # -------------------------------------------------------------
        # 2. RIGHT NAVIGATION TOOLBAR
        # -------------------------------------------------------------
        self._render_right_toolbar(surface, camera, mx, my)

        # -------------------------------------------------------------
        # 3. TOP BAR (Opaque solid overlay, drawn on top)
        # -------------------------------------------------------------
        top_rect = pygame.Rect(0, 0, self.width, 48)
        draw_panel(surface, top_rect, border_color=PANEL_BORDER, fill_color=(6, 12, 24, 255))

        # Point 40: Menu button (hitbox >= 44x44px)
        btn_toggle = pygame.Rect(4, 2, 44, 44)
        is_hov_menu = btn_toggle.collidepoint(mx, my)
        pygame.draw.rect(surface, BUTTON_HOVER if is_hov_menu else BUTTON_BG, btn_toggle)
        pygame.draw.rect(surface, ACCENT_CYAN if is_hov_menu else BUTTON_BORDER, btn_toggle, 1)

        # 3 clean horizontal bars centered inside 44x44 hitbox
        cx = btn_toggle.centerx
        cy = btn_toggle.centery
        bar_col = ACCENT_CYAN if is_hov_menu else TEXT_PRIMARY
        for dy in [-6, 0, 6]:
            pygame.draw.line(surface, bar_col, (cx - 9, cy + dy), (cx + 9, cy + dy), 2)

        # World Title
        w_title = f_header.render(f"RLLS 16  |  {world_instance.world_name}", True, ACCENT_CYAN)
        surface.blit(w_title, (58, 14))

        # Simulation Time Readout
        year_str, day_str, time_str = world_instance.get_formatted_time()
        time_display = f"{year_str}  |  {day_str}  |  {time_str}"
        time_surf = f_body.render(time_display, True, TEXT_PRIMARY)
        surface.blit(time_surf, (self.width // 2 - 130, 16))

        # Point 16: Time controls (Play/Pause & 8 Exponential Speeds)
        time_ctrl_x = self.width - 390
        btn_pp = pygame.Rect(time_ctrl_x, 8, 32, 32)
        is_paused = world_instance.is_paused
        draw_button(surface, btn_pp, "⏸" if not is_paused else "▶", f_header, btn_pp.collidepoint(mx, my), is_active=not is_paused)

        for i, (spd, lbl) in enumerate(zip(SPEED_VALUES, SPEED_LABELS)):
            rect = pygame.Rect(time_ctrl_x + 40 + i * 42, 8, 38, 32)
            is_active = (world_instance.time_speed == spd and not is_paused)
            draw_button(surface, rect, lbl, f_small, rect.collidepoint(mx, my), is_active=is_active)

        # -------------------------------------------------------------
        # 4. BOTTOM STATUS BAR (Point 42 Specification)
        # -------------------------------------------------------------
        bot_rect = pygame.Rect(0, self.height - 28, self.width, 28)
        draw_panel(surface, bot_rect, border_color=PANEL_BORDER, fill_color=(6, 12, 24, 255))

        day_num = int(world_instance.sim_time_sec // 86400) + 1
        obs = world_instance.get_current_observation()
        human = world_instance.human
        rain_str = f"{obs.precipitation_rate_mm_h:.1f} mm/h"
        humidity_pct = min(99, max(12, int(obs.humidity * 3500)))
        sim_speed_str = f"{int(world_instance.time_speed)}×" if world_instance.time_speed >= 1.0 else f"{world_instance.time_speed:.1f}×"
        if world_instance.is_paused:
            sim_speed_str = "PAUSED"

        status_text = (
            f"DAY {day_num:03d}   |   "
            f"{obs.temperature_c:+.1f}°C   |   "
            f"Humidity {humidity_pct}%   |   "
            f"Wind {obs.wind_speed:.1f} m/s   |   "
            f"Rain {rain_str}   |   "
            f"Population {human.population:,}   |   "
            f"FPS {fps:.0f}   |   "
            f"SIM {sim_speed_str}"
        )
        surface.blit(f_small.render(status_text, True, TEXT_PRIMARY), (16, self.height - 20))

        # Dynamic GPU status tag
        gpu_label = "GL 3.3"
        is_ded = False
        if self.gpu_info and self.gpu_info.get("renderer"):
            raw_r = self.gpu_info["renderer"].split("/")[0].strip()
            is_ded = self.gpu_info.get("is_dedicated", False)
            if "3050" in raw_r: gpu_label = "RTX 3050"
            elif "GeForce" in raw_r or "NVIDIA" in raw_r: gpu_label = "NVIDIA GPU"
            elif "Radeon" in raw_r: gpu_label = "RADEON"
            elif "Intel" in raw_r: gpu_label = "Intel iGPU"
            else: gpu_label = raw_r[:12]

        gpu_tag = f"{gpu_label} [DEDICATED]" if is_ded else gpu_label
        gpu_surf = f_small.render(gpu_tag, True, ACCENT_GREEN if is_ded else ACCENT_CYAN)
        surface.blit(gpu_surf, (self.width - gpu_surf.get_width() - 16, self.height - 20))

        # -------------------------------------------------------------
        # 5. NOTIFICATION BANNER
        # -------------------------------------------------------------
        if self.notification:
            notif_surf = f_header.render(self.notification, True, ACCENT_CYAN)
            nw = notif_surf.get_width() + 32
            nr = pygame.Rect(self.width // 2 - nw // 2, 60, nw, 32)
            draw_panel(surface, nr, border_color=ACCENT_CYAN, fill_color=(10, 24, 44, 240))
            surface.blit(notif_surf, notif_surf.get_rect(center=nr.center))

        # -------------------------------------------------------------
        # 6. INFO OVERLAY CARD (if toggled)
        # -------------------------------------------------------------
        if self.show_info_overlay:
            self._render_info_card(surface, world_instance, astronomy_data, camera, fps=fps, renderer=renderer)

    def _render_left_panel_sections(
        self, surface: pygame.Surface, inst, astro: dict,
        panel_x: int, renderer, mx: int, my: int
    ):
        f_title, f_header, f_body, f_small = self.fonts
        start_y = 56 + self.scroll_y
        content_y = start_y
        pw = self.left_panel_width

        for sec_name, is_exp in self.sections_expanded.items():
            hdr_rect = pygame.Rect(panel_x + 8, content_y, pw - 20, 26)
            is_hov = hdr_rect.collidepoint(mx, my) and (48 <= my <= self.height - 28)
            pygame.draw.rect(surface, BUTTON_HOVER if is_hov else BUTTON_BG, hdr_rect)
            pygame.draw.rect(surface, ACCENT_CYAN if is_hov else PANEL_BORDER, hdr_rect, 1)

            draw_chevron(surface, panel_x + 16, content_y + 9, is_exp, color=ACCENT_CYAN if is_hov else TEXT_MUTED)
            surface.blit(f_header.render(sec_name, True, TEXT_PRIMARY if is_hov else TEXT_MUTED), (panel_x + 32, content_y + 5))

            content_y += 28

            if is_exp:
                if sec_name == "WORLD":
                    surface.blit(f_small.render(f"World: {inst.world_name}", True, TEXT_MUTED), (panel_x + 16, content_y + 2))
                    b_save = pygame.Rect(panel_x + 16, content_y + 18, 80, 24)
                    b_reset = pygame.Rect(panel_x + 102, content_y + 18, 80, 24)
                    b_home = pygame.Rect(panel_x + 188, content_y + 18, 80, 24)
                    draw_button(surface, b_save, "SAVE", f_small, b_save.collidepoint(mx, my))
                    draw_button(surface, b_reset, "RESET", f_small, b_reset.collidepoint(mx, my))
                    draw_button(surface, b_home, "MENU", f_small, b_home.collidepoint(mx, my))
                    content_y += 48

                elif sec_name == "WORLD / SPACE":
                    lines = [
                        ("Sun Luminosity", "3.828 × 10²⁶ W (Fixed)"),
                        ("Sun Radiation", "1,361 W/m² (Canonical)"),
                        ("Earth Orbit", f"365.25 d  [{astro['season']}]"),
                        ("Earth Obliquity", "23.44° (Axial Tilt)"),
                        ("Moon Orbit", "27.32 d (Tidally Locked)"),
                    ]
                    for label, val in lines:
                        surface.blit(f_small.render(label, True, TEXT_MUTED), (panel_x + 16, content_y))
                        surface.blit(f_small.render(val, True, TEXT_PRIMARY), (panel_x + 145, content_y))
                        content_y += 18
                    content_y += 6

                elif sec_name == "EARTH":
                    sliders = [
                        ("Solar Irrad.", f"{inst.environment.solar_irradiance:.2f}×", (inst.environment.solar_irradiance - 0.5) / 1.0),
                        ("Temp Offset", f"{inst.environment.global_temp_offset:+.1f}°C", (inst.environment.global_temp_offset + 10.0) / 20.0),
                        ("Cloud Cover", f"{inst.environment.cloud_cover_factor:.2f}×", inst.environment.cloud_cover_factor / 2.0),
                        ("CO2 Level", f"{inst.environment.co2_ppm:.0f} ppm", (inst.environment.co2_ppm - 280.0) / 600.0),
                    ]
                    for s_lbl, s_val_str, s_ratio in sliders:
                        surface.blit(f_small.render(s_lbl, True, TEXT_MUTED), (panel_x + 16, content_y))
                        surface.blit(f_small.render(s_val_str, True, ACCENT_CYAN), (panel_x + 225, content_y))

                        track_rect = pygame.Rect(panel_x + 105, content_y + 4, 110, 8)
                        pygame.draw.rect(surface, BUTTON_BG, track_rect)
                        pygame.draw.rect(surface, PANEL_BORDER, track_rect, 1)

                        fill_w = int(track_rect.width * max(0.0, min(1.0, s_ratio)))
                        if fill_w > 0:
                            pygame.draw.rect(surface, ACCENT_BLUE, (track_rect.x, track_rect.y, fill_w, track_rect.height))

                        handle_x = track_rect.x + fill_w
                        pygame.draw.circle(surface, ACCENT_CYAN, (handle_x, track_rect.centery), 5)
                        content_y += 22
                    content_y += 8

                elif sec_name == "LIFE":
                    # Point 29: Restructured Life panel (Population Dynamics vs Individual Physiology)
                    human = inst.human
                    obs = inst.get_current_observation()

                    surface.blit(f_small.render("POPULATION DYNAMICS", True, ACCENT_CYAN), (panel_x + 16, content_y))
                    content_y += 18
                    pop_lines = [
                        ("Population", f"{human.population:,}", ACCENT_GREEN),
                        ("Birth Rate", "+5.5% / yr", TEXT_PRIMARY),
                        ("Death Rate", "-3.6% / yr", TEXT_PRIMARY),
                        ("Food Security", f"{obs.vegetation_biomass * 100:.0f}%", ACCENT_GREEN if obs.vegetation_biomass > 0.3 else ACCENT_AMBER),
                        ("Shelter Level", f"{human.shelter_level * 100:.0f}%", TEXT_PRIMARY),
                    ]
                    for label, val, col in pop_lines:
                        surface.blit(f_small.render(label, True, TEXT_MUTED), (panel_x + 22, content_y))
                        surface.blit(f_small.render(val, True, col), (panel_x + 155, content_y))
                        content_y += 16

                    content_y += 6
                    surface.blit(f_small.render("INDIVIDUAL PHYSIOLOGY", True, ACCENT_CYAN), (panel_x + 16, content_y))
                    content_y += 18
                    vital_lines = [
                        ("Cohort Health", f"{human.vitals.health:.1f}%", ACCENT_GREEN if human.vitals.health > 70 else ACCENT_AMBER),
                        ("Hydration", f"{human.vitals.hydration_thirst:.1f}%", TEXT_PRIMARY),
                        ("Energy/Satiety", f"{human.vitals.energy_hunger:.1f}%", TEXT_PRIMARY),
                        ("Thermal Comfort", f"{human.vitals.thermal_comfort:.1f}%", TEXT_PRIMARY),
                    ]
                    for label, val, col in vital_lines:
                        surface.blit(f_small.render(label, True, TEXT_MUTED), (panel_x + 22, content_y))
                        surface.blit(f_small.render(val, True, col), (panel_x + 155, content_y))
                        content_y += 16
                    content_y += 8

                elif sec_name == "REINFORCEMENT LEARNING":
                    surface.blit(f_small.render("Active Goal", True, TEXT_MUTED), (panel_x + 16, content_y + 2))
                    goal_btn = pygame.Rect(panel_x + 140, content_y, 120, 22)
                    draw_button(surface, goal_btn, inst.rl_interface.goal_type.upper(), f_small, goal_btn.collidepoint(mx, my), is_active=True)
                    content_y += 26

                    mask = inst.rl_interface.get_action_mask()
                    valid_actions = int(np.sum(mask))

                    lines = [
                        ("Observation", "12-dim Normalized"),
                        ("Action Space", f"14 Discrete ({valid_actions} Valid)"),
                        ("Discoveries", f"{inst.human.knowledge_discoveries}"),
                    ]
                    for label, val in lines:
                        surface.blit(f_small.render(label, True, TEXT_MUTED), (panel_x + 16, content_y))
                        surface.blit(f_small.render(val, True, TEXT_PRIMARY), (panel_x + 145, content_y))
                        content_y += 18
                    content_y += 6

                elif sec_name == "VISUALIZATION":
                    # Point 44: Debug View Selector
                    cur_mode = renderer.earth_renderer.debug_mode if (renderer and hasattr(renderer, "earth_renderer")) else 0
                    mode_name = DEBUG_MODE_NAMES[cur_mode][1]

                    prev_btn = pygame.Rect(panel_x + 16, content_y, 22, 20)
                    next_btn = pygame.Rect(panel_x + pw - 38, content_y, 22, 20)
                    mid_btn = pygame.Rect(panel_x + 40, content_y, pw - 80, 20)

                    draw_button(surface, prev_btn, "◀", f_small, prev_btn.collidepoint(mx, my))
                    draw_button(surface, mid_btn, f"DEBUG: {mode_name[:14]}", f_small, mid_btn.collidepoint(mx, my), is_active=(cur_mode > 0))
                    draw_button(surface, next_btn, "▶", f_small, next_btn.collidepoint(mx, my))

                    content_y += 26

                    toggles = [
                        ("Atmosphere Layer", renderer.show_atmo if renderer else True),
                        ("Dynamic Clouds", renderer.show_clouds if renderer else True),
                        ("Orbital Trajectories", renderer.show_orbits if renderer else True),
                        ("Spatial Grid (Off)", renderer.show_grid if renderer else False),
                    ]
                    for idx, (label, val) in enumerate(toggles):
                        chk_rect = pygame.Rect(panel_x + 16, content_y + idx * 22, 14, 14)
                        pygame.draw.rect(surface, BUTTON_BG, chk_rect)
                        pygame.draw.rect(surface, ACCENT_CYAN if val else PANEL_BORDER, chk_rect, 1)

                        if val:
                            pygame.draw.line(surface, ACCENT_CYAN, (chk_rect.x + 2, chk_rect.y + 7), (chk_rect.x + 5, chk_rect.y + 11), 2)
                            pygame.draw.line(surface, ACCENT_CYAN, (chk_rect.x + 5, chk_rect.y + 11), (chk_rect.x + 11, chk_rect.y + 3), 2)

                        surface.blit(f_small.render(label, True, TEXT_PRIMARY if val else TEXT_MUTED), (panel_x + 36, content_y + idx * 22 + 1))
                    content_y += 94

                elif sec_name == "DATA & ANALYTICS":
                    # Points 32, 33: Telemetry & Sparkline Microcharts
                    surface.blit(f_small.render("ENVIRONMENTAL & LIFE TELEMETRY", True, ACCENT_CYAN), (panel_x + 16, content_y))
                    content_y += 18

                    charts = [
                        ("Temperature", list(self.history["temperature"]), ACCENT_AMBER, f"{inst.get_current_observation().temperature_c:.1f}", "°C"),
                        ("Population", list(self.history["population"]), ACCENT_GREEN, f"{inst.human.population:,}", ""),
                        ("Rainfall", list(self.history["rainfall"]), ACCENT_BLUE, f"{inst.get_current_observation().precipitation_rate_mm_h:.1f}", " mm/h"),
                        ("Biomass", list(self.history["biomass"]), ACCENT_GREEN, f"{inst.get_current_observation().vegetation_biomass * 100:.0f}", "%"),
                        ("Water / Soil", list(self.history["water"]), ACCENT_CYAN, f"{inst.get_current_observation().soil_moisture * 100:.0f}", "%"),
                        ("Health", list(self.history["health"]), ACCENT_GREEN, f"{inst.human.vitals.health:.0f}", "%"),
                    ]

                    for c_label, c_data, c_col, c_curr, c_unit in charts:
                        c_rect = pygame.Rect(panel_x + 16, content_y, pw - 32, 34)
                        draw_sparkline(
                            surface, c_rect, c_data, c_col,
                            label=c_label, current_str=c_curr, unit=c_unit,
                            fonts=self.fonts
                        )
                        content_y += 38

                    content_y += 6

        self.total_content_height = max(100, content_y - start_y)

    def _render_right_toolbar(self, surface: pygame.Surface, camera, mx: int, my: int):
        f_title, f_header, f_body, f_small = self.fonts

        for tool_id, rect in self._get_right_toolbar_rects():
            is_hov = rect.collidepoint(mx, my) and (48 <= my <= self.height - 28)

            pygame.draw.rect(surface, BUTTON_HOVER if is_hov else BUTTON_BG, rect)
            pygame.draw.rect(surface, ACCENT_CYAN if is_hov else PANEL_BORDER, rect, 1)

            if tool_id == "LOCATE":
                draw_gps_icon(surface, rect.centerx, rect.y + 15, size=15, color=ACCENT_CYAN if is_hov else ACCENT_AMBER)
                lbl_surf = f_small.render("LOCATE", True, TEXT_PRIMARY if is_hov else TEXT_MUTED)
                surface.blit(lbl_surf, lbl_surf.get_rect(center=(rect.centerx, rect.y + 33)))
            else:
                lbl_surf = f_small.render(tool_id, True, TEXT_PRIMARY if is_hov else TEXT_MUTED)
                sub_label = camera.focus_target_name if tool_id == "FOCUS" else (
                    "3D" if tool_id == "EARTH" else (
                    "PATH" if tool_id == "ORBIT" else (
                    "GRID" if tool_id == "GRID" else (
                    "DATA" if tool_id == "INFO" else "DIST"))))
                sub_surf = f_small.render(sub_label, True, ACCENT_CYAN)
                surface.blit(lbl_surf, lbl_surf.get_rect(center=(rect.centerx, rect.y + 14)))
                surface.blit(sub_surf, sub_surf.get_rect(center=(rect.centerx, rect.y + 28)))

    def _render_info_card(self, surface: pygame.Surface, inst, astro: dict, camera, fps: float = 60.0, renderer=None):
        f_title, f_header, f_body, f_small = self.fonts
        cw, ch = 320, 276
        cx = self.width - 75 - cw
        cy = 56
        card_rect = pygame.Rect(cx, cy, cw, ch)
        draw_panel(surface, card_rect, border_color=ACCENT_CYAN, fill_color=(8, 18, 34, 250))

        title = f_header.render("SYSTEM & TELEMETRY", True, ACCENT_CYAN)
        surface.blit(title, (cx + 14, cy + 12))
        pygame.draw.line(surface, PANEL_BORDER, (cx + 14, cy + 34), (cx + cw - 14, cy + 34), 1)

        gpu_name = self.gpu_info.get("renderer", "ModernGL / WGL") if self.gpu_info else "ModernGL / WGL"
        gpu_short = gpu_name.split("/")[0].strip()
        is_ded = self.gpu_info.get("is_dedicated", False) if self.gpu_info else False

        telem = renderer.telemetry if renderer else {}
        vis_tiles = telem.get("tiles_visible", 6)
        tot_tiles = telem.get("tiles_total", 6)
        cur_lod = telem.get("max_lod", 0)
        vram_mb = telem.get("vram_mb", 1.2)
        frame_ms = 1000.0 / max(1.0, fps)

        items = [
            ("Central Body", "Sun (Type G2V)"),
            ("Earth Distance", "1.000 AU (149.6M km)"),
            ("Moon Phase", f"{math.degrees(astro['moon_rot_angle']):.1f}°"),
            ("Active GPU", f"{gpu_short[:20]}"),
            ("Render Rate", f"{fps:.0f} FPS ({frame_ms:.1f} ms)"),
            ("Quadtree Tiles", f"{vis_tiles} vis / {tot_tiles} tot [LOD {cur_lod}]"),
            ("Terrain VRAM", f"{vram_mb:.2f} MB Resident"),
            ("Fall-back Tier", f"{renderer.earth_renderer.active_tier_name if (renderer and hasattr(renderer, 'earth_renderer')) else 'ACTIVE'}"),
        ]

        y = cy + 42
        for lbl, val in items:
            surface.blit(f_small.render(lbl, True, TEXT_MUTED), (cx + 14, y))
            val_col = ACCENT_GREEN if ("Dedicated" in val or "FPS" in val or "vis" in val) else (ACCENT_CYAN if "Tier" in lbl else TEXT_PRIMARY)
            surface.blit(f_small.render(val, True, val_col), (cx + 125, y))
            y += 22
