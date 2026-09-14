"""
RLLS16 2D Orthographic Watcher Camera.
Provides smooth panning, cursor-pinned mouse wheel zoom, fly-to animation,
and viewport culling rectangle calculation.
"""

import math
import numpy as np


class Camera2D:
    """
    2D Orthographic Watcher Camera.
    World coordinates:
      X in [0.0, 1.0] (West to East)
      Y in [0.0, 0.5] (North to South, 2:1 Earth aspect ratio)
    """

    # Semantic Zoom Tiers
    TIER_WORLD = "WORLD"           # 1.0x - 2.5x
    TIER_CONTINENT = "CONTINENT"   # 2.5x - 7.0x
    TIER_REGION = "REGION"         # 7.0x - 20.0x
    TIER_LOCAL = "LOCAL"           # 20.0x - 50.0x
    TIER_SETTLEMENT = "SETTLEMENT" # 50.0x - 120.0x
    TIER_AGENT = "AGENT"           # > 120.0x

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = max(100, screen_width)
        self.screen_height = max(100, screen_height)

        # Center in normalized world coordinates [0.0, 1.0] x [0.0, 0.5]
        self.center_x: float = 0.50
        self.center_y: float = 0.25

        # Zoom factor (1.0 = entire world map fills screen width)
        self.zoom: float = 1.0
        self.min_zoom: float = 0.85
        self.max_zoom: float = 250.0

        # Mouse interaction tracking
        self.is_panning: bool = False
        self.last_mouse_pos: tuple[int, int] = (0, 0)

        # Fly-to smooth animation
        self.is_flying: bool = False
        self.fly_start_x: float = 0.50
        self.fly_start_y: float = 0.25
        self.fly_start_zoom: float = 1.0
        self.fly_target_x: float = 0.50
        self.fly_target_y: float = 0.25
        self.fly_target_zoom: float = 50.0
        self.fly_duration: float = 1.2
        self.fly_elapsed: float = 0.0

    def resize(self, width: int, height: int):
        self.screen_width = max(100, width)
        self.screen_height = max(100, height)

    @property
    def pixels_per_world_unit(self) -> float:
        """Pixels per 1.0 world width unit."""
        return self.screen_width * self.zoom

    @property
    def zoom_tier(self) -> str:
        if self.zoom < 2.5:
            return self.TIER_WORLD
        elif self.zoom < 7.0:
            return self.TIER_CONTINENT
        elif self.zoom < 20.0:
            return self.TIER_REGION
        elif self.zoom < 50.0:
            return self.TIER_LOCAL
        elif self.zoom < 120.0:
            return self.TIER_SETTLEMENT
        else:
            return self.TIER_AGENT

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        """Convert normalized world coords (wx in [0,1], wy in [0,0.5]) to screen pixel coords."""
        scale = self.pixels_per_world_unit
        sx = (wx - self.center_x) * scale + self.screen_width / 2.0
        sy = (wy - self.center_y) * scale + self.screen_height / 2.0
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert screen pixel coords to normalized world coords."""
        scale = self.pixels_per_world_unit
        wx = (sx - self.screen_width / 2.0) / scale + self.center_x
        wy = (sy - self.screen_height / 2.0) / scale + self.center_y
        return wx, wy

    def get_visible_world_rect(self) -> tuple[float, float, float, float]:
        """
        Returns bounding rectangle of visible world:
        (min_wx, min_wy, max_wx, max_wy).
        """
        scale = self.pixels_per_world_unit
        half_w = (self.screen_width / 2.0) / scale
        half_h = (self.screen_height / 2.0) / scale
        return (
            self.center_x - half_w,
            self.center_y - half_h,
            self.center_x + half_w,
            self.center_y + half_h,
        )

    def _clamp_bounds(self):
        """Keep camera centered within sensible bounds around Earth map."""
        # Allow slight padding beyond edges for smooth viewing
        margin_x = 0.25 / self.zoom
        margin_y = 0.25 / self.zoom
        self.center_x = max(-margin_x, min(1.0 + margin_x, self.center_x))
        self.center_y = max(-margin_y, min(0.5 + margin_y, self.center_y))

    def pan_pixels(self, dx: float, dy: float):
        """Pan camera by screen pixel delta."""
        scale = self.pixels_per_world_unit
        self.center_x -= dx / scale
        self.center_y -= dy / scale
        self._clamp_bounds()

    def zoom_around_cursor(self, delta: float, mouse_x: int, mouse_y: int):
        """
        Cursor-anchored continuous zoom.
        Ensures the world point directly under the cursor remains geographically pinned!
        """
        # 1. World point under cursor before zoom
        wx, wy = self.screen_to_world(mouse_x, mouse_y)

        # 2. Adjust zoom smoothly
        # Normalized delta (standard scroll wheel = 1 or -1)
        step = math.copysign(min(3.0, abs(delta)), delta)
        zoom_mult = 1.15 ** step
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * zoom_mult))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom

        # 3. Recalculate camera center so that world_to_screen(wx, wy) == (mouse_x, mouse_y)
        new_scale = self.pixels_per_world_unit
        self.center_x = wx - (mouse_x - self.screen_width / 2.0) / new_scale
        self.center_y = wy - (mouse_y - self.screen_height / 2.0) / new_scale
        self._clamp_bounds()

    def fly_to(
        self,
        target_wx: float,
        target_wy: float,
        target_zoom: float = 60.0,
        duration: float = 1.2,
    ):
        """Smooth animated fly-to towards a target coordinate and zoom level."""
        self.is_flying = True
        self.fly_start_x = self.center_x
        self.fly_start_y = self.center_y
        self.fly_start_zoom = self.zoom
        self.fly_target_x = target_wx
        self.fly_target_y = target_wy
        self.fly_target_zoom = max(self.min_zoom, min(self.max_zoom, target_zoom))
        self.fly_duration = max(0.1, duration)
        self.fly_elapsed = 0.0

    def fly_to_latlon(
        self,
        lat_deg: float,
        lon_deg: float,
        target_zoom: float = 60.0,
        duration: float = 1.2,
    ):
        """Convenience method to fly to real latitude and longitude degrees."""
        wx = (lon_deg + 180.0) / 360.0
        wy = (90.0 - lat_deg) / 180.0 * 0.5  # World height is 0.5
        self.fly_to(wx, wy, target_zoom, duration)

    def handle_mouse_down(self, button: int, pos: tuple[int, int]):
        self.last_mouse_pos = pos
        # Middle click (button 2) or Right click (button 3) initiate pan
        if button in (2, 3):
            self.is_panning = True

    def handle_mouse_up(self, button: int):
        if button in (2, 3):
            self.is_panning = False

    def handle_mouse_motion(self, pos: tuple[int, int]):
        if self.is_panning:
            dx = pos[0] - self.last_mouse_pos[0]
            dy = pos[1] - self.last_mouse_pos[1]
            self.pan_pixels(dx, dy)
        self.last_mouse_pos = pos

    def handle_keyboard(self, key_state, dt: float):
        import pygame
        speed_pixels = 450.0 * dt  # pixels per second

        def is_down(k):
            try:
                return bool(key_state[k])
            except (IndexError, KeyError):
                return False

        if is_down(pygame.K_LSHIFT) or is_down(pygame.K_RSHIFT):
            speed_pixels *= 2.5

        dx = 0.0
        dy = 0.0
        if is_down(pygame.K_a) or is_down(pygame.K_LEFT):
            dx += speed_pixels
        if is_down(pygame.K_d) or is_down(pygame.K_RIGHT):
            dx -= speed_pixels
        if is_down(pygame.K_w) or is_down(pygame.K_UP):
            dy += speed_pixels
        if is_down(pygame.K_s) or is_down(pygame.K_DOWN):
            dy -= speed_pixels

        if dx != 0.0 or dy != 0.0:
            self.pan_pixels(dx, dy)

    def update(self, dt: float):
        """Update smooth fly-to animation with cubic ease-out."""
        if self.is_flying:
            self.fly_elapsed += dt
            t = min(1.0, self.fly_elapsed / self.fly_duration)
            ease = 1.0 - (1.0 - t) ** 3  # Cubic ease-out

            self.center_x = self.fly_start_x + (self.fly_target_x - self.fly_start_x) * ease
            self.center_y = self.fly_start_y + (self.fly_target_y - self.fly_start_y) * ease
            self.zoom = self.fly_start_zoom + (self.fly_target_zoom - self.fly_start_zoom) * ease
            self._clamp_bounds()

            if t >= 1.0:
                self.is_flying = False
