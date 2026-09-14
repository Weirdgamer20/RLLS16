"""
RLLS16 2D Earth Map Renderer.
High-performance CPU 2D orthographic renderer targeting 120 FPS.
Renders canonical Earth layers, chunked tiles, 16-bit icons, agent positions,
and environmental visualization overlays.
"""

import math
import time
import pygame
import numpy as np

from .camera_2d import Camera2D
from ..map.chunk_tile import ChunkManager, ChunkTile
from ..map.layers import MapLayers, BIOME_COLORS, BIOME_NAMES
from ..assets_loader import get_icon, get_cosmetic


# Visual Layer Modes
LAYER_NATURAL = "NATURAL"
LAYER_ELEVATION = "ELEVATION"
LAYER_TEMPERATURE = "TEMPERATURE"
LAYER_PRECIPITATION = "PRECIPITATION"
LAYER_BIOMES = "BIOMES"
LAYER_WATER = "WATER"

ALL_LAYER_MODES = [
    LAYER_NATURAL,
    LAYER_ELEVATION,
    LAYER_TEMPERATURE,
    LAYER_PRECIPITATION,
    LAYER_BIOMES,
    LAYER_WATER,
]


class Renderer2D:
    """
    2D Watcher Map Renderer.
    Presents the canonical rectangular Earth from above with smooth viewport culling,
    multi-layer inspection, 16-bit iconography at deep zoom, and real-time agent tracking.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        chunk_manager: ChunkManager,
        camera: Camera2D,
    ):
        self.width = screen_width
        self.height = screen_height
        self.chunk_manager = chunk_manager
        self.camera = camera
        self.layers = chunk_manager.layers

        self.current_layer_mode = LAYER_NATURAL
        self.show_grid = True
        self.show_weather = False
        self.show_icons = True
        self.selected_agent_id: int | None = None

        # Font caching for coordinates, labels, and scale bar
        pygame.font.init()
        self.font_tiny = pygame.font.SysFont("Consolas, Courier, monospace", 11)
        self.font_small = pygame.font.SysFont("Consolas, Courier, monospace", 13, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas, Courier, monospace", 16, bold=True)

        # Pre-calculated scale cache for macro surface
        self._last_macro_size: tuple[int, int] = (0, 0)
        self._cached_macro_scaled: pygame.Surface | None = None

        # Dynamic weather particle state
        self._weather_time = 0.0

    def resize(self, width: int, height: int):
        self.width = max(100, width)
        self.height = max(100, height)
        self.camera.resize(self.width, self.height)
        self._cached_macro_scaled = None

    def set_layer_mode(self, mode: str):
        if mode in ALL_LAYER_MODES and mode != self.current_layer_mode:
            self.current_layer_mode = mode
            # Invalidate chunk cache so chunks are re-rendered in the chosen mode
            self.chunk_manager.cache.clear()
            self._cached_macro_scaled = None

    def toggle_grid(self) -> bool:
        self.show_grid = not self.show_grid
        return self.show_grid

    def toggle_weather(self) -> bool:
        self.show_weather = not self.show_weather
        return self.show_weather

    def toggle_icons(self) -> bool:
        self.show_icons = not self.show_icons
        return self.show_icons

    def render(self, target_surface: pygame.Surface, agents: list | None = None, dt: float = 0.016):
        """
        Main 2D render pass.
        Renders base map, environmental layers, icons, agents, and overlays.
        """
        self._weather_time += dt
        self.chunk_manager.step_frame()

        # 1. Clear background to dark space color
        target_surface.fill((8, 12, 18))

        # 2. Render Earth Map
        if self.current_layer_mode == LAYER_NATURAL:
            self._render_natural_map(target_surface)
        else:
            self._render_analytical_layer(target_surface, self.current_layer_mode)

        # 3. Optional Geographic Grid Lines
        if self.show_grid:
            self._render_grid(target_surface)

        # 4. Deep Zoom 16-bit Ecological & Resource Icons
        if self.show_icons and self.camera.zoom >= 25.0:
            self._render_environmental_icons(target_surface)

        # 5. Render Agents and Settlements
        if agents:
            self._render_agents(target_surface, agents)

        # 6. Optional Dynamic Weather Overlay
        if self.show_weather:
            self._render_weather_overlay(target_surface)

        # 7. Map HUD Indicators (Scale bar, Cursor lat/lon, Zoom tier)
        self._render_map_hud(target_surface)

    def _render_natural_map(self, surface: pygame.Surface):
        """Renders natural Earth map using either pre-baked macro overview or spatial chunk tiles."""
        min_wx, min_wy, max_wx, max_wy = self.camera.get_visible_world_rect()

        if self.camera.zoom < 2.5:
            # Macro view: single fast blit of scaled overview
            sx0, sy0 = self.camera.world_to_screen(0.0, 0.0)
            sx1, sy1 = self.camera.world_to_screen(1.0, 0.5)
            sw = max(1, round(sx1 - sx0))
            sh = max(1, round(sy1 - sy0))

            macro_surf = self.chunk_manager.macro_surface
            if (sw, sh) != self._last_macro_size or self._cached_macro_scaled is None:
                self._cached_macro_scaled = pygame.transform.scale(macro_surf, (sw, sh))
                self._last_macro_size = (sw, sh)

            surface.blit(self._cached_macro_scaled, (round(sx0), round(sy0)))
        else:
            # Regional & local view: chunk viewport culling
            visible_chunks = self.chunk_manager.get_visible_chunks(min_wx, min_wy, max_wx, max_wy)
            for chunk in visible_chunks:
                csx0, csy0 = self.camera.world_to_screen(chunk.world_x0, chunk.world_y0)
                csx1, csy1 = self.camera.world_to_screen(chunk.world_x1, chunk.world_y1)
                sw = max(1, round(csx1 - csx0))
                sh = max(1, round(csy1 - csy0))

                chunk_surf = self.chunk_manager.get_chunk_surface(chunk)
                scaled_surf = pygame.transform.scale(chunk_surf, (sw, sh))
                surface.blit(scaled_surf, (round(csx0), round(csy0)))

    def _render_analytical_layer(self, surface: pygame.Surface, mode: str):
        """Renders false-color analytical visualization for specific environmental layers."""
        # For analytical layers, we sample the layers directly and blit with color mapping
        min_wx, min_wy, max_wx, max_wy = self.camera.get_visible_world_rect()
        sx0, sy0 = self.camera.world_to_screen(0.0, 0.0)
        sx1, sy1 = self.camera.world_to_screen(1.0, 0.5)
        sw = max(1, round(sx1 - sx0))
        sh = max(1, round(sy1 - sy0))

        if not hasattr(self, f"_analytical_surf_{mode}"):
            # Build analytical RGB array once
            h, w = self.layers.height, self.layers.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

            if mode == LAYER_ELEVATION:
                elev = self.layers.elevation
                # Hypsometric tint: deep blue for ocean, green to brown to white for land
                land = self.layers.land_mask
                ocean_v = np.clip(elev / 0.50, 0.0, 1.0) * 160.0
                rgb[..., 0] = np.where(land, (elev * 255.0).astype(np.uint8), 20)
                rgb[..., 1] = np.where(land, (elev * 230.0).astype(np.uint8), (30 + ocean_v * 0.4).astype(np.uint8))
                rgb[..., 2] = np.where(land, (elev * 190.0).astype(np.uint8), (70 + ocean_v * 0.8).astype(np.uint8))

            elif mode == LAYER_TEMPERATURE:
                temp = self.layers.temperature
                # Thermal ramp: blue (cold) -> green -> yellow -> red (hot)
                r = np.clip((temp - 0.45) * 3.0, 0.0, 1.0) * 255.0
                g = (np.sin(temp * np.pi) * 230.0)
                b = np.clip((0.55 - temp) * 3.0, 0.0, 1.0) * 255.0
                rgb[..., 0] = r.astype(np.uint8)
                rgb[..., 1] = g.astype(np.uint8)
                rgb[..., 2] = b.astype(np.uint8)

            elif mode == LAYER_PRECIPITATION:
                precip = self.layers.precipitation
                # Aridity/moisture ramp: dry tan -> green -> deep cyan/blue
                rgb[..., 0] = ((1.0 - precip * 0.7) * 190.0).astype(np.uint8)
                rgb[..., 1] = ((0.5 + precip * 0.5) * 200.0).astype(np.uint8)
                rgb[..., 2] = ((0.3 + precip * 0.7) * 245.0).astype(np.uint8)

            elif mode == LAYER_BIOMES:
                biome = self.layers.base_biome
                land = self.layers.land_mask
                for b_id, color in BIOME_COLORS.items():
                    mask = (biome == b_id)
                    rgb[mask] = color
                # Highlight ocean distinctly
                rgb[~land] = (15, 35, 65)

            elif mode == LAYER_WATER:
                land = self.layers.land_mask
                rivers = self.layers.rivers_lakes
                water = (~land) | (rivers > 0.45)
                rgb[water] = (46, 175, 240)
                rgb[~water] = (45, 48, 55)

            rgb_t = np.transpose(rgb, (1, 0, 2))
            surf = pygame.surfarray.make_surface(rgb_t)
            setattr(self, f"_analytical_surf_{mode}", surf)

        base_surf = getattr(self, f"_analytical_surf_{mode}")
        scaled = pygame.transform.scale(base_surf, (sw, sh))
        surface.blit(scaled, (round(sx0), round(sy0)))

    def _render_grid(self, surface: pygame.Surface):
        """Draws geographic lat/lon grid lines and coordinate annotations."""
        grid_color = (255, 255, 255, 32)
        equator_color = (255, 200, 80, 70)

        grid_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Latitude lines (North/South)
        latitudes = [-60, -30, 0, 30, 60]
        for lat in latitudes:
            wy = (90.0 - lat) / 180.0 * 0.5
            _, sy = self.camera.world_to_screen(0.0, wy)
            if 0 <= sy <= self.height:
                col = equator_color if lat == 0 else grid_color
                pygame.draw.line(grid_surface, col, (0, round(sy)), (self.width, round(sy)), 1)
                # Label
                label = f"{abs(lat)}°{'N' if lat > 0 else ('S' if lat < 0 else ' EQ')}"
                txt = self.font_tiny.render(label, True, (160, 180, 200))
                grid_surface.blit(txt, (8, round(sy) + 2))

        # Longitude lines (West/East)
        longitudes = [-120, -60, 0, 60, 120]
        for lon in longitudes:
            wx = (lon + 180.0) / 360.0
            sx, _ = self.camera.world_to_screen(wx, 0.0)
            if 0 <= sx <= self.width:
                col = equator_color if lon == 0 else grid_color
                pygame.draw.line(grid_surface, col, (round(sx), 0), (round(sx), self.height), 1)
                # Label
                label = f"{abs(lon)}°{'E' if lon > 0 else ('W' if lon < 0 else ' PM')}"
                txt = self.font_tiny.render(label, True, (160, 180, 200))
                grid_surface.blit(txt, (round(sx) + 4, 8))

        surface.blit(grid_surface, (0, 0))

    def _render_environmental_icons(self, surface: pygame.Surface):
        """
        Renders 16-bit vegetation, wildlife, and river icons when zoomed into
        regional and local tiers.
        """
        min_wx, min_wy, max_wx, max_wy = self.camera.get_visible_world_rect()

        # Determine cell bounds in canonical grid
        c_x0 = max(0, int(min_wx * self.layers.width))
        c_y0 = max(0, int((min_wy / 0.5) * self.layers.height))
        c_x1 = min(self.layers.width - 1, int(max_wx * self.layers.width) + 1)
        c_y1 = min(self.layers.height - 1, int((max_wy / 0.5) * self.layers.height) + 1)

        # Subsample rate depending on zoom to avoid icon clutter
        step = max(1, int(8.0 / (self.camera.zoom / 25.0)))
        icon_size = (20, 20) if self.camera.zoom < 60.0 else (28, 28)

        tree_icon = get_icon("tree", icon_size)
        water_icon = get_icon("water", (16, 16))
        animal_icon = get_icon("animal", icon_size)

        for cy in range(c_y0, c_y1 + 1, step):
            for cx in range(c_x0, c_x1 + 1, step):
                if not self.layers.land_mask[cy, cx]:
                    continue

                wx = cx / self.layers.width
                wy = (cy / self.layers.height) * 0.5
                sx, sy = self.camera.world_to_screen(wx, wy)

                veg = self.layers.vegetation[cy, cx]
                wild = self.layers.wildlife[cy, cx]
                water = self.layers.rivers_lakes[cy, cx]

                # Blit tree icon on dense vegetation
                if veg > 0.65:
                    surface.blit(tree_icon, (round(sx - icon_size[0] // 2), round(sy - icon_size[1] // 2)))
                # Blit animal icon in wildlife habitats
                elif wild > 0.70 and (cx + cy) % 3 == 0:
                    surface.blit(animal_icon, (round(sx - icon_size[0] // 2), round(sy - icon_size[1] // 2)))
                # Blit water droplet near river banks
                elif water > 0.50 and (cx + cy) % 2 == 0:
                    surface.blit(water_icon, (round(sx - 8), round(sy - 8)))

    def _render_agents(self, surface: pygame.Surface, agents: list):
        """Renders intelligent beings and human settlements."""
        human_icon = get_icon("human", (28, 28))
        shelter_icon = get_icon("shelter", (30, 30))
        reticle_icon = get_icon("cursor_select", (40, 40))

        for agent in agents:
            # Extract world position
            wx = getattr(agent, "wx", None)
            wy = getattr(agent, "wy", None)
            if wx is None or wy is None:
                # Fallback to lat/lon degrees
                lat = getattr(agent, "latitude_deg", 0.0)
                lon = getattr(agent, "longitude_deg", 0.0)
                wx = (lon + 180.0) / 360.0
                wy = (90.0 - lat) / 180.0 * 0.5

            sx, sy = self.camera.world_to_screen(wx, wy)

            # Viewport culling
            if not (-50 <= sx <= self.width + 50 and -50 <= sy <= self.height + 50):
                continue

            rx, ry = round(sx), round(sy)
            is_selected = (self.selected_agent_id is not None and getattr(agent, "id", None) == self.selected_agent_id)

            if self.camera.zoom < 15.0:
                # Distant zoom: bright glowing marker dot
                color = (255, 180, 50) if is_selected else (46, 216, 232)
                pygame.draw.circle(surface, color, (rx, ry), 5)
                pygame.draw.circle(surface, (255, 255, 255), (rx, ry), 2)
            else:
                # Deep zoom: 16-bit sprite, vitals bar, and name label
                is_settlement = hasattr(agent, "population")
                icon = shelter_icon if is_settlement else human_icon
                surface.blit(icon, (rx - icon.get_width() // 2, ry - icon.get_height() // 2))

                # Name / Population Tag
                label_text = getattr(agent, "name", f"Settlement ({getattr(agent, 'population', 1)})")
                tag_surf = self.font_tiny.render(label_text, True, (240, 245, 255))
                # Dark background backing for legibility
                bg_rect = tag_surf.get_rect(center=(rx, ry - 22))
                bg_rect.inflate_ip(6, 4)
                pygame.draw.rect(surface, (15, 20, 30, 200), bg_rect, border_radius=3)
                surface.blit(tag_surf, (bg_rect.x + 3, bg_rect.y + 2))

                # Health mini-bar
                health = getattr(getattr(agent, "vitals", None), "health", 100.0)
                bar_w = 28
                bar_h = 3
                bx = rx - bar_w // 2
                by = ry + 16
                pygame.draw.rect(surface, (40, 45, 55), (bx, by, bar_w, bar_h))
                health_w = int(bar_w * max(0.0, min(1.0, health / 100.0)))
                pygame.draw.rect(surface, (46, 220, 120), (bx, by, health_w, bar_h))

            # Selected focus reticle
            if is_selected:
                # Pulsing reticle
                pulse = 1.0 + 0.1 * math.sin(time.time() * 6.0)
                sz = int(40 * pulse)
                scaled_reticle = pygame.transform.scale(reticle_icon, (sz, sz))
                surface.blit(scaled_reticle, (rx - sz // 2, ry - sz // 2))

    def _render_weather_overlay(self, surface: pygame.Surface):
        """Subtle animated cloud drift and atmospheric humidity effect."""
        weather_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Dynamic cloud ribbons drifting eastward
        cloud_color = (245, 250, 255, 18)
        offset = (self._weather_time * 15.0) % 200.0
        for y in range(40, self.height, 120):
            pygame.draw.ellipse(
                weather_surf,
                cloud_color,
                (offset - 100, y, 320, 60)
            )
            pygame.draw.ellipse(
                weather_surf,
                cloud_color,
                (offset + 300, y + 25, 450, 75)
            )

        surface.blit(weather_surf, (0, 0))

    def _render_map_hud(self, surface: pygame.Surface):
        """Renders 2D watcher HUD overlay: scale bar, cursor coordinates, and zoom tier."""
        # 1. Zoom Tier & Layer Badge (top left of world viewport)
        tier_str = f"VIEW: {self.camera.zoom_tier} ({self.camera.zoom:.1f}x) | LAYER: {self.current_layer_mode}"
        tier_txt = self.font_small.render(tier_str, True, (46, 216, 232))
        badge_rect = tier_txt.get_rect(topleft=(20, 16))
        badge_rect.inflate_ip(12, 8)
        pygame.draw.rect(surface, (12, 18, 28, 220), badge_rect, border_radius=4)
        pygame.draw.rect(surface, (46, 216, 232, 100), badge_rect, width=1, border_radius=4)
        surface.blit(tier_txt, (badge_rect.x + 6, badge_rect.y + 4))

        # 2. Scale Bar (bottom left)
        # Earth equator circumference ~ 40,075 km
        km_per_world_unit = 40075.0
        km_visible_width = km_per_world_unit / self.camera.zoom
        # Choose a round scale bar width
        bar_screen_pixels = 140
        fraction_of_screen = bar_screen_pixels / self.width
        bar_km = km_visible_width * fraction_of_screen
        if bar_km > 1000.0:
            scale_label = f"{round(bar_km / 1000.0) * 1000:,} km"
        elif bar_km > 100.0:
            scale_label = f"{round(bar_km / 100.0) * 100:,} km"
        else:
            scale_label = f"{max(1, round(bar_km)):,} km"

        by = self.height - 30
        bx = 20
        pygame.draw.line(surface, (230, 240, 255), (bx, by), (bx + bar_screen_pixels, by), 2)
        pygame.draw.line(surface, (230, 240, 255), (bx, by - 4), (bx, by + 4), 2)
        pygame.draw.line(surface, (230, 240, 255), (bx + bar_screen_pixels, by - 4), (bx + bar_screen_pixels, by + 4), 2)

        scale_txt = self.font_tiny.render(scale_label, True, (220, 235, 250))
        surface.blit(scale_txt, (bx + 12, by - 14))

        # 3. Cursor Lat/Lon Position
        mx, my = pygame.mouse.get_pos()
        c_wx, c_wy = self.camera.screen_to_world(mx, my)
        if 0.0 <= c_wx <= 1.0 and 0.0 <= c_wy <= 0.5:
            c_lon = c_wx * 360.0 - 180.0
            c_lat = 90.0 - (c_wy / 0.5) * 180.0
            coord_str = f"LAT: {abs(c_lat):.2f}°{'N' if c_lat >= 0 else 'S'}  LON: {abs(c_lon):.2f}°{'E' if c_lon >= 0 else 'W'}"
            coord_txt = self.font_tiny.render(coord_str, True, (160, 185, 210))
            surface.blit(coord_txt, (bx + bar_screen_pixels + 24, by - 12))
