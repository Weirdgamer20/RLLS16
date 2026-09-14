"""
RLLS16 2D Earth Map Renderer.
High-performance CPU 2D orthographic renderer targeting 120 FPS.
Renders canonical Earth layers, 5-tier hierarchical LOD tiles, resolution-specific
surface caches, batched SDL blitting, spatial object layers, and real-time telemetry.
"""

import math
import time
import pygame
import numpy as np

from .camera_2d import Camera2D
from ..map.chunk_tile import (
    ChunkManager, ChunkTile,
    LOD_0_GLOBAL, LOD_1_CONTINENT, LOD_2_REGIONAL, LOD_3_LOCAL, LOD_4_GROUND,
    LOD_NAMES, get_lod_for_zoom
)
from ..map.layers import MapLayers, BIOME_COLORS, BIOME_NAMES
from ..map.spatial_objects import SpatialObjectManager, SpatialObject
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
    Presents the canonical rectangular Earth from above with 5-tier hierarchical LOD,
    resolution-specific cached surface tiles, batched blitting, discrete spatial object indexing,
    and granular frame profiling telemetry.
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

        # Spatial object manager: partitions trees, wildlife, water into O(K) lookup bins
        self.spatial_objects = SpatialObjectManager(self.layers)

        self.current_layer_mode = LAYER_NATURAL
        # Grid disabled by default to eliminate visual noise and CPU overhead
        self.show_grid = False
        self.show_weather = False
        self.show_icons = True
        self.selected_agent_id: int | None = None

        # Font caching
        pygame.font.init()
        self.font_tiny = pygame.font.SysFont("Consolas, Courier, monospace", 11)
        self.font_small = pygame.font.SysFont("Consolas, Courier, monospace", 13, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas, Courier, monospace", 16, bold=True)

        # Scale cache for macro overview (LOD 0)
        self._last_macro_key: tuple[int, int, str] = (0, 0, "")
        self._cached_macro_scaled: pygame.Surface | None = None

        # Dynamic weather particle state
        self._weather_time = 0.0

        # Detailed profiling telemetry
        self.telemetry = {
            "fps": 0.0,
            "render_ms": 0.0,
            "prep_ms": 0.0,
            "blit_ms": 0.0,
            "objects_ms": 0.0,
            "visible_chunks": 0,
            "cached_chunks": 0,
            "chunk_rebuilds": 0,
            "cache_hit_rate": 100.0,
            "lod": LOD_0_GLOBAL,
            "lod_name": LOD_NAMES[LOD_0_GLOBAL],
            "visible_objects": 0,
        }

    def resize(self, width: int, height: int):
        self.width = max(100, width)
        self.height = max(100, height)
        self.camera.resize(self.width, self.height)
        self._cached_macro_scaled = None
        self._last_macro_key = (0, 0, "")

    def set_layer_mode(self, mode: str):
        if mode in ALL_LAYER_MODES and mode != self.current_layer_mode:
            self.current_layer_mode = mode
            self.chunk_manager.clear_scaled_cache()
            self._cached_macro_scaled = None
            self._last_macro_key = (0, 0, "")

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
        Main 2D render pass with instrumentation for 120 FPS verification.
        Renders base map, hierarchical LOD tiles, spatial objects, agents, and overlays.
        """
        t0 = time.perf_counter()
        self._weather_time += dt
        self.chunk_manager.step_frame()

        lod = get_lod_for_zoom(self.camera.zoom)
        self.telemetry["lod"] = lod
        self.telemetry["lod_name"] = LOD_NAMES[lod]

        # 1. Clear background
        target_surface.fill((8, 12, 18))

        # 2. Render Earth Map (LOD 0 Macro or LOD 1-4 Batched Chunks)
        t_prep_start = time.perf_counter()
        self._render_map(target_surface, lod)
        t_prep_end = time.perf_counter()

        # 3. Optional Geographic Grid Lines
        if self.show_grid:
            self._render_grid(target_surface)

        # 4. Discrete Spatial Object Layer (Trees, Wildlife, Springs) at LOD >= 3
        t_obj_start = time.perf_counter()
        num_visible_objs = 0
        if self.show_icons and lod >= LOD_3_LOCAL:
            num_visible_objs = self._render_spatial_objects(target_surface, lod)
        self.telemetry["visible_objects"] = num_visible_objs

        # 5. Render Agents and Settlements
        if agents:
            self._render_agents(target_surface, agents)
        t_obj_end = time.perf_counter()

        # 6. Optional Dynamic Weather Overlay
        if self.show_weather:
            self._render_weather_overlay(target_surface)

        # 7. Map HUD Indicators (Scale bar, Cursor lat/lon, Zoom tier, Telemetry)
        self._render_map_hud(target_surface)

        t_total_end = time.perf_counter()

        # Record profiling telemetry
        stats = self.chunk_manager.cache_stats
        self.telemetry["render_ms"] = (t_total_end - t0) * 1000.0
        self.telemetry["prep_ms"] = (t_prep_end - t_prep_start) * 1000.0
        self.telemetry["objects_ms"] = (t_obj_end - t_obj_start) * 1000.0
        self.telemetry["cached_chunks"] = stats["cached_surfaces"]
        self.telemetry["chunk_rebuilds"] = stats["rebuilds_this_frame"]
        self.telemetry["cache_hit_rate"] = stats["hit_rate"]

    def _render_map(self, surface: pygame.Surface, lod: int):
        """
        Renders Earth map using either precomputed immutable macro overview (LOD 0)
        or batched resolution-specific cached chunk tiles (LOD 1-4).
        """
        min_wx, min_wy, max_wx, max_wy = self.camera.get_visible_world_rect()

        if lod == LOD_0_GLOBAL:
            # LOD 0: Fast single blit of scaled precomputed macro surface
            sx0, sy0 = self.camera.world_to_screen(0.0, 0.0)
            sx1, sy1 = self.camera.world_to_screen(1.0, 0.5)
            sw = max(1, round(sx1 - sx0))
            sh = max(1, round(sy1 - sy0))

            macro_key = (sw, sh, self.current_layer_mode)
            if macro_key != self._last_macro_key or self._cached_macro_scaled is None:
                macro_surf = self.chunk_manager.macro_surfaces.get(
                    self.current_layer_mode,
                    self.chunk_manager.macro_surfaces["NATURAL"]
                )
                self._cached_macro_scaled = pygame.transform.scale(macro_surf, (sw, sh))
                self._last_macro_key = macro_key

            surface.blit(self._cached_macro_scaled, (round(sx0), round(sy0)))
            self.telemetry["visible_chunks"] = 0

        else:
            # LOD 1-4: Spatial Chunk Tiles with Batched Blitting
            visible_chunks = self.chunk_manager.get_visible_chunks(min_wx, min_wy, max_wx, max_wy)
            self.telemetry["visible_chunks"] = len(visible_chunks)

            # Build batch for native SDL batched blitting
            blit_batch: list[tuple[pygame.Surface, tuple[int, int]]] = []
            for chunk in visible_chunks:
                csx0, csy0 = self.camera.world_to_screen(chunk.world_x0, chunk.world_y0)
                csx1, csy1 = self.camera.world_to_screen(chunk.world_x1, chunk.world_y1)
                sw = max(1, round(csx1 - csx0))
                sh = max(1, round(csy1 - csy0))

                # Retrieve resolution-specific scaled surface from LRU cache
                chunk_surf = self.chunk_manager.get_scaled_chunk_surface(
                    chunk, lod, sw, sh, self.current_layer_mode
                )
                blit_batch.append((chunk_surf, (round(csx0), round(csy0))))

            if blit_batch:
                surface.blits(blit_batch)

    def _render_spatial_objects(self, surface: pygame.Surface, lod: int) -> int:
        """
        Renders discrete ecological objects (trees, wildlife herds, freshwater springs)
        queried in O(K) time from the SpatialObjectManager rather than scanning raster cells.
        """
        min_wx, min_wy, max_wx, max_wy = self.camera.get_visible_world_rect()
        objects = self.spatial_objects.get_objects_in_world_rect(min_wx, min_wy, max_wx, max_wy)

        icon_size = (28, 28) if lod == LOD_4_GROUND else (20, 20)
        tree_icon = get_icon("tree", icon_size)
        water_icon = get_icon("water", (16, 16))
        animal_icon = get_icon("animal", icon_size)

        blit_list = []
        for obj in objects:
            sx, sy = self.camera.world_to_screen(obj.wx, obj.wy)
            rx, ry = round(sx), round(sy)

            if not (-30 <= rx <= self.width + 30 and -30 <= ry <= self.height + 30):
                continue

            if obj.obj_type == "tree":
                blit_list.append((tree_icon, (rx - icon_size[0] // 2, ry - icon_size[1] // 2)))
            elif obj.obj_type == "animal":
                blit_list.append((animal_icon, (rx - icon_size[0] // 2, ry - icon_size[1] // 2)))
            elif obj.obj_type == "water":
                blit_list.append((water_icon, (rx - 8, ry - 8)))

        if blit_list:
            surface.blits(blit_list)

        return len(blit_list)

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
                label = f"{abs(lon)}°{'E' if lon > 0 else ('W' if lon < 0 else ' PM')}"
                txt = self.font_tiny.render(label, True, (160, 180, 200))
                grid_surface.blit(txt, (round(sx) + 4, 8))

        surface.blit(grid_surface, (0, 0))

    def _render_agents(self, surface: pygame.Surface, agents: list):
        """Renders intelligent beings and human settlements."""
        human_icon = get_icon("human", (28, 28))
        shelter_icon = get_icon("shelter", (30, 30))
        reticle_icon = get_icon("cursor_select", (40, 40))

        for agent in agents:
            wx = getattr(agent, "wx", None)
            wy = getattr(agent, "wy", None)
            if wx is None or wy is None:
                lat = getattr(agent, "latitude_deg", 0.0)
                lon = getattr(agent, "longitude_deg", 0.0)
                wx = (lon + 180.0) / 360.0
                wy = (90.0 - lat) / 180.0 * 0.5

            sx, sy = self.camera.world_to_screen(wx, wy)

            if not (-50 <= sx <= self.width + 50 and -50 <= sy <= self.height + 50):
                continue

            rx, ry = round(sx), round(sy)
            is_selected = (self.selected_agent_id is not None and getattr(agent, "id", None) == self.selected_agent_id)

            if self.camera.zoom < 15.0:
                color = (255, 180, 50) if is_selected else (46, 216, 232)
                pygame.draw.circle(surface, color, (rx, ry), 5)
                pygame.draw.circle(surface, (255, 255, 255), (rx, ry), 2)
            else:
                is_settlement = hasattr(agent, "population")
                icon = shelter_icon if is_settlement else human_icon
                surface.blit(icon, (rx - icon.get_width() // 2, ry - icon.get_height() // 2))

                label_text = getattr(agent, "name", f"Settlement ({getattr(agent, 'population', 1)})")
                tag_surf = self.font_tiny.render(label_text, True, (240, 245, 255))
                bg_rect = tag_surf.get_rect(center=(rx, ry - 22))
                bg_rect.inflate_ip(6, 4)
                pygame.draw.rect(surface, (15, 20, 30, 200), bg_rect, border_radius=3)
                surface.blit(tag_surf, (bg_rect.x + 3, bg_rect.y + 2))

                health = getattr(getattr(agent, "vitals", None), "health", 100.0)
                bar_w = 28
                bar_h = 3
                bx = rx - bar_w // 2
                by = ry + 16
                pygame.draw.rect(surface, (40, 45, 55), (bx, by, bar_w, bar_h))
                health_w = int(bar_w * max(0.0, min(1.0, health / 100.0)))
                pygame.draw.rect(surface, (46, 220, 120), (bx, by, health_w, bar_h))

            if is_selected:
                pulse = 1.0 + 0.1 * math.sin(time.time() * 6.0)
                sz = int(40 * pulse)
                scaled_reticle = pygame.transform.scale(reticle_icon, (sz, sz))
                surface.blit(scaled_reticle, (rx - sz // 2, ry - sz // 2))

    def _render_weather_overlay(self, surface: pygame.Surface):
        """Subtle animated cloud drift and atmospheric humidity effect."""
        weather_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        cloud_color = (245, 250, 255, 18)
        offset = (self._weather_time * 15.0) % 200.0
        for y in range(40, self.height, 120):
            pygame.draw.ellipse(weather_surf, cloud_color, (offset - 100, y, 320, 60))
            pygame.draw.ellipse(weather_surf, cloud_color, (offset + 300, y + 25, 450, 75))

        surface.blit(weather_surf, (0, 0))

    def _render_map_hud(self, surface: pygame.Surface):
        """Renders 2D watcher HUD overlay: scale bar, cursor coordinates, and zoom tier."""
        # 1. Zoom Tier & Layer Badge with LOD indicator
        lod = self.telemetry["lod"]
        lod_name = self.telemetry["lod_name"]
        tier_str = f"{lod_name} ({self.camera.zoom:.1f}x) | LAYER: {self.current_layer_mode}"
        tier_txt = self.font_small.render(tier_str, True, (46, 216, 232))
        badge_rect = tier_txt.get_rect(topleft=(20, 16))
        badge_rect.inflate_ip(12, 8)
        pygame.draw.rect(surface, (12, 18, 28, 220), badge_rect, border_radius=4)
        pygame.draw.rect(surface, (46, 216, 232, 100), badge_rect, width=1, border_radius=4)
        surface.blit(tier_txt, (badge_rect.x + 6, badge_rect.y + 4))

        # 2. Scale Bar (bottom left)
        km_per_world_unit = 40075.0
        km_visible_width = km_per_world_unit / self.camera.zoom
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
