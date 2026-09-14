"""
RLLS16 2D Spatial Chunk & Tile Cache System.
Partitions the rectangular 2D Earth into spatial chunks for viewport culling,
5-tier hierarchical LOD rendering, resolution-specific surface caching,
and memory-bounded LRU eviction.
"""

from collections import OrderedDict
import math
import pygame
import numpy as np
from .layers import MapLayers, BIOME_COLORS


# 5-Tier Hierarchical LOD Definitions
LOD_0_GLOBAL = 0       # Full Earth macro overview (zoom < 1.8x)
LOD_1_CONTINENT = 1    # Continental scale tiles (1.8x <= zoom < 6.0x)
LOD_2_REGIONAL = 2     # Regional scale tiles (6.0x <= zoom < 18.0x)
LOD_3_LOCAL = 3        # Local terrain scale + spatial objects (18.0x <= zoom < 50.0x)
LOD_4_GROUND = 4       # Ground inspection scale + full discrete entities (zoom >= 50.0x)

LOD_NAMES = {
    LOD_0_GLOBAL: "LOD 0 GLOBAL",
    LOD_1_CONTINENT: "LOD 1 CONTINENT",
    LOD_2_REGIONAL: "LOD 2 REGIONAL",
    LOD_3_LOCAL: "LOD 3 LOCAL",
    LOD_4_GROUND: "LOD 4 GROUND",
}


def get_lod_for_zoom(zoom: float) -> int:
    """Calculates authoritative hierarchical LOD level from camera zoom factor."""
    if zoom < 1.8:
        return LOD_0_GLOBAL
    elif zoom < 6.0:
        return LOD_1_CONTINENT
    elif zoom < 18.0:
        return LOD_2_REGIONAL
    elif zoom < 50.0:
        return LOD_3_LOCAL
    else:
        return LOD_4_GROUND


class ChunkTile:
    """A spatial 2D tile representing a slice of the rectangular Earth."""
    __slots__ = (
        'chunk_x', 'chunk_y', 'cell_x0', 'cell_y0', 'cell_x1', 'cell_y1',
        'world_x0', 'world_y0', 'world_x1', 'world_y1',
        'key', 'last_used_frame'
    )

    def __init__(
        self,
        chunk_x: int,
        chunk_y: int,
        cell_x0: int,
        cell_y0: int,
        cell_x1: int,
        cell_y1: int,
        total_w: int,
        total_h: int,
    ):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.cell_x0 = cell_x0
        self.cell_y0 = cell_y0
        self.cell_x1 = cell_x1
        self.cell_y1 = cell_y1

        # World coordinates: X in [0, 1], Y in [0, 0.5]
        self.world_x0 = cell_x0 / total_w
        self.world_y0 = (cell_y0 / total_h) * 0.5
        self.world_x1 = cell_x1 / total_w
        self.world_y1 = (cell_y1 / total_h) * 0.5

        self.key = f"{chunk_x}_{chunk_y}"
        self.last_used_frame = 0

    def intersects(self, min_wx: float, min_wy: float, max_wx: float, max_wy: float) -> bool:
        """Test intersection with camera's visible world rectangle."""
        return not (
            self.world_x1 < min_wx or
            self.world_x0 > max_wx or
            self.world_y1 < min_wy or
            self.world_y0 > max_wy
        )


class ChunkManager:
    """
    Manages 2D spatial chunks, 5-tier hierarchical LOD, precomputed analytical surfaces,
    and resolution-specific scaled surface caching with LRU eviction.
    """

    def __init__(self, layers: MapLayers, chunk_size: int = 32, max_cached: int = 1024):
        self.layers = layers
        self.chunk_size = chunk_size
        self.max_cached = max_cached
        self.current_frame = 0

        self.total_h = layers.height
        self.total_w = layers.width

        self.num_chunks_x = int(math.ceil(self.total_w / chunk_size))
        self.num_chunks_y = int(math.ceil(self.total_h / chunk_size))

        self.chunks: list[ChunkTile] = []
        self._chunks_by_key: dict[str, ChunkTile] = {}
        for cy in range(self.num_chunks_y):
            for cx in range(self.num_chunks_x):
                x0 = cx * chunk_size
                y0 = cy * chunk_size
                x1 = min(self.total_w, x0 + chunk_size)
                y1 = min(self.total_h, y0 + chunk_size)
                tile = ChunkTile(cx, cy, x0, y0, x1, y1, self.total_w, self.total_h)
                self.chunks.append(tile)
                self._chunks_by_key[tile.key] = tile

        # Profiling and cache telemetry counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.rebuilds_this_frame: int = 0

        # Unscaled base chunk surface cache: (chunk_key, layer_mode) -> pygame.Surface
        self._unscaled_chunk_cache: dict[tuple[str, str], pygame.Surface] = {}

        # Resolution-specific scaled surface cache with LRU eviction:
        # Key: (chunk_key, lod, pixel_w, pixel_h, layer_mode) -> pygame.Surface
        self._scaled_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()
        self.cache: dict[str, pygame.Surface] = {}

        # Immutable precomputed macro surfaces for all layer modes at LOD 0
        self.macro_surfaces: dict[str, pygame.Surface] = self._precompute_all_macro_surfaces()

    @property
    def macro_surface(self) -> pygame.Surface:
        """Default natural macro surface for backward compatibility."""
        return self.macro_surfaces["NATURAL"]

    @property
    def cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100.0) if total > 0 else 100.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_surfaces": len(self._scaled_cache),
            "rebuilds_this_frame": self.rebuilds_this_frame,
        }

    def _precompute_all_macro_surfaces(self) -> dict[str, pygame.Surface]:
        """Precomputes immutable macro surfaces for all supported layer modes."""
        modes = ["NATURAL", "ELEVATION", "TEMPERATURE", "PRECIPITATION", "BIOMES", "WATER"]
        surfs = {}
        for m in modes:
            surfs[m] = self._build_macro_surface(m)
        return surfs

    def _build_macro_surface(self, mode: str = "NATURAL") -> pygame.Surface:
        """Renders the entire 2D Earth into a crisp 16-bit macro overview surface."""
        w, h = self.total_w, self.total_h
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        land = self.layers.land_mask
        elev = self.layers.elevation

        if mode == "NATURAL":
            biome = self.layers.base_biome
            rivers = self.layers.rivers_lakes

            # Ocean coloring with bathymetric gradient
            ocean_depth = np.clip(1.0 - elev / 0.50, 0.0, 1.0)
            rgb[..., 0] = (14 + 18 * (1.0 - ocean_depth)).astype(np.uint8)
            rgb[..., 1] = (38 + 48 * (1.0 - ocean_depth)).astype(np.uint8)
            rgb[..., 2] = (85 + 75 * (1.0 - ocean_depth)).astype(np.uint8)

            # Land biome coloration
            for b_id, color in BIOME_COLORS.items():
                if b_id == 0:
                    continue
                mask = land & (biome == b_id)
                if np.any(mask):
                    rgb[mask] = color

            # Altitude relief shading
            alt = np.maximum(0.0, elev - 0.50) * 1.8
            if np.any(land):
                shaded = rgb[land].astype(np.float32) * (0.88 + 0.28 * alt[land, np.newaxis])
                rgb[land] = np.clip(shaded, 0, 255).astype(np.uint8)

            # Snow-capped peaks
            snow = land & (elev > 0.80)
            if np.any(snow):
                rgb[snow] = np.clip(rgb[snow].astype(np.float32) * 0.4 + np.array([245, 248, 255]) * 0.6, 0, 255).astype(np.uint8)

            # Major rivers and lakes
            water_mask = (rivers > 0.45) & land
            if np.any(water_mask):
                rgb[water_mask] = (42, 140, 168)

        elif mode == "ELEVATION":
            ocean_v = np.clip(elev / 0.50, 0.0, 1.0) * 160.0
            rgb[..., 0] = np.where(land, (elev * 255.0).astype(np.uint8), 20)
            rgb[..., 1] = np.where(land, (elev * 230.0).astype(np.uint8), (30 + ocean_v * 0.4).astype(np.uint8))
            rgb[..., 2] = np.where(land, (elev * 190.0).astype(np.uint8), (70 + ocean_v * 0.8).astype(np.uint8))

        elif mode == "TEMPERATURE":
            temp = self.layers.temperature
            r = np.clip((temp - 0.45) * 3.0, 0.0, 1.0) * 255.0
            g = (np.sin(temp * np.pi) * 230.0)
            b = np.clip((0.55 - temp) * 3.0, 0.0, 1.0) * 255.0
            rgb[..., 0] = r.astype(np.uint8)
            rgb[..., 1] = g.astype(np.uint8)
            rgb[..., 2] = b.astype(np.uint8)

        elif mode == "PRECIPITATION":
            precip = self.layers.precipitation
            rgb[..., 0] = ((1.0 - precip * 0.7) * 190.0).astype(np.uint8)
            rgb[..., 1] = ((0.5 + precip * 0.5) * 200.0).astype(np.uint8)
            rgb[..., 2] = ((0.3 + precip * 0.7) * 245.0).astype(np.uint8)

        elif mode == "BIOMES":
            biome = self.layers.base_biome
            for b_id, color in BIOME_COLORS.items():
                mask = (biome == b_id)
                rgb[mask] = color
            rgb[~land] = (15, 35, 65)

        elif mode == "WATER":
            rivers = self.layers.rivers_lakes
            water = (~land) | (rivers > 0.45)
            rgb[water] = (46, 175, 240)
            rgb[~water] = (45, 48, 55)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_t = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(rgb_t)

    def _render_chunk_surface(self, chunk: ChunkTile, mode: str = "NATURAL") -> pygame.Surface:
        """Renders unscaled raster graphic for an individual chunk tile."""
        cw = chunk.cell_x1 - chunk.cell_x0
        ch = chunk.cell_y1 - chunk.cell_y0

        land_sub = self.layers.land_mask[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
        elev_sub = self.layers.elevation[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]

        rgb = np.zeros((ch, cw, 3), dtype=np.uint8)

        if mode == "NATURAL":
            biome_sub = self.layers.base_biome[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
            rivers_sub = self.layers.rivers_lakes[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]

            # Ocean
            ocean_depth = np.clip(1.0 - elev_sub / 0.50, 0.0, 1.0)
            rgb[..., 0] = (14 + 18 * (1.0 - ocean_depth)).astype(np.uint8)
            rgb[..., 1] = (38 + 48 * (1.0 - ocean_depth)).astype(np.uint8)
            rgb[..., 2] = (85 + 75 * (1.0 - ocean_depth)).astype(np.uint8)

            # Biomes on land
            for b_id, color in BIOME_COLORS.items():
                if b_id == 0:
                    continue
                m = land_sub & (biome_sub == b_id)
                if np.any(m):
                    rgb[m] = color

            # Altitude relief
            alt = np.maximum(0.0, elev_sub - 0.50) * 1.8
            if np.any(land_sub):
                shaded = rgb[land_sub].astype(np.float32) * (0.88 + 0.28 * alt[land_sub, np.newaxis])
                rgb[land_sub] = np.clip(shaded, 0, 255).astype(np.uint8)

            # Snow peaks
            snow = land_sub & (elev_sub > 0.80)
            if np.any(snow):
                rgb[snow] = np.clip(rgb[snow].astype(np.float32) * 0.4 + np.array([245, 248, 255]) * 0.6, 0, 255).astype(np.uint8)

            # Rivers & lakes
            water_mask = (rivers_sub > 0.45) & land_sub
            if np.any(water_mask):
                rgb[water_mask] = (42, 140, 168)

        elif mode == "ELEVATION":
            ocean_v = np.clip(elev_sub / 0.50, 0.0, 1.0) * 160.0
            rgb[..., 0] = np.where(land_sub, (elev_sub * 255.0).astype(np.uint8), 20)
            rgb[..., 1] = np.where(land_sub, (elev_sub * 230.0).astype(np.uint8), (30 + ocean_v * 0.4).astype(np.uint8))
            rgb[..., 2] = np.where(land_sub, (elev_sub * 190.0).astype(np.uint8), (70 + ocean_v * 0.8).astype(np.uint8))

        elif mode == "TEMPERATURE":
            temp_sub = self.layers.temperature[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
            r = np.clip((temp_sub - 0.45) * 3.0, 0.0, 1.0) * 255.0
            g = (np.sin(temp_sub * np.pi) * 230.0)
            b = np.clip((0.55 - temp_sub) * 3.0, 0.0, 1.0) * 255.0
            rgb[..., 0] = r.astype(np.uint8)
            rgb[..., 1] = g.astype(np.uint8)
            rgb[..., 2] = b.astype(np.uint8)

        elif mode == "PRECIPITATION":
            precip_sub = self.layers.precipitation[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
            rgb[..., 0] = ((1.0 - precip_sub * 0.7) * 190.0).astype(np.uint8)
            rgb[..., 1] = ((0.5 + precip_sub * 0.5) * 200.0).astype(np.uint8)
            rgb[..., 2] = ((0.3 + precip_sub * 0.7) * 245.0).astype(np.uint8)

        elif mode == "BIOMES":
            biome_sub = self.layers.base_biome[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
            for b_id, color in BIOME_COLORS.items():
                m = (biome_sub == b_id)
                rgb[m] = color
            rgb[~land_sub] = (15, 35, 65)

        elif mode == "WATER":
            rivers_sub = self.layers.rivers_lakes[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
            water = (~land_sub) | (rivers_sub > 0.45)
            rgb[water] = (46, 175, 240)
            rgb[~water] = (45, 48, 55)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_t = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(rgb_t)

    def get_chunk_surface(self, chunk: ChunkTile, mode: str = "NATURAL") -> pygame.Surface:
        """Retrieves or builds base unscaled surface for this chunk."""
        key = (chunk.key, mode)
        if key in self._unscaled_chunk_cache:
            surf = self._unscaled_chunk_cache[key]
            self.cache[chunk.key] = surf
            return surf

        surf = self._render_chunk_surface(chunk, mode)
        self._unscaled_chunk_cache[key] = surf
        self.cache[chunk.key] = surf
        return surf

    def get_scaled_chunk_surface(
        self,
        chunk: ChunkTile,
        lod: int,
        pixel_w: int,
        pixel_h: int,
        mode: str = "NATURAL",
    ) -> pygame.Surface:
        """
        Retrieves resolution-specific scaled chunk surface from LRU cache.
        Cached by (chunk.key, lod, pixel_w, pixel_h, mode).
        During camera panning, pixel dimensions remain identical, achieving 100% cache hit rate.
        Scale transformation executes ONLY on cache misses.
        """
        cache_key = (chunk.key, lod, pixel_w, pixel_h, mode)

        if cache_key in self._scaled_cache:
            self._scaled_cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self._scaled_cache[cache_key]

        # Cache Miss: Scale base surface once and cache
        self.cache_misses += 1
        self.rebuilds_this_frame += 1

        base_surf = self.get_chunk_surface(chunk, mode)
        scaled_surf = pygame.transform.scale(base_surf, (pixel_w, pixel_h))

        self._scaled_cache[cache_key] = scaled_surf

        # Memory-bounded LRU eviction
        if len(self._scaled_cache) > self.max_cached:
            self._scaled_cache.popitem(last=False)

        return scaled_surf

    def get_visible_chunks(self, min_wx: float, min_wy: float, max_wx: float, max_wy: float) -> list[ChunkTile]:
        """Viewport culling: returns only chunks that intersect the camera's visible world rectangle."""
        visible = []
        for chunk in self.chunks:
            if chunk.intersects(min_wx, min_wy, max_wx, max_wy):
                chunk.last_used_frame = self.current_frame
                visible.append(chunk)
        return visible

    def clear_scaled_cache(self):
        """Clears scaled cache (e.g., when layer mode changes)."""
        self._scaled_cache.clear()

    def step_frame(self):
        """Advances frame index and resets per-frame rebuild counter."""
        self.current_frame += 1
        self.rebuilds_this_frame = 0
