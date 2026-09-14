"""
RLLS16 2D Spatial Chunk & Tile Cache System.
Partitions the rectangular 2D Earth into spatial chunks for viewport culling,
hierarchical LOD rendering, and memory-bounded surface caching.
"""

from collections import OrderedDict
import math
import pygame
import numpy as np
from .layers import MapLayers, BIOME_COLORS
from ..assets_loader import get_icon


class ChunkTile:
    """A spatial 2D tile representing a slice of the rectangular Earth."""
    __slots__ = (
        'chunk_x', 'chunk_y', 'cell_x0', 'cell_y0', 'cell_x1', 'cell_y1',
        'world_x0', 'world_y0', 'world_x1', 'world_y1',
        'surface', 'key', 'last_used_frame'
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
        self.surface: pygame.Surface | None = None
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
    """Manages 2D spatial chunks, raster surface generation, and LRU cache."""

    def __init__(self, layers: MapLayers, chunk_size: int = 32, max_cached: int = 512):
        self.layers = layers
        self.chunk_size = chunk_size
        self.max_cached = max_cached
        self.current_frame = 0

        self.total_h = layers.height
        self.total_w = layers.width

        self.num_chunks_x = int(math.ceil(self.total_w / chunk_size))
        self.num_chunks_y = int(math.ceil(self.total_h / chunk_size))

        self.chunks: list[ChunkTile] = []
        for cy in range(self.num_chunks_y):
            for cx in range(self.num_chunks_x):
                x0 = cx * chunk_size
                y0 = cy * chunk_size
                x1 = min(self.total_w, x0 + chunk_size)
                y1 = min(self.total_h, y0 + chunk_size)
                self.chunks.append(ChunkTile(cx, cy, x0, y0, x1, y1, self.total_w, self.total_h))

        # Pre-render a full macro world surface for distant zoom
        self.macro_surface = self._build_macro_surface()
        self.cache: OrderedDict[str, pygame.Surface] = OrderedDict()

    def _build_macro_surface(self) -> pygame.Surface:
        """Renders the entire 2D Earth into a crisp 16-bit macro overview surface."""
        w, h = self.total_w, self.total_h
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        land = self.layers.land_mask
        elev = self.layers.elevation
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

        # Burn major rivers and lakes
        water_mask = (rivers > 0.45) & land
        if np.any(water_mask):
            rgb[water_mask] = (42, 140, 168)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # Transpose for Pygame: shape (width, height, 3)
        rgb_t = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(rgb_t)

    def _render_chunk_surface(self, chunk: ChunkTile) -> pygame.Surface:
        """Renders raster graphic for an individual chunk tile."""
        cw = chunk.cell_x1 - chunk.cell_x0
        ch = chunk.cell_y1 - chunk.cell_y0

        land_sub = self.layers.land_mask[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
        elev_sub = self.layers.elevation[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
        biome_sub = self.layers.base_biome[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]
        rivers_sub = self.layers.rivers_lakes[chunk.cell_y0:chunk.cell_y1, chunk.cell_x0:chunk.cell_x1]

        rgb = np.zeros((ch, cw, 3), dtype=np.uint8)

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

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_t = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(rgb_t)

    def get_chunk_surface(self, chunk: ChunkTile) -> pygame.Surface:
        """Retrieves or builds cached surface for this chunk with LRU eviction."""
        if chunk.key in self.cache:
            self.cache.move_to_end(chunk.key)
            return self.cache[chunk.key]

        surf = self._render_chunk_surface(chunk)
        self.cache[chunk.key] = surf

        if len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)

        return surf

    def get_visible_chunks(self, min_wx: float, min_wy: float, max_wx: float, max_wy: float) -> list[ChunkTile]:
        """Viewport culling: returns only chunks that intersect the camera's visible world rectangle."""
        visible = []
        for chunk in self.chunks:
            if chunk.intersects(min_wx, min_wy, max_wx, max_wy):
                visible.append(chunk)
        return visible

    def step_frame(self):
        self.current_frame += 1
