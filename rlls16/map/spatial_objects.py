"""
RLLS16 Spatial Object Layer.
Separates canonical continuous raster data from discrete renderable simulation objects.
Implements spatial partitioning for vegetation (trees), wildlife herds, freshwater springs,
and resource deposits, enabling high-performance O(K) querying at deep zoom without raster scanning.
"""

import math
from dataclasses import dataclass
import numpy as np

from .layers import MapLayers


@dataclass(slots=True)
class SpatialObject:
    """Discrete renderable ecological or geographical entity."""
    obj_type: str        # 'tree', 'animal', 'water', 'resource'
    wx: float            # Normalized world x [0.0, 1.0]
    wy: float            # Normalized world y [0.0, 0.5]
    biome_id: int
    size: tuple[int, int]
    variant: int = 0


class SpatialObjectManager:
    """
    Spatial partitioning index for discrete ecological objects.
    Pre-generates deterministic entity distributions from layer fields once,
    and indexes them per chunk cell for instant viewport retrieval.
    """

    def __init__(self, layers: MapLayers, chunk_cols: int = 16, chunk_rows: int = 8, seed: int = 16001):
        self.layers = layers
        self.chunk_cols = chunk_cols
        self.chunk_rows = chunk_rows
        self.seed = seed

        # Spatial grid bins: (chunk_c, chunk_r) -> list[SpatialObject]
        self.grid: dict[tuple[int, int], list[SpatialObject]] = {}
        self._total_objects = 0

        self._build_spatial_index()

    @property
    def total_objects(self) -> int:
        return self._total_objects

    def _build_spatial_index(self):
        """Deterministically populate spatial objects from canonical layers."""
        rng = np.random.RandomState(self.seed)

        H, W = self.layers.height, self.layers.width
        col_step = W // self.chunk_cols
        row_step = H // self.chunk_rows

        for cr in range(self.chunk_rows):
            for cc in range(self.chunk_cols):
                chunk_key = (cc, cr)
                objects: list[SpatialObject] = []

                c0 = cc * col_step
                c1 = min(W, (cc + 1) * col_step)
                r0 = cr * row_step
                r1 = min(H, (cr + 1) * row_step)

                # Subsample grid cells within chunk
                for r in range(r0, r1, 2):
                    for c in range(c0, c1, 2):
                        if not self.layers.land_mask[r, c]:
                            continue

                        veg = float(self.layers.vegetation[r, c])
                        wild = float(self.layers.wildlife[r, c])
                        biome = int(self.layers.base_biome[r, c])
                        river = float(self.layers.rivers_lakes[r, c])

                        # Base normalized coords
                        base_wx = c / float(W)
                        base_wy = (r / float(H)) * 0.5

                        # 1. Vegetation / Trees
                        # High biomass generates trees with jitter
                        if veg > 0.45:
                            num_trees = 1 if veg < 0.70 else 2
                            for i in range(num_trees):
                                jx = (rng.uniform(-0.4, 0.4)) / float(W)
                                jy = (rng.uniform(-0.4, 0.4) / float(H)) * 0.5
                                sz = (20, 20) if veg < 0.75 else (24, 24)
                                objects.append(
                                    SpatialObject(
                                        obj_type="tree",
                                        wx=max(0.0, min(1.0, base_wx + jx)),
                                        wy=max(0.0, min(0.5, base_wy + jy)),
                                        biome_id=biome,
                                        size=sz,
                                        variant=rng.randint(0, 4),
                                    )
                                )

                        # 2. Wildlife Herds
                        if wild > 0.65 and rng.rand() < 0.35:
                            jx = (rng.uniform(-0.3, 0.3)) / float(W)
                            jy = (rng.uniform(-0.3, 0.3) / float(H)) * 0.5
                            objects.append(
                                SpatialObject(
                                    obj_type="animal",
                                    wx=max(0.0, min(1.0, base_wx + jx)),
                                    wy=max(0.0, min(0.5, base_wy + jy)),
                                    biome_id=biome,
                                    size=(20, 20),
                                    variant=rng.randint(0, 3),
                                )
                            )

                        # 3. Freshwater Springs / River nodes
                        if river > 0.55 and rng.rand() < 0.25:
                            objects.append(
                                SpatialObject(
                                    obj_type="water",
                                    wx=base_wx,
                                    wy=base_wy,
                                    biome_id=biome,
                                    size=(16, 16),
                                    variant=0,
                                )
                            )

                self.grid[chunk_key] = objects
                self._total_objects += len(objects)

    def get_objects_in_chunk(self, chunk_c: int, chunk_r: int) -> list[SpatialObject]:
        """Get pre-indexed objects for a specific chunk in O(1) time."""
        return self.grid.get((chunk_c, chunk_r), [])

    def get_objects_in_world_rect(self, min_wx: float, min_wy: float, max_wx: float, max_wy: float) -> list[SpatialObject]:
        """
        Query all spatial objects inside visible world rectangle.
        Operates in O(Visible Chunks) time rather than O(World Pixels).
        """
        c0 = max(0, int(math.floor(min_wx * self.chunk_cols)))
        c1 = min(self.chunk_cols - 1, int(math.floor(max_wx * self.chunk_cols)))
        r0 = max(0, int(math.floor((min_wy / 0.5) * self.chunk_rows)))
        r1 = min(self.chunk_rows - 1, int(math.floor((max_wy / 0.5) * self.chunk_rows)))

        visible_objs = []
        for cr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                objs = self.grid.get((cc, cr), [])
                # Filter precisely inside rect
                for o in objs:
                    if min_wx <= o.wx <= max_wx and min_wy <= o.wy <= max_wy:
                        visible_objs.append(o)

        return visible_objs
