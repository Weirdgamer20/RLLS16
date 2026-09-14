"""
RLLS16 2D Map Layers.
Manages 13 independent, queryable simulation layers for the rectangular Earth:
Layer 0  — Ocean/Land
Layer 1  — Elevation
Layer 2  — Coastline
Layer 3  — Rivers/Lakes (Hydrology)
Layer 4  — Climate
Layer 5  — Temperature
Layer 6  — Precipitation
Layer 7  — Soil Moisture
Layer 8  — Biome
Layer 9  — Vegetation / Biomass
Layer 10 — Wildlife
Layer 11 — Resources
Layer 12 — Agents

Separates immutable geographic baseline from mutable simulation states.
"""

import math
from dataclasses import dataclass
import numpy as np


BIOME_NAMES = {
    0: "Ocean",
    1: "Polar Ice / Glaciers",
    2: "Tundra / Alpine",
    3: "Boreal Taiga",
    4: "Temperate Broadleaf Forest",
    5: "Temperate Grassland",
    6: "Mediterranean Scrub",
    7: "Tropical Dry Forest",
    8: "Tropical Rainforest",
    9: "Tropical Savanna",
    10: "Desert / Xeric Shrubland",
    11: "Freshwater Wetlands / Lakes",
    12: "Coastal & Reef",
    13: "Highland Meadow",
}

# 16-bit inspired crisp palette for biomes
BIOME_COLORS = {
    0: (16, 44, 98),       # Ocean (deep blue)
    1: (230, 240, 248),    # Ice (crisp snow white)
    2: (158, 168, 142),    # Tundra (gray-green)
    3: (38, 92, 64),       # Taiga (deep pine green)
    4: (56, 138, 62),      # Temperate Forest (rich leaf green)
    5: (142, 182, 82),     # Grassland (warm olive green)
    6: (172, 162, 88),     # Mediterranean (olive scrub)
    7: (108, 148, 72),     # Dry Forest
    8: (24, 115, 52),      # Rainforest (dense emerald green)
    9: (204, 178, 88),     # Savanna (golden grassland)
    10: (218, 188, 118),   # Desert (sand yellow)
    11: (42, 140, 168),    # Wetlands/Lakes (cyan-blue)
    12: (58, 175, 195),    # Reef
    13: (132, 126, 118),   # Highland (rock gray)
}


class MapLayers:
    """
    13 Independent Simulation Layers for 2D Rectangular Earth.
    """

    def __init__(self, world_dict: dict):
        self.height, self.width = world_dict["elevation"].shape
        H, W = self.height, self.width

        # Coordinate axes
        self.latitude = world_dict["latitude"].astype(np.float32)
        self.longitude = world_dict["longitude"].astype(np.float32)
        self.dlat = float(self.latitude[1] - self.latitude[0])
        self.dlon = float(self.longitude[1] - self.longitude[0])

        # --- IMMUTABLE GEOGRAPHIC BASELINE ---
        self.land_mask: np.ndarray = world_dict["land_mask"].astype(bool)          # Layer 0
        self.elevation: np.ndarray = world_dict["elevation"].astype(np.float32)    # Layer 1
        self.coastline: np.ndarray = world_dict.get("coastline", np.zeros((H, W), dtype=bool)).astype(bool) # Layer 2
        self.rivers_lakes: np.ndarray = world_dict.get("rivers_lakes", world_dict.get("hydrology", np.zeros((H, W), dtype=np.float32))).astype(np.float32) # Layer 3
        self.base_climate: np.ndarray = world_dict.get("climate", np.zeros((H, W), dtype=np.uint8)).astype(np.uint8) # Layer 4
        self.base_biome: np.ndarray = world_dict["biome"].astype(np.uint8)         # Layer 8

        # --- MUTABLE SIMULATION STATES ---
        self.temperature: np.ndarray = world_dict["temperature"].astype(np.float32).copy()      # Layer 5
        self.precipitation: np.ndarray = world_dict["precipitation"].astype(np.float32).copy()  # Layer 6
        self.soil_moisture: np.ndarray = world_dict.get("soil_moisture", world_dict["precipitation"]).astype(np.float32).copy() # Layer 7
        self.vegetation: np.ndarray = world_dict.get("vegetation", np.zeros((H, W), dtype=np.float32)).astype(np.float32).copy() # Layer 9
        self.wildlife: np.ndarray = world_dict.get("wildlife", self.vegetation * 0.8).astype(np.float32).copy() # Layer 10
        self.resources: np.ndarray = world_dict.get("resources", np.clip(self.vegetation + self.rivers_lakes, 0.0, 1.0)).astype(np.float32).copy() # Layer 11
        self.agents: np.ndarray = np.zeros((H, W), dtype=np.float32)               # Layer 12

        self.metadata = world_dict.get("metadata", {})

    def get_metadata(self) -> dict:
        return self.metadata

    def _coord_to_indices(self, nx: float, ny: float) -> tuple[int, int, float, float]:
        """Convert normalized [0, 1] map coordinates to row/col and fractional subpixels."""
        fx = np.clip(nx, 0.0, 1.0) * (self.width - 1)
        fy = np.clip(ny, 0.0, 1.0) * (self.height - 1)
        c0 = int(math.floor(fx))
        r0 = int(math.floor(fy))
        c1 = min(self.width - 1, c0 + 1)
        r1 = min(self.height - 1, r0 + 1)
        tx = fx - c0
        ty = fy - r0
        return r0, c0, r1, c1, tx, ty

    def _sample_bilinear(self, field: np.ndarray, nx: float, ny: float) -> float:
        r0, c0, r1, c1, tx, ty = self._coord_to_indices(nx, ny)
        top = field[r0, c0] * (1.0 - tx) + field[r0, c1] * tx
        bot = field[r1, c0] * (1.0 - tx) + field[r1, c1] * tx
        return float(top * (1.0 - ty) + bot * ty)

    def _sample_nearest(self, field: np.ndarray, nx: float, ny: float):
        r = int(np.clip(round(ny * (self.height - 1)), 0, self.height - 1))
        c = int(np.clip(round(nx * (self.width - 1)), 0, self.width - 1))
        return field[r, c]

    # --- Query API ---

    def is_land(self, nx: float, ny: float) -> bool:
        """Returns True if the coordinate is Land, False if Ocean."""
        return bool(self._sample_nearest(self.land_mask, nx, ny))

    def get_elevation(self, nx: float, ny: float) -> float:
        """Returns normalized elevation [0.0, 1.0]. Sea level is at 0.50."""
        return self._sample_bilinear(self.elevation, nx, ny)

    def get_elevation_meters(self, nx: float, ny: float) -> float:
        """Returns physical elevation in meters above sea level (-10000m to +8500m)."""
        elev = self.get_elevation(nx, ny)
        if elev >= 0.50:
            return (elev - 0.50) * 17000.0
        else:
            return (elev - 0.50) * 20000.0

    def get_temperature(self, nx: float, ny: float) -> float:
        """Returns normalized temperature [0.0, 1.0]."""
        return self._sample_bilinear(self.temperature, nx, ny)

    def get_temperature_celsius(self, nx: float, ny: float) -> float:
        """Returns temperature in Celsius (-45°C to +45°C)."""
        norm_t = self.get_temperature(nx, ny)
        return norm_t * 90.0 - 45.0

    def get_precipitation(self, nx: float, ny: float) -> float:
        """Returns normalized precipitation [0.0, 1.0]."""
        return self._sample_bilinear(self.precipitation, nx, ny)

    def get_water(self, nx: float, ny: float) -> float:
        """Returns hydrology / water presence index [0.0, 1.0]."""
        return self._sample_bilinear(self.rivers_lakes, nx, ny)

    def get_soil(self, nx: float, ny: float) -> float:
        """Returns soil moisture index [0.0, 1.0]."""
        return self._sample_bilinear(self.soil_moisture, nx, ny)

    def get_biome(self, nx: float, ny: float) -> int:
        """Returns integer biome classification (0..13)."""
        return int(self._sample_nearest(self.base_biome, nx, ny))

    def get_biome_name(self, nx: float, ny: float) -> str:
        b_id = self.get_biome(nx, ny)
        return BIOME_NAMES.get(b_id, "Unknown")

    def get_vegetation(self, nx: float, ny: float) -> float:
        """Returns vegetation biomass density [0.0, 1.0]."""
        return self._sample_bilinear(self.vegetation, nx, ny)

    def get_wildlife(self, nx: float, ny: float) -> float:
        """Returns wildlife population index [0.0, 1.0]."""
        return self._sample_bilinear(self.wildlife, nx, ny)

    def get_resources(self, nx: float, ny: float) -> float:
        """Returns natural resources index [0.0, 1.0]."""
        return self._sample_bilinear(self.resources, nx, ny)

    # Conversions between Normalized Map [0, 1] and Lat/Lon degrees
    @staticmethod
    def latlon_to_normalized(lat_deg: float, lon_deg: float) -> tuple[float, float]:
        nx = (lon_deg + 180.0) / 360.0
        ny = (90.0 - lat_deg) / 180.0
        return np.clip(nx, 0.0, 1.0), np.clip(ny, 0.0, 1.0)

    @staticmethod
    def normalized_to_latlon(nx: float, ny: float) -> tuple[float, float]:
        lon_deg = nx * 360.0 - 180.0
        lat_deg = 90.0 - ny * 180.0
        return lat_deg, lon_deg
