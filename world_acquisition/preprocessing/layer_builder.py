"""
RLLS16 Canonical Earth Layer Builder.
Constructs the 13 canonical simulation layers from rasterized geographic baselines.
"""

from dataclasses import dataclass
import math
import numpy as np

@dataclass
class CanonicalEarthLayers:
    latitude: np.ndarray          # 1D latitude coordinates [-pi/2, pi/2] (H,)
    longitude: np.ndarray         # 1D longitude coordinates [-pi, pi] (W,)
    land_mask: np.ndarray         # Layer 0: bool [H, W]
    elevation: np.ndarray         # Layer 1: float32 [H, W]
    coastline: np.ndarray         # Layer 2: bool [H, W]
    rivers_lakes: np.ndarray      # Layer 3: float32 [H, W]
    climate: np.ndarray           # Layer 4: uint8 [H, W]
    temperature: np.ndarray       # Layer 5: float32 [H, W]
    precipitation: np.ndarray     # Layer 6: float32 [H, W]
    soil_moisture: np.ndarray     # Layer 7: float32 [H, W]
    biome: np.ndarray             # Layer 8: uint8 [H, W]
    vegetation: np.ndarray        # Layer 9: float32 [H, W]
    wildlife: np.ndarray          # Layer 10: float32 [H, W]
    resources: np.ndarray         # Layer 11: float32 [H, W]
    agents: np.ndarray            # Layer 12: float32 [H, W]

def build_canonical_layers(
    land_mask: np.ndarray,
    coastline_mask: np.ndarray,
    rivers_mask: np.ndarray,
    lakes_mask: np.ndarray,
    seed: int = 16001,
) -> CanonicalEarthLayers:
    """
    Builds the 13 canonical deterministic simulation layers.
    """
    H, W = land_mask.shape
    lat = np.linspace(-np.pi / 2, np.pi / 2, H, dtype=np.float32)
    lon = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)

    lat_grid = np.repeat(lat[:, np.newaxis], W, axis=1)
    lon_grid = np.repeat(lon[np.newaxis, :], H, axis=0)
    lat_deg = lat_grid * (180.0 / np.pi)
    lon_deg = lon_grid * (180.0 / np.pi)

    # 1. Real Mountain Cordilleras & Orography
    # Coordinates of major mountain belts:
    # Himalayas / Tibetan Plateau: lat ~ 27..38, lon ~ 75..105
    himalayas = np.exp(-((lat_deg - 32.0) / 7.0) ** 2 - ((lon_deg - 88.0) / 16.0) ** 2) * 0.42
    # Andes: lat ~ -55..10, lon ~ -78..-65 (narrow north-south)
    andes = np.exp(-((lon_deg + 72.0) / 4.0) ** 2) * np.clip((lat_deg + 55.0) / 65.0, 0.0, 1.0) * 0.35
    andes *= np.exp(-((lat_deg - (-20.0)) / 35.0) ** 2)
    # Rockies & Coast Ranges: lat ~ 30..65, lon ~ -125..-105
    rockies = np.exp(-((lon_deg + 115.0) / 6.0) ** 2) * np.clip((lat_deg - 30.0) / 35.0, 0.0, 1.0) * 0.30
    rockies *= np.exp(-((lat_deg - 48.0) / 20.0) ** 2)
    # Alps: lat ~ 45..48, lon ~ 5..16
    alps = np.exp(-((lat_deg - 46.5) / 3.0) ** 2 - ((lon_deg - 10.0) / 6.0) ** 2) * 0.28
    # Urals: lat ~ 50..68, lon ~ 58..62
    urals = np.exp(-((lon_deg - 60.0) / 3.0) ** 2) * np.clip((lat_deg - 50.0) / 20.0, 0.0, 1.0) * 0.18
    # Ethiopian Highlands: lat ~ 5..15, lon ~ 35..42
    ethiopia = np.exp(-((lat_deg - 10.0) / 5.0) ** 2 - ((lon_deg - 39.0) / 5.0) ** 2) * 0.25
    # Greenland / Antarctica Ice Massifs
    greenland = np.exp(-((lat_deg - 72.0) / 8.0) ** 2 - ((lon_deg + 40.0) / 15.0) ** 2) * 0.30
    antarctica_elev = np.clip((-lat_deg - 65.0) / 25.0, 0.0, 1.0) * 0.38

    mountain_relief = himalayas + andes + rockies + alps + urals + ethiopia + greenland + antarctica_elev

    # 2. Elevation Field [0.0, 1.0] (Sea level = 0.50)
    elevation = np.zeros((H, W), dtype=np.float32)
    # Base land elevation: 0.52 to 0.95
    base_land_elev = 0.53 + 0.08 * (1.0 - np.abs(lat_deg) / 90.0)
    elevation[land_mask] = np.clip(base_land_elev[land_mask] + mountain_relief[land_mask], 0.505, 0.95)

    # Ocean bathymetry: 0.08 to 0.495 (deep ocean away from coasts, shelf near coasts)
    dist_to_coast_approx = np.where(land_mask, 1.0, 0.0)
    # Shallow continental shelf near coastline
    shelf = coastline_mask.astype(np.float32) * 0.15
    ocean_elev = 0.20 + shelf
    elevation[~land_mask] = np.clip(ocean_elev[~land_mask], 0.08, 0.495)

    altitude = np.maximum(0.0, elevation - 0.50)

    # 3. Rivers & Lakes (Layer 3)
    hydrology = np.zeros((H, W), dtype=np.float32)
    hydrology[rivers_mask] = 0.85
    hydrology[lakes_mask] = 0.95
    # Drainage gradient: rivers follow slope towards ocean
    grad_y, grad_x = np.gradient(elevation)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    hydrology = np.clip(hydrology + np.clip(slope * 0.2, 0.0, 0.3), 0.0, 1.0)

    # 4. Temperature Field [0.0, 1.0] (Layer 5)
    # Normalized: 0.0 ~ -45°C, 0.50 ~ +15°C, 1.0 ~ +45°C
    lat_factor = 1.0 - (np.abs(lat_deg) / 90.0) ** 1.35
    lapse_rate = altitude * 0.45  # Mountain cooling
    ocean_thermal_moderation = np.where(land_mask, 0.0, 0.06 * (1.0 - np.abs(lat_deg) / 90.0))
    temperature = np.clip(lat_factor * 0.85 - lapse_rate + ocean_thermal_moderation + 0.08, 0.02, 0.98)

    # 5. Precipitation Field [0.0, 1.0] (Layer 6)
    # ITCZ equatorial belt (0..10 lat)
    itcz = np.exp(-((lat_deg) / 10.0) ** 2) * 0.45
    # Subtropical dry belts (Sahara, Kalahari, Australian interior ~ 18..32 lat)
    subtropical_dry = np.exp(-((np.abs(lat_deg) - 24.0) / 8.0) ** 2) * 0.35
    # Mid-latitude storm tracks (40..60 lat)
    midlat_wet = np.exp(-((np.abs(lat_deg) - 50.0) / 10.0) ** 2) * 0.25
    # Orographic rain on windward mountain slopes, rain shadow on leeward
    orographic_precip = np.clip(mountain_relief * 0.4, 0.0, 0.3)

    precip_base = 0.35 + itcz - subtropical_dry + midlat_wet + orographic_precip
    precipitation = np.clip(precip_base, 0.05, 0.95)

    # 6. Soil Moisture (Layer 7)
    soil = np.zeros((H, W), dtype=np.float32)
    soil[land_mask] = np.clip(precipitation[land_mask] * 0.75 + hydrology[land_mask] * 0.25, 0.02, 1.0)
    soil[~land_mask] = 1.0

    # 7. WWF 14 Biome Classification (Layer 8)
    # 0: Ocean
    # 1: Polar Ice / Glaciers
    # 2: Tundra / Alpine
    # 3: Boreal Forest / Taiga
    # 4: Temperate Broadleaf Forest
    # 5: Temperate Grasslands / Steppe
    # 6: Mediterranean Forest & Scrub
    # 7: Tropical Dry Broadleaf Forest
    # 8: Tropical Rainforest (Moist Broadleaf)
    # 9: Tropical Savanna / Grasslands
    # 10: Deserts & Xeric Shrublands
    # 11: Mangroves & Wetlands
    # 12: Coastal & Reef
    # 13: Mountain Meadow / Highland
    biome = np.zeros((H, W), dtype=np.uint8)

    # Default land cells to 5 (Grassland)
    biome[land_mask] = 5

    # Polar ice: poles or extreme cold
    biome[land_mask & (temperature < 0.18)] = 1
    # Tundra: cold northern / alpine latitudes
    biome[land_mask & (temperature >= 0.18) & (temperature < 0.32)] = 2
    # Boreal Taiga: subpolar forested zone
    biome[land_mask & (temperature >= 0.32) & (temperature < 0.45) & (precipitation >= 0.30)] = 3
    # Deserts: low precipitation
    biome[land_mask & (temperature >= 0.35) & (precipitation < 0.22)] = 10
    # Temperate Forest: moderate temp, moderate/high precip
    biome[land_mask & (temperature >= 0.45) & (temperature < 0.65) & (precipitation >= 0.42)] = 4
    # Mediterranean: mid-latitude, warm, moderate precip
    biome[land_mask & (np.abs(lat_deg) >= 30.0) & (np.abs(lat_deg) <= 45.0) & (temperature >= 0.55) & (precipitation >= 0.25) & (precipitation < 0.45)] = 6
    # Tropical Rainforest: high temp, high precip
    biome[land_mask & (temperature >= 0.65) & (precipitation >= 0.60)] = 8
    # Tropical Savanna: high temp, seasonal precip
    biome[land_mask & (temperature >= 0.65) & (precipitation >= 0.32) & (precipitation < 0.60)] = 9
    # High altitude alpine / highland
    biome[land_mask & (altitude > 0.20)] = 13
    # Lakes / inland freshwater bodies
    biome[land_mask & lakes_mask] = 11

    # 8. Climate Zones (Layer 4)
    # 0: Ocean, 1: Polar, 2: Boreal, 3: Temperate, 4: Subtropical Arid, 5: Tropical
    climate = np.zeros((H, W), dtype=np.uint8)
    climate[land_mask & (temperature < 0.25)] = 1
    climate[land_mask & (temperature >= 0.25) & (temperature < 0.45)] = 2
    climate[land_mask & (temperature >= 0.45) & (temperature < 0.65)] = 3
    climate[land_mask & (temperature >= 0.50) & (precipitation < 0.25)] = 4
    climate[land_mask & (temperature >= 0.65)] = 5

    # 9. Vegetation Biomass (Layer 9)
    temp_suit = np.exp(-((temperature - 0.62) ** 2) / 0.15)
    water_suit = np.clip((soil - 0.15) / 0.70, 0.0, 1.0)
    vegetation = np.zeros((H, W), dtype=np.float32)
    vegetation[land_mask] = np.clip(temp_suit[land_mask] * water_suit[land_mask] * 1.3, 0.0, 1.0)

    # 10. Wildlife Carrying Capacity (Layer 10)
    wildlife = np.zeros((H, W), dtype=np.float32)
    wildlife[land_mask] = np.clip(vegetation[land_mask] * 0.70 + soil[land_mask] * 0.30, 0.0, 1.0)
    # Exclude extreme ice sheets
    wildlife[biome == 1] *= 0.05

    # 11. Resources Distribution (Layer 11)
    resources = np.zeros((H, W), dtype=np.float32)
    # Fresh water, minerals in mountains, timber in forests, food in savannas/grasslands
    mineral_bonus = np.clip(mountain_relief * 0.6, 0.0, 0.4)
    timber_bonus = np.where((biome == 3) | (biome == 4) | (biome == 8), 0.35, 0.0)
    food_bonus = vegetation * 0.35
    water_bonus = hydrology * 0.30
    resources[land_mask] = np.clip(mineral_bonus[land_mask] + timber_bonus[land_mask] + food_bonus[land_mask] + water_bonus[land_mask], 0.05, 1.0)

    # 12. Agents Initial Spatial Occupancy (Layer 12)
    agents = np.zeros((H, W), dtype=np.float32)

    return CanonicalEarthLayers(
        latitude=lat,
        longitude=lon,
        land_mask=land_mask,
        elevation=elevation,
        coastline=coastline_mask,
        rivers_lakes=hydrology,
        climate=climate,
        temperature=temperature,
        precipitation=precipitation,
        soil_moisture=soil,
        biome=biome,
        vegetation=vegetation,
        wildlife=wildlife,
        resources=resources,
        agents=agents,
    )
