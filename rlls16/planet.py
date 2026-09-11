from dataclasses import dataclass
import math
import numpy as np
from .noise import fbm_2d, ridged_fbm


@dataclass
class PlanetData:
    latitude: np.ndarray          # 1D latitude coordinates [-pi/2, pi/2]
    longitude: np.ndarray         # 1D longitude coordinates [-pi, pi]
    elevation: np.ndarray         # 2D normalized elevation [0.0, 1.0]
    land_mask: np.ndarray         # 2D boolean mask (True = Land, False = Ocean)
    temperature: np.ndarray       # 2D normalized temperature [0.0, 1.0]
    precipitation: np.ndarray     # 2D normalized precipitation [0.0, 1.0]
    biome: np.ndarray             # 2D uint8 biome classifications (0..6)
    hydrology: np.ndarray         # 2D normalized water flow/drainage potential [0.0, 1.0]
    vegetation: np.ndarray        # 2D normalized biomass/vegetation potential [0.0, 1.0]


def _normalize(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def generate_planet(cfg) -> PlanetData:
    """
    Generate authoritative canonical Earth fields:
    - 29.2% Land / 70.8% Ocean ratio strictly calibrated to Earth
    - Major Earth-like continental configurations (Eurasia-Africa, Americas, Antarctica, Australia)
    - Major mountain cordilleras (Himalayas, Andes, Rockies)
    - Realistic Earth climatic latitudinal zones (polar caps, subtropical deserts, equatorial rainforests)
    - Strict invariant: Land cells never receive ocean biomes.
    """
    lat = np.linspace(-np.pi / 2, np.pi / 2, cfg.lat_samples, dtype=np.float32)
    lon = np.linspace(-np.pi, np.pi, cfg.lon_samples, endpoint=False, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 3D Cartesian coordinates on unit sphere
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.sin(lon_grid)

    lat_deg = lat_grid * (180.0 / np.pi)
    lon_deg = lon_grid * (180.0 / np.pi)

    # 1. Earth Continental Foundation
    # Calibrated spherical basis functions for Earth's actual major landmasses:
    # A. Antarctica (South Pole landmass below -65 lat)
    antarctica = np.clip((-(lat_deg + 62.0) / 18.0), 0.0, 1.0) * 0.85

    # B. Eurasia & Africa landmass (lon ~ -20 to 140, lat ~ -35 to 75)
    afro_eurasia_dist = np.sqrt(((lon_deg - 50.0) / 75.0) ** 2 + ((lat_deg - 30.0) / 45.0) ** 2)
    afro_eurasia = np.clip(1.35 - afro_eurasia_dist, 0.0, 1.0)

    # C. Americas landmass (lon ~ -120 to -40, lat ~ -55 to 70)
    americas_dist = np.sqrt(((lon_deg + 80.0) / 32.0) ** 2 + ((lat_deg - 10.0) / 60.0) ** 2)
    americas = np.clip(1.25 - americas_dist, 0.0, 1.0)

    # D. Australia (lon ~ 135, lat ~ -25)
    australia_dist = np.sqrt(((lon_deg - 135.0) / 20.0) ** 2 + ((lat_deg + 25.0) / 16.0) ** 2)
    australia = np.clip(1.20 - australia_dist, 0.0, 1.0)

    macro_continents = np.maximum.reduce([antarctica, afro_eurasia, americas, australia])

    # 2. Add organic continental coastline fractal noise (fBm)
    coast_noise = fbm_2d(
        x * 1.6 + z * 0.4,
        y * 1.6 - z * 0.2,
        seed=cfg.seed + 10,
        base_grid=1.4,
        octaves=4,
    )

    # 3. Mountain ridges (Himalayas, Andes, Rockies belts)
    # Belt 1: Alpine-Himalayan belt (lat ~ 25..45, lon ~ 10..100)
    himalaya_belt = np.exp(-((lat_deg - 32.0) / 10.0) ** 2 - ((lon_deg - 80.0) / 40.0) ** 2)
    # Belt 2: American cordillera (lon ~ -115..-70, North-South oriented)
    cordillera_belt = np.exp(-((lon_deg + 95.0) / 18.0) ** 2) * np.clip(1.0 - np.abs(lat_deg) / 65.0, 0.0, 1.0)

    ridge_noise = ridged_fbm(
        x * 2.5 + z * 0.5,
        y * 2.5 - x * 0.3,
        seed=cfg.seed + 30,
        octaves=4,
    )
    mountains = (himalaya_belt * 0.35 + cordillera_belt * 0.25) * (ridge_noise * 0.8 + 0.2)

    # 4. Erosion and valley formations
    erosion = fbm_2d(
        x * 2.2 - y * 0.3,
        z * 2.2 + y * 0.2,
        seed=cfg.seed + 20,
        base_grid=2.2,
        octaves=4,
    )

    # 5. Composite raw density
    composite = (
        macro_continents * 0.55
        + coast_noise * 0.30
        + mountains * 0.25
        - erosion * 0.10
    )

    # 6. Strict Land-Ocean calibration to ~29.2% land
    # Target: 29.2% of cells are above sea level (elevation 0.50)
    target_land_pct = 29.2
    # Threshold that separates ocean and land
    sea_level = 0.50
    # Calculate exact percentile for 70.8% ocean
    cutoff = float(np.percentile(composite, 100.0 - target_land_pct))

    # Transform composite so that cutoff maps cleanly to 0.50
    # Below cutoff: ocean bathymetry mapped to [0.05, 0.499]
    # Above cutoff: land elevation mapped to [0.501, 0.95]
    elevation = np.zeros_like(composite, dtype=np.float32)
    ocean_mask = composite < cutoff
    land_mask = ~ocean_mask

    # Normalized ocean depths [0.05, 0.49]
    c_ocean = composite[ocean_mask]
    if len(c_ocean) > 0:
        o_min, o_max = float(np.min(c_ocean)), float(np.max(c_ocean))
        if o_max - o_min > 1e-6:
            elevation[ocean_mask] = 0.08 + 0.41 * ((c_ocean - o_min) / (o_max - o_min)) ** 1.3
        else:
            elevation[ocean_mask] = 0.25

    # Normalized land elevations [0.50, 0.95]
    c_land = composite[land_mask]
    if len(c_land) > 0:
        l_min, l_max = float(np.min(c_land)), float(np.max(c_land))
        if l_max - l_min > 1e-6:
            elevation[land_mask] = 0.50 + 0.45 * ((c_land - l_min) / (l_max - l_min)) ** 1.5
        else:
            elevation[land_mask] = 0.65

    # Recompute strict land_mask
    land_mask = elevation >= sea_level
    altitude = np.maximum(elevation - sea_level, 0.0)

    # 7. Earth Temperature Field
    # Insolation by latitude + altitude cooling lapse rate + atmospheric noise
    lat_abs = np.abs(lat_grid) / (np.pi / 2.0) # 0 at equator, 1 at poles
    base_temp = 1.0 - 0.82 * (lat_abs ** 1.3)
    base_temp -= 0.38 * altitude # Mountain lapse rate
    base_temp += 0.04 * fbm_2d(x * 1.5, z * 1.5, seed=cfg.seed + 50, base_grid=2.0, octaves=3)
    temperature = np.clip(base_temp, 0.0, 1.0)

    # 8. Earth Precipitation Field
    # ITCZ equatorial belt + Subtropical dry belts (Sahara, Kalahari, Australian interior) + Midlat storm tracks
    equator_belt = np.exp(-((lat_abs) ** 2) / 0.06) * 0.42
    subtropical_dry = np.exp(-((lat_abs - 0.30) ** 2) / 0.025) * 0.32
    midlat_wet = np.exp(-((lat_abs - 0.58) ** 2) / 0.04) * 0.26

    circ_precip = 0.42 + equator_belt - subtropical_dry + midlat_wet
    circ_precip += 0.12 * fbm_2d(x * 1.8, z * 1.8, seed=cfg.seed + 60, base_grid=1.8, octaves=3)
    circ_precip -= 0.16 * altitude
    precipitation = np.clip(circ_precip, 0.0, 1.0)

    # 9. Hydrology
    grad_y, grad_x = np.gradient(elevation)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    hydrology = np.clip(precipitation * (1.0 - np.clip(slope * 4.0, 0.0, 0.7)) + 0.15 * (1.0 - altitude), 0.0, 1.0)

    # 10. Vegetation Biomass Potential
    temp_suitability = np.exp(-((temperature - 0.65) ** 2) / 0.12)
    water_suitability = np.clip((precipitation - 0.15) / 0.70, 0.0, 1.0)
    vegetation = np.zeros_like(elevation, dtype=np.float32)
    vegetation[land_mask] = np.clip(temp_suitability[land_mask] * water_suitability[land_mask] * 1.2, 0.0, 1.0)

    # 11. Planetary Biomes with Guaranteed Invariant
    # 0 = Ocean, 1 = Ice / Glaciers, 2 = Desert / Arid, 3 = Grassland / Savannah,
    # 4 = Temperate Forest, 5 = Tropical Rainforest, 6 = Tundra / Alpine Highland
    biome = np.zeros(elevation.shape, dtype=np.uint8)

    # Rule: Ocean cells are strictly biome 0
    # Land cells default to 3 (Grassland) to prevent any land cell from being 0 (Ocean)
    biome[land_mask] = 3

    # Apply specific land biome classifications
    biome[land_mask & (temperature < 0.20)] = 1                                         # Polar Ice / Glaciers
    biome[land_mask & (temperature >= 0.20) & (precipitation < 0.25)] = 2               # Desert / Arid
    biome[land_mask & (temperature >= 0.20) & (precipitation >= 0.25) & (precipitation < 0.48)] = 3 # Grassland
    biome[land_mask & (temperature >= 0.20) & (precipitation >= 0.48) & (precipitation < 0.72)] = 4 # Forest
    biome[land_mask & (temperature >= 0.50) & (precipitation >= 0.72)] = 5             # Rainforest
    biome[land_mask & (temperature < 0.35) & (altitude > 0.18)] = 6                     # Tundra / Alpine

    return PlanetData(
        latitude=lat,
        longitude=lon,
        elevation=elevation.astype(np.float32),
        land_mask=land_mask,
        temperature=temperature.astype(np.float32),
        precipitation=precipitation.astype(np.float32),
        biome=biome,
        hydrology=hydrology.astype(np.float32),
        vegetation=vegetation.astype(np.float32),
    )
