"""
RLLS16 Canonical Earth Validator.
Verifies finite numbers, bounds [0, 1], geographic bounds, and biome invariants.
"""

import numpy as np
from ..preprocessing.layer_builder import CanonicalEarthLayers

def validate_canonical_world(layers: CanonicalEarthLayers) -> dict:
    errors = []

    # 1. Finite check
    fields = [
        "elevation", "temperature", "precipitation", "soil_moisture",
        "vegetation", "wildlife", "resources", "rivers_lakes"
    ]
    for field in fields:
        arr = getattr(layers, field)
        if not np.isfinite(arr).all():
            errors.append(f"Non-finite (NaN/Inf) values found in layer: {field}")

    # 2. Normalized bounds [0.0, 1.0]
    for field in fields:
        arr = getattr(layers, field)
        if (arr < 0.0).any() or (arr > 1.0).any():
            errors.append(f"Layer '{field}' values exceed normalized bounds [0.0, 1.0]")

    # 3. Coordinate bounds
    if abs(float(layers.latitude[0]) + np.pi / 2.0) > 1e-4 or abs(float(layers.latitude[-1]) - np.pi / 2.0) > 1e-4:
        errors.append("Latitude does not span [-pi/2, pi/2]")
    if abs(float(layers.longitude[0]) + np.pi) > 1e-4:
        errors.append("Longitude does not start at -pi")

    # 4. Biome invariants: ocean cells should be 0 (Ocean), land cells should not be 0
    land_with_ocean_biome = int((layers.land_mask & (layers.biome == 0)).sum())
    if land_with_ocean_biome > 0:
        errors.append(f"Biome inconsistency: {land_with_ocean_biome} land cells classified with ocean biome")

    # 5. Land/Ocean ratio
    total_cells = layers.elevation.size
    land_cells = int(layers.land_mask.sum())
    ocean_cells = total_cells - land_cells
    land_frac = land_cells / total_cells
    ocean_frac = ocean_cells / total_cells

    biome_counts = {int(b): int((layers.biome == b).sum()) for b in np.unique(layers.biome)}

    return {
        "status": "VALID" if not errors else "FAILED",
        "errors": errors,
        "resolution": f"{layers.elevation.shape[0]} x {layers.elevation.shape[1]}",
        "land_fraction": round(land_frac, 4),
        "ocean_fraction": round(ocean_frac, 4),
        "elevation_min": round(float(np.min(layers.elevation)), 4),
        "elevation_max": round(float(np.max(layers.elevation)), 4),
        "elevation_mean": round(float(np.mean(layers.elevation)), 4),
        "biome_distribution": biome_counts,
    }
