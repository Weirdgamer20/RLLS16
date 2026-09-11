from pathlib import Path
import json
import numpy as np

MAGIC = "RLLS16-CANONICAL-WORLD"

def _u16(a):
    return np.round(np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)

def save_world(data, cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "magic": MAGIC,
        "format_version": 2,
        "generator_seed": cfg.seed,
        "lat_samples": cfg.lat_samples,
        "lon_samples": cfg.lon_samples,
        "radius": cfg.radius,
        "terrain_amplitude": cfg.terrain_amplitude,
        "sea_level": cfg.sea_level,
        "storage": "uint16 normalized fields",
        "features": ["elevation", "land_mask", "temperature", "precipitation", "biome", "hydrology", "vegetation"],
    }

    save_dict = {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "elevation": _u16(data.elevation),
        "land_mask": data.land_mask.astype(np.uint8),
        "temperature": _u16(data.temperature),
        "precipitation": _u16(data.precipitation),
        "biome": data.biome,
        "metadata": json.dumps(metadata),
    }

    if hasattr(data, "hydrology") and data.hydrology is not None:
        save_dict["hydrology"] = _u16(data.hydrology)
    if hasattr(data, "vegetation") and data.vegetation is not None:
        save_dict["vegetation"] = _u16(data.vegetation)

    np.savez_compressed(path, **save_dict)

def load_world(path):
    z = np.load(path, allow_pickle=False)
    metadata = json.loads(str(z["metadata"]))

    def f16(name):
        if name in z:
            return z[name].astype(np.float32) / 65535.0
        return np.zeros_like(z["elevation"], dtype=np.float32)

    return {
        "latitude": z["latitude"],
        "longitude": z["longitude"],
        "elevation": f16("elevation"),
        "land_mask": z["land_mask"].astype(bool),
        "temperature": f16("temperature"),
        "precipitation": f16("precipitation"),
        "biome": z["biome"],
        "hydrology": f16("hydrology"),
        "vegetation": f16("vegetation"),
        "metadata": metadata,
    }
