"""
RLLS16 Canonical World Exporter.
Serializes the 13 canonical simulation layers into compressed .npz with SHA-256 checksum and metadata.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
from ..preprocessing.layer_builder import CanonicalEarthLayers

MAGIC = "RLLS16-2D-CANONICAL-WORLD"

def _u16(a: np.ndarray) -> np.ndarray:
    return np.round(np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)

def export_canonical_world(
    layers: CanonicalEarthLayers,
    out_dir: Path | str,
    seed: int = 16001,
    source_datasets: list[str] | None = None
) -> tuple[Path, str, dict]:
    """
    Exports the canonical 2D Earth package (.npz) and validation report.
    Returns: (output_npz_path, sha256_checksum, metadata)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "canonical_world.npz"

    sources = source_datasets or [
        "Natural Earth 110m Physical (Land, Ocean, Coastline, Rivers, Lakes)",
        "NOAA/NCEI Historical Climate & Insolation",
        "WWF Terrestrial Ecoregions 14-Biome Classification"
    ]

    H, W = layers.elevation.shape

    save_dict = {
        "latitude": layers.latitude,
        "longitude": layers.longitude,
        "land_mask": layers.land_mask.astype(np.uint8),
        "elevation": _u16(layers.elevation),
        "coastline": layers.coastline.astype(np.uint8),
        "rivers_lakes": _u16(layers.rivers_lakes),
        "climate": layers.climate,
        "temperature": _u16(layers.temperature),
        "precipitation": _u16(layers.precipitation),
        "soil_moisture": _u16(layers.soil_moisture),
        "biome": layers.biome,
        "vegetation": _u16(layers.vegetation),
        "wildlife": _u16(layers.wildlife),
        "resources": _u16(layers.resources),
        "agents": _u16(layers.agents),
        # Legacy compatibility aliases
        "hydrology": _u16(layers.rivers_lakes),
    }

    # First write to temporary file to compute checksum
    temp_path = out_dir / "temp_canonical.npz"
    np.savez_compressed(temp_path, **save_dict)

    sha256 = hashlib.sha256()
    with open(temp_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    checksum = f"sha256:{sha256.hexdigest()}"

    metadata = {
        "magic": MAGIC,
        "world_id": "RLLS16_EARTH_CANONICAL_2D",
        "version": "1.0",
        "projection": "Equirectangular (Plate Carree)",
        "resolution": f"{H}x{W}",
        "generator_seed": seed,
        "storage": "uint16 normalized / uint8 categorical",
        "source_datasets": sources,
        "layers": [
            "0: land_mask",
            "1: elevation",
            "2: coastline",
            "3: rivers_lakes",
            "4: climate",
            "5: temperature",
            "6: precipitation",
            "7: soil_moisture",
            "8: biome",
            "9: vegetation",
            "10: wildlife",
            "11: resources",
            "12: agents",
        ],
        "checksum": checksum,
    }

    save_dict["metadata"] = json.dumps(metadata)
    np.savez_compressed(npz_path, **save_dict)

    if temp_path.exists():
        temp_path.unlink()

    # Write metadata JSON file
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, checksum, metadata
