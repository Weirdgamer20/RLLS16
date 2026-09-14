"""
RLLS 16 — Canonical 2D Earth Configuration.
"""

from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for canonical 2D Earth offline preprocessing."""
    lat_samples: int = 256
    lon_samples: int = 512
    seed: int = 16001
    output_dir: str = "worlds/canonical_world"
    projection: str = "Equirectangular (Plate Carree)"
    sea_level: float = 0.50
    num_layers: int = 13
