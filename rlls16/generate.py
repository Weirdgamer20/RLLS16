"""
RLLS 16 — Canonical 2D Earth Generator Module Interface.
"""

from pathlib import Path
from .config import GeneratorConfig
from world_acquisition.generate_canonical_world import generate as run_pipeline


def run(cfg: GeneratorConfig | None = None):
    if cfg is None:
        cfg = GeneratorConfig()

    w = cfg.lon_samples if hasattr(cfg, "lon_samples") else 512
    h = cfg.lat_samples if hasattr(cfg, "lat_samples") else 256
    s = cfg.seed if hasattr(cfg, "seed") else 16001

    pkg_root = Path(__file__).resolve().parent.parent
    raw_dir = pkg_root / "world_data" / "raw"
    canonical_dir = pkg_root / "world_data" / "canonical"
    worlds_dir = pkg_root / "worlds"

    return run_pipeline(
        width=w,
        height=h,
        seed=s,
        raw_dir=raw_dir,
        canonical_dir=canonical_dir,
        worlds_dir=worlds_dir,
    )


if __name__ == "__main__":
    run()
