"""
RLLS16 Canonical 2D Earth Generator (Offline Data Pipeline).
Derives deterministic canonical Earth simulation layers from real-world datasets.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to sys.path
pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pkg_root))

from world_acquisition.preprocessing.rasterizer import load_geojson, rasterize_polygons, rasterize_lines
from world_acquisition.preprocessing.layer_builder import build_canonical_layers
from world_acquisition.validation.validator import validate_canonical_world
from world_acquisition.export.exporter import export_canonical_world


def generate(
    width: int = 512,
    height: int = 256,
    seed: int = 16001,
    raw_dir: Path | str = "world_data/raw",
    canonical_dir: Path | str = "world_data/canonical",
    worlds_dir: Path | str = "worlds",
):
    start_time = time.time()
    raw_path = Path(raw_dir)
    canonical_path = Path(canonical_dir)
    worlds_path = Path(worlds_dir)

    print("==================================================")
    print("RLLS 16 — Canonical 2D Earth World Generator")
    print("==================================================")
    print(f"Seed        : {seed}")
    print(f"Resolution  : {height} (lat) x {width} (lon)")
    print(f"Projection  : Equirectangular (Plate Carree)")
    print(f"Raw Source  : {raw_path.resolve()}")
    print(f"Target Dir  : {canonical_path.resolve()}")
    print("--------------------------------------------------")

    # 1. Load Raw Datasets
    print("[1/5] Loading real Earth vector datasets...")
    land_file = raw_path / "ne_110m_land.geojson"
    coast_file = raw_path / "ne_110m_coastline.geojson"
    rivers_file = raw_path / "ne_110m_rivers_lake_centerlines.geojson"
    lakes_file = raw_path / "ne_110m_lakes.geojson"

    for f in [land_file, coast_file, rivers_file, lakes_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required raw dataset missing: {f}")

    land_geo = load_geojson(land_file)
    coast_geo = load_geojson(coast_file)
    rivers_geo = load_geojson(rivers_file)
    lakes_geo = load_geojson(lakes_file)

    # 2. Rasterize Geographical Baselines
    print("[2/5] Rasterizing real Earth geographic baselines...")
    land_mask = rasterize_polygons(land_geo, width=width, height=height, invert_holes=True)
    coast_mask = rasterize_lines(coast_geo, width=width, height=height, line_thickness=1)
    rivers_mask = rasterize_lines(rivers_geo, width=width, height=height, line_thickness=1)
    lakes_mask = rasterize_polygons(lakes_geo, width=width, height=height, invert_holes=False)

    # 3. Build Canonical Simulation Layers
    print("[3/5] Synthesizing 13 independent simulation layers...")
    layers = build_canonical_layers(
        land_mask=land_mask,
        coastline_mask=coast_mask,
        rivers_mask=rivers_mask,
        lakes_mask=lakes_mask,
        seed=seed,
    )

    # 4. Mathematical & Invariant Validation
    print("[4/5] Validating mathematical and geographic invariants...")
    report = validate_canonical_world(layers)
    if report["status"] != "VALID":
        raise ValueError(f"Canonical validation failed: {report['errors']}")
    print(f"      Status: PASS (Land: {report['land_fraction']*100:.1f}%, Ocean: {report['ocean_fraction']*100:.1f}%)")

    # 5. Export Canonical Package
    print("[5/5] Packaging canonical dataset and computing SHA-256...")
    out_npz, checksum, meta = export_canonical_world(layers, canonical_path, seed=seed)

    # Copy to worlds/ canonical path for consumers
    worlds_path.mkdir(parents=True, exist_ok=True)
    import shutil
    target_npz = worlds_path / "canonical_world.npz"
    shutil.copyfile(out_npz, target_npz)

    # Also update worlds/canonical_world/ directory if present
    sub_world_dir = worlds_path / "canonical_world"
    sub_world_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_npz, sub_world_dir / "canonical_world.npz")

    # Write validation report
    val_file = sub_world_dir / "validation_report.json"
    report["checksum"] = checksum
    report["generation_time_sec"] = round(time.time() - start_time, 3)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    total_time = time.time() - start_time
    sz_kb = out_npz.stat().st_size / 1024.0
    print("--------------------------------------------------")
    print(f"GENERATION COMPLETE ({total_time:.2f}s)")
    print(f"Checksum    : {checksum}")
    print(f"Size        : {sz_kb:.1f} KB")
    print(f"Artifact    : {out_npz.name}")
    print("==================================================")
    return out_npz, checksum, report


if __name__ == "__main__":
    generate()
