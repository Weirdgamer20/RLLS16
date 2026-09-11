import time
import json
from pathlib import Path
import numpy as np

from .config import GeneratorConfig
from .planet import generate_planet, PlanetData
from .storage import save_world
from .export_3d import generate_pbr_textures, export_obj_mesh, export_glb_model


def validate_planet(planet: PlanetData) -> dict:
    """Execute rigorous mathematical validation checks on generated fields."""
    errors = []

    # 1. Finite values check
    for field_name in ["elevation", "temperature", "precipitation", "hydrology", "vegetation"]:
        arr = getattr(planet, field_name)
        if not np.isfinite(arr).all():
            errors.append(f"Non-finite (NaN or Inf) detected in field: {field_name}")

    # 2. Bound checks [0.0, 1.0]
    if (planet.elevation < 0.0).any() or (planet.elevation > 1.0).any():
        errors.append("Elevation exceeds normalized bounds [0.0, 1.0]")
    if (planet.temperature < 0.0).any() or (planet.temperature > 1.0).any():
        errors.append("Temperature exceeds normalized bounds [0.0, 1.0]")
    if (planet.precipitation < 0.0).any() or (planet.precipitation > 1.0).any():
        errors.append("Precipitation exceeds normalized bounds [0.0, 1.0]")

    # 3. Latitude and Longitude continuous spherical bounds
    if abs(float(planet.latitude[0]) + np.pi / 2.0) > 1e-4 or abs(float(planet.latitude[-1]) - np.pi / 2.0) > 1e-4:
        errors.append("Latitude does not span [-pi/2, pi/2]")
    if abs(float(planet.longitude[0]) + np.pi) > 1e-4:
        errors.append("Longitude does not start at -pi")

    # 4. Biome consistency invariants (No ocean biome on land, no land biome on ocean)
    land_with_ocean_biome = int((planet.land_mask & (planet.biome == 0)).sum())
    if land_with_ocean_biome > 0:
        errors.append(f"Biome inconsistency: {land_with_ocean_biome} land cells classified with ocean biome (0)")

    ocean_with_land_biome = int((~planet.land_mask & (planet.biome != 0)).sum())
    if ocean_with_land_biome > 0:
        errors.append(f"Biome inconsistency: {ocean_with_land_biome} ocean cells classified with land biome (>0)")

    # 5. Metrics calculation
    total_cells = planet.elevation.size
    land_cells = int(planet.land_mask.sum())
    ocean_cells = total_cells - land_cells
    land_fraction = land_cells / total_cells
    ocean_fraction = ocean_cells / total_cells

    biome_counts = {int(b): int((planet.biome == b).sum()) for b in range(7)}

    return {
        "status": "VALID" if not errors else "FAILED",
        "errors": errors,
        "resolution": f"{planet.elevation.shape[0]} x {planet.elevation.shape[1]}",
        "land_fraction": round(land_fraction, 4),
        "ocean_fraction": round(ocean_fraction, 4),
        "elevation_min": round(float(np.min(planet.elevation)), 4),
        "elevation_max": round(float(np.max(planet.elevation)), 4),
        "elevation_mean": round(float(np.mean(planet.elevation)), 4),
        "biome_distribution": biome_counts,
    }


def run(cfg: GeneratorConfig | None = None):
    if cfg is None:
        cfg = GeneratorConfig()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("RLLS 16 — Standalone Canonical World Generator")
    print("==================================================")
    print(f"Seed        : {cfg.seed}")
    print(f"Resolution  : {cfg.lat_samples} (lat) x {cfg.lon_samples} (lon)")
    print(f"Relief Amp  : {cfg.terrain_amplitude} (radius {cfg.radius})")
    print(f"Output Dir  : {out_dir.resolve()}")
    print("--------------------------------------------------")

    start_time = time.time()
    print("[1/5] Synthesizing continuous planetary fields...")
    planet = generate_planet(cfg)

    print("[2/5] Validating mathematical integrity...")
    val_report = validate_planet(planet)
    if val_report["status"] != "VALID":
        raise ValueError(f"World validation failed: {val_report['errors']}")
    print(f"      Status: PASS (Land: {val_report['land_fraction']*100:.1f}%, Ocean: {val_report['ocean_fraction']*100:.1f}%)")

    albedo_bytes = b""
    generated_files = []

    # 3. Export PBR Textures
    if cfg.export_textures:
        print("[3/5] Generating PBR Planetary Texture Suite...")
        tex_paths = generate_pbr_textures(planet, out_dir)
        albedo_bytes = tex_paths.get("albedo_bytes", b"")
        for k, p in tex_paths.items():
            if k != "albedo_bytes":
                sz_kb = Path(p).stat().st_size / 1024.0
                generated_files.append((f"Texture: {k}", p, f"{sz_kb:.1f} KB"))

    # 4. Export 3D Mesh & Models
    print("[4/5] Building 3D models for external applications...")
    if cfg.export_glb:
        glb_path = export_glb_model(planet, out_dir, cfg.radius, cfg.terrain_amplitude, albedo_bytes)
        sz_mb = Path(glb_path).stat().st_size / (1024.0 * 1024.0)
        generated_files.append(("glTF 2.0 Binary Model", glb_path, f"{sz_mb:.2f} MB"))

    if cfg.export_obj:
        obj_path = export_obj_mesh(planet, out_dir, cfg.radius, cfg.terrain_amplitude)
        sz_mb = Path(obj_path).stat().st_size / (1024.0 * 1024.0)
        generated_files.append(("Wavefront OBJ Mesh", obj_path, f"{sz_mb:.2f} MB"))
        mtl_path = out_dir / "canonical_world.mtl"
        if mtl_path.exists():
            generated_files.append(("Material Library", str(mtl_path), f"{mtl_path.stat().st_size / 1024.0:.1f} KB"))

    # 5. Export Canonical Data Package (.npz) & Validation Report (.json)
    print("[5/5] Packaging canonical scientific data and validation...")
    if cfg.export_npz:
        npz_path = out_dir / "canonical_world.npz"
        save_world(planet, cfg, npz_path)
        sz_kb = npz_path.stat().st_size / 1024.0
        generated_files.append(("Canonical Data Package", str(npz_path), f"{sz_kb:.1f} KB"))

        # Also update root worlds/canonical_world.npz for existing consumers
        root_npz = Path("worlds/canonical_world.npz")
        if root_npz.parent.exists():
            save_world(planet, cfg, root_npz)

    if cfg.export_validation_json:
        elapsed = time.time() - start_time
        val_report["generation_time_sec"] = round(elapsed, 3)
        val_report["seed"] = cfg.seed
        report_path = out_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)
        generated_files.append(("Validation Report", str(report_path), f"{report_path.stat().st_size / 1024.0:.1f} KB"))

    total_elapsed = time.time() - start_time
    print("--------------------------------------------------")
    print(f"GENERATION COMPLETE ({total_elapsed:.2f}s)")
    print("--------------------------------------------------")
    print("READY TO USE 3D ARTIFACTS:")
    for desc, fpath, size_str in generated_files:
        print(f"  * {desc:<24} : {Path(fpath).name:<26} ({size_str})")
    print("==================================================")
