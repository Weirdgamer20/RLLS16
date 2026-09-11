from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    # Grid resolution for planetary continuous fields (lat x lon)
    # Default 256 x 512 generates fast and produces high fidelity 3D spherical meshes.
    lat_samples: int = 256
    lon_samples: int = 512

    # Deterministic seed. The same seed reproduces the exact same canonical world.
    seed: int = 16001

    # Base planet radius in standard 3D world units (1.0 = unit sphere)
    radius: float = 1.0

    # Terrain relief height scaling (vertical displacement amplitude)
    terrain_amplitude: float = 0.045

    # Sea level in normalized elevation space [0.0, 1.0]
    sea_level: float = 0.50

    # Multi-frequency noise octaves for terrain shaping
    octaves: int = 6

    # Output directory for exported 3D model, PBR textures, and data
    output_dir: str = "worlds/canonical_world"

    # Export formats
    export_glb: bool = True               # Self-contained glTF 2.0 binary 3D model
    export_obj: bool = True               # Universal Wavefront OBJ + MTL mesh
    export_textures: bool = True          # Complete PBR Equirectangular Texture Pack (PNG)
    export_npz: bool = True               # Compact 16-bit quantized numerical data package
    export_validation_json: bool = True   # Quality validation report and telemetry
