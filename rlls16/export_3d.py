import json
import math
import struct
from pathlib import Path
import numpy as np
import pygame

from .planet import PlanetData


def save_surface_as_png(surf: pygame.Surface, filepath: Path):
    """Save a pygame Surface directly to PNG."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(filepath))


def generate_pbr_textures(planet: PlanetData, output_dir: Path) -> dict:
    """
    Generate and export a complete planetary PBR equirectangular texture suite (PNG).
    Returns a dictionary of texture file paths and in-memory albedo PNG bytes.
    """
    tex_dir = output_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    elevation = planet.elevation
    land_mask = planet.land_mask
    biome = planet.biome
    temp = planet.temperature
    precip = planet.precipitation
    h, w = elevation.shape

    # -------------------------------------------------------------
    # 1. Albedo (True-Color Planetary Surface)
    # -------------------------------------------------------------
    albedo_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Ocean depth coloring
    ocean_depth = np.clip(1.0 - elevation / 0.50, 0.0, 1.0)
    ocean_r = (12 + 22 * (1.0 - ocean_depth)).astype(np.uint8)
    ocean_g = (38 + 55 * (1.0 - ocean_depth)).astype(np.uint8)
    ocean_b = (88 + 105 * (1.0 - ocean_depth)).astype(np.uint8)

    albedo_rgb[..., 0] = ocean_r
    albedo_rgb[..., 1] = ocean_g
    albedo_rgb[..., 2] = ocean_b

    # Biome coloration for land
    biome_palette = np.array([
        [25, 75, 140],    # 0 Ocean fallback
        [235, 242, 248],  # 1 Ice / Glaciers
        [215, 185, 110],  # 2 Desert / Arid
        [138, 178, 78],   # 3 Grassland
        [48, 122, 54],    # 4 Temperate Forest
        [22, 98, 48],     # 5 Tropical Rainforest
        [142, 148, 128],  # 6 Tundra / Highland
    ], dtype=np.float32)

    land_idx = np.where(land_mask)
    land_biomes = np.clip(biome[land_idx], 0, 6)
    land_colors = biome_palette[land_biomes]

    # Altitude relief shading
    altitude = np.maximum(elevation[land_idx] - 0.50, 0.0) * 2.0
    land_colors = land_colors * (0.86 + 0.32 * altitude[:, None])

    # High-altitude snow peaks
    snow_mask = elevation[land_idx] > 0.82
    land_colors[snow_mask] = land_colors[snow_mask] * 0.35 + np.array([245, 248, 255]) * 0.65

    land_colors = np.clip(land_colors, 0, 255).astype(np.uint8)
    albedo_rgb[land_idx[0], land_idx[1], 0] = land_colors[:, 0]
    albedo_rgb[land_idx[0], land_idx[1], 1] = land_colors[:, 1]
    albedo_rgb[land_idx[0], land_idx[1], 2] = land_colors[:, 2]

    # Flip vertically for standard equirectangular texture layout (North at top)
    albedo_rgb = np.ascontiguousarray(np.flipud(albedo_rgb))
    albedo_surf = pygame.image.frombuffer(albedo_rgb.tobytes(), (w, h), "RGB")
    albedo_path = tex_dir / "albedo.png"
    save_surface_as_png(albedo_surf, albedo_path)

    # -------------------------------------------------------------
    # 2. Heightmap (Grayscale Normalized Displacement)
    # -------------------------------------------------------------
    height_u8 = np.ascontiguousarray((np.flipud(elevation) * 255.0).astype(np.uint8))
    height_rgb = np.ascontiguousarray(np.repeat(height_u8[..., None], 3, axis=-1))
    height_surf = pygame.image.frombuffer(height_rgb.tobytes(), (w, h), "RGB")
    height_path = tex_dir / "heightmap.png"
    save_surface_as_png(height_surf, height_path)

    # -------------------------------------------------------------
    # 3. Tangent-Space Normal Map (Sobel Derivative)
    # -------------------------------------------------------------
    elev_flip = np.flipud(elevation)
    gy, gx = np.gradient(elev_flip)
    scale_factor = 28.0
    nx = -gx * scale_factor
    ny = -gy * scale_factor
    nz = np.ones_like(elevation)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= norm
    ny /= norm
    nz /= norm

    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb = np.ascontiguousarray(normal_rgb)

    normal_surf = pygame.image.frombuffer(normal_rgb.tobytes(), (w, h), "RGB")
    normal_path = tex_dir / "normal_map.png"
    save_surface_as_png(normal_surf, normal_path)

    # -------------------------------------------------------------
    # 4. Ocean / Specular / Roughness Mask (Water = 255, Land = 0)
    # -------------------------------------------------------------
    ocean_u8 = np.ascontiguousarray((np.flipud(~land_mask) * 255).astype(np.uint8))
    ocean_rgb = np.ascontiguousarray(np.repeat(ocean_u8[..., None], 3, axis=-1))
    ocean_surf = pygame.image.frombuffer(ocean_rgb.tobytes(), (w, h), "RGB")
    ocean_path = tex_dir / "ocean_mask.png"
    save_surface_as_png(ocean_surf, ocean_path)

    # -------------------------------------------------------------
    # 5. Dynamic Cloud Layer (RGBA with Transparency)
    # -------------------------------------------------------------
    lat_arr = np.linspace(-math.pi / 2, math.pi / 2, h, dtype=np.float32)[:, None]
    lon_arr = np.linspace(-math.pi, math.pi, w, dtype=np.float32)[None, :]
    band1 = np.exp(-((lat_arr - 0.08) ** 2) / 0.06) * 0.75
    band2 = np.exp(-((np.abs(lat_arr) - 0.82) ** 2) / 0.10) * 0.60
    waves = (
        np.sin(lon_arr * 4.0 + lat_arr * 3.0) * 0.25
        + np.cos(lon_arr * 7.0 - lat_arr * 5.0) * 0.20
    )
    cloud_density = np.clip(band1 + band2 + waves + 0.12, 0.0, 1.0)
    cloud_u8 = (np.flipud(cloud_density) * 255).astype(np.uint8)

    cloud_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cloud_rgba[..., 0] = 255
    cloud_rgba[..., 1] = 255
    cloud_rgba[..., 2] = 255
    cloud_rgba[..., 3] = cloud_u8 # Alpha is density
    cloud_rgba = np.ascontiguousarray(cloud_rgba)

    cloud_surf = pygame.image.frombuffer(cloud_rgba.tobytes(), (w, h), "RGBA")
    cloud_path = tex_dir / "clouds.png"
    save_surface_as_png(cloud_surf, cloud_path)

    # -------------------------------------------------------------
    # 6. Biomes Map
    # -------------------------------------------------------------
    biome_colors = np.zeros((h, w, 3), dtype=np.uint8)
    biome_flip = np.flipud(biome)
    for b_id in range(7):
        mask_b = (biome_flip == b_id)
        biome_colors[mask_b] = biome_palette[b_id].astype(np.uint8)
    biome_colors = np.ascontiguousarray(biome_colors)
    biome_surf = pygame.image.frombuffer(biome_colors.tobytes(), (w, h), "RGB")
    biome_path = tex_dir / "biomes.png"
    save_surface_as_png(biome_surf, biome_path)

    # Read raw albedo PNG bytes for embedding in glTF
    with open(albedo_path, "rb") as f:
        albedo_bytes = f.read()

    return {
        "albedo": str(albedo_path),
        "heightmap": str(height_path),
        "normal_map": str(normal_path),
        "ocean_mask": str(ocean_path),
        "clouds": str(cloud_path),
        "biomes": str(biome_path),
        "albedo_bytes": albedo_bytes,
    }


def build_spherical_mesh(
    planet: PlanetData,
    radius: float = 1.0,
    terrain_amplitude: float = 0.045,
    lat_segs: int = 128,
    lon_segs: int = 256
):
    """
    Build a continuous spherical 3D mesh with vertex elevation displacement,
    radial/perturbed normals, and UV texture coordinates.
    """
    elevation = planet.elevation
    h, w = elevation.shape

    verts = []
    normals = []
    uvs = []

    for i in range(lat_segs + 1):
        v = i / lat_segs
        lat = math.pi * (0.5 - v) # +pi/2 (North) to -pi/2 (South)
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)

        row_idx = int(v * (h - 1))

        for j in range(lon_segs + 1):
            u = j / lon_segs
            lon = 2.0 * math.pi * (u - 0.5) # -pi to +pi
            cos_lon = math.cos(lon)
            sin_lon = math.sin(lon)

            col_idx = int(u * (w - 1))

            # Sample elevation for relief displacement
            elev_val = float(elevation[row_idx, col_idx])
            r = radius + (elev_val - 0.50) * 2.0 * terrain_amplitude

            # 3D Position
            x = r * cos_lat * sin_lon
            y = r * sin_lat
            z = r * cos_lat * cos_lon

            # Surface Normal (radial base)
            nx = cos_lat * sin_lon
            ny = sin_lat
            nz = cos_lat * cos_lon

            verts.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([u, v])

    verts = np.array(verts, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    for i in range(lat_segs):
        for j in range(lon_segs):
            top_left = i * (lon_segs + 1) + j
            top_right = top_left + 1
            bot_left = (i + 1) * (lon_segs + 1) + j
            bot_right = bot_left + 1

            indices.append(top_left)
            indices.append(bot_left)
            indices.append(top_right)

            indices.append(top_right)
            indices.append(bot_left)
            indices.append(bot_right)

    indices = np.array(indices, dtype=np.uint32)
    return verts, normals, uvs, indices


def export_obj_mesh(planet: PlanetData, output_dir: Path, radius: float = 1.0, terrain_amplitude: float = 0.045) -> str:
    """Export standard Wavefront OBJ and MTL mesh with UVs and material reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "canonical_world.obj"
    mtl_path = output_dir / "canonical_world.mtl"

    verts, normals, uvs, indices = build_spherical_mesh(planet, radius, terrain_amplitude, lat_segs=96, lon_segs=192)

    # 1. Write MTL Material
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# RLLS 16 Canonical World Material\n")
        f.write("newmtl Canonical_Earth_Material\n")
        f.write("Ka 0.05 0.05 0.05\n")
        f.write("Kd 1.00 1.00 1.00\n")
        f.write("Ks 0.20 0.20 0.20\n")
        f.write("Ns 16.0\n")
        f.write("map_Kd textures/albedo.png\n")
        f.write("bump textures/normal_map.png\n\n")

    # 2. Write OBJ Geometry
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# RLLS 16 Canonical World 3D Mesh\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("o Canonical_Earth\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for vt in uvs:
            # OBJ texture coordinates: invert V
            f.write(f"vt {vt[0]:.6f} {1.0 - vt[1]:.6f}\n")

        for vn in normals:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

        f.write("usemtl Canonical_Earth_Material\n")
        f.write("s 1\n")

        # 1-indexed faces: f v/vt/vn
        for i in range(0, len(indices), 3):
            i1 = indices[i] + 1
            i2 = indices[i + 1] + 1
            i3 = indices[i + 2] + 1
            f.write(f"f {i1}/{i1}/{i1} {i2}/{i2}/{i2} {i3}/{i3}/{i3}\n")

    return str(obj_path)


def export_glb_model(
    planet: PlanetData,
    output_dir: Path,
    radius: float = 1.0,
    terrain_amplitude: float = 0.045,
    albedo_png_bytes: bytes = b""
) -> str:
    """
    Self-contained glTF 2.0 Binary (.glb) exporter with embedded geometry,
    normals, UVs, indices, PBR materials, and embedded Albedo texture.
    Can be loaded directly into Godot, Unity, Unreal, Blender, Three.js, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / "canonical_world.glb"

    verts, normals, uvs, indices = build_spherical_mesh(planet, radius, terrain_amplitude, lat_segs=96, lon_segs=192)

    # Binary buffers
    pos_bytes = verts.tobytes()
    norm_bytes = normals.tobytes()
    uv_bytes = uvs.tobytes()
    idx_bytes = indices.tobytes()

    # Calculate buffer view byte offsets (aligned to 4-byte boundaries)
    def pad4(data_bytes: bytes) -> bytes:
        pad = (4 - (len(data_bytes) % 4)) % 4
        return data_bytes + (b"\x00" * pad)

    b_pos = pad4(pos_bytes)
    b_norm = pad4(norm_bytes)
    b_uv = pad4(uv_bytes)
    b_idx = pad4(idx_bytes)
    b_img = pad4(albedo_png_bytes)

    offset_pos = 0
    offset_norm = offset_pos + len(b_pos)
    offset_uv = offset_norm + len(b_norm)
    offset_idx = offset_uv + len(b_uv)
    offset_img = offset_idx + len(b_idx)
    total_bin_len = offset_img + len(b_img)

    bin_data = b_pos + b_norm + b_uv + b_idx + b_img

    # Min/Max bounding box for positions
    min_pos = [float(np.min(verts[:, 0])), float(np.min(verts[:, 1])), float(np.min(verts[:, 2]))]
    max_pos = [float(np.max(verts[:, 0])), float(np.max(verts[:, 1])), float(np.max(verts[:, 2]))]

    num_verts = len(verts)
    num_indices = len(indices)

    # Assemble glTF JSON structure
    gltf_dict = {
        "asset": {
            "version": "2.0",
            "generator": "RLLS 16 Canonical World Generator"
        },
        "scene": 0,
        "scenes": [
            {"name": "Scene", "nodes": [0]}
        ],
        "nodes": [
            {"name": "Canonical_Earth", "mesh": 0}
        ],
        "meshes": [
            {
                "name": "Earth_Sphere",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4 # TRIANGLES
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "Earth_PBR_Material",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {
                        "index": 0
                    },
                    "metallicFactor": 0.05,
                    "roughnessFactor": 0.70
                },
                "doubleSided": False
            }
        ],
        "textures": [
            {"sampler": 0, "source": 0}
        ],
        "images": [
            {
                "name": "Earth_Albedo",
                "mimeType": "image/png",
                "bufferView": 4
            }
        ],
        "samplers": [
            {
                "magFilter": 9729, # LINEAR
                "minFilter": 9987, # LINEAR_MIPMAP_LINEAR
                "wrapS": 10497,    # REPEAT
                "wrapT": 33071     # CLAMP_TO_EDGE
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126, # FLOAT
                "count": num_verts,
                "type": "VEC3",
                "max": max_pos,
                "min": min_pos
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126, # FLOAT
                "count": num_verts,
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126, # FLOAT
                "count": num_verts,
                "type": "VEC2"
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5125, # UNSIGNED_INT
                "count": num_indices,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_pos, "byteLength": len(pos_bytes), "target": 34962}, # ARRAY_BUFFER
            {"buffer": 0, "byteOffset": offset_norm, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": offset_uv, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": len(idx_bytes), "target": 34963}, # ELEMENT_ARRAY_BUFFER
            {"buffer": 0, "byteOffset": offset_img, "byteLength": len(albedo_png_bytes)}
        ],
        "buffers": [
            {"byteLength": total_bin_len}
        ]
    }

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    json_padded = pad4(json_bytes)
    # Pad JSON chunk with trailing spaces as per glTF spec
    space_pad = len(json_padded) - len(json_bytes)
    if space_pad > 0:
        json_bytes += b" " * space_pad

    # GLB Header (12 bytes)
    magic = 0x46546C67 # "glTF"
    version = 2
    total_glb_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<III", magic, version, total_glb_len)

    # Chunk 0: JSON (type = 0x4E4F534A)
    chunk0_hdr = struct.pack("<II", len(json_bytes), 0x4E4F534A)

    # Chunk 1: BIN (type = 0x004E4942)
    chunk1_hdr = struct.pack("<II", len(bin_data), 0x004E4942)

    with open(glb_path, "wb") as f:
        f.write(header)
        f.write(chunk0_hdr)
        f.write(json_bytes)
        f.write(chunk1_hdr)
        f.write(bin_data)

    return str(glb_path)
