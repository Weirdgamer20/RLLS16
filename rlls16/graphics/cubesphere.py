"""
RLLS 16 — Cube-Sphere Planetary Geometry & Spatial Mapping.

Implements the normalized 6-face cube-sphere:
- Cardinal faces (+X, -X, +Y, -Y, +Z, -Z)
- Tangent-to-sphere radial projection
- Spherical coordinates to continuous Latitude / Longitude mapping
- Continuous canonical field interpolation across spherical coordinates
"""

import math
import numpy as np


CARDINAL_FACES = [
    "+X (Right)",
    "-X (Left)",
    "+Y (North Pole)",
    "-Y (South Pole)",
    "+Z (Front)",
    "-Z (Back)",
]

# Face basis vectors (Normal, Right, Up)
# Chosen so that adjacent edges match seamlessly in right-handed space
FACE_BASIS = [
    # 0: +X face
    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
    # 1: -X face
    (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    # 2: +Y face (North)
    (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    # 3: -Y face (South)
    (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    # 4: +Z face
    (np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    # 5: -Z face
    (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
]


def cube_to_sphere(face_id: int, u: float | np.ndarray, v: float | np.ndarray) -> np.ndarray:
    """
    Project cube face coordinates u, v in [-1, 1] onto the unit sphere.
    Returns unit vector (N, 3) or (3,).
    """
    normal, right, up = FACE_BASIS[face_id]

    if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
        # Vectorized array evaluation
        u = np.asarray(u)
        v = np.asarray(v)
        c_x = normal[0] + u * right[0] + v * up[0]
        c_y = normal[1] + u * right[1] + v * up[1]
        c_z = normal[2] + u * right[2] + v * up[2]
        c = np.stack([c_x, c_y, c_z], axis=-1)
        length = np.linalg.norm(c, axis=-1, keepdims=True)
        return c / np.maximum(length, 1e-12)
    else:
        # Scalar evaluation
        cx = normal[0] + u * right[0] + v * up[0]
        cy = normal[1] + u * right[1] + v * up[1]
        cz = normal[2] + u * right[2] + v * up[2]
        length = math.sqrt(cx * cx + cy * cy + cz * cz)
        inv = 1.0 / max(length, 1e-12)
        return np.array([cx * inv, cy * inv, cz * inv], dtype=np.float32)


def sphere_to_latlon(pos: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Convert 3D unit sphere point(s) to Latitude [-pi/2, pi/2] and Longitude [-pi, pi].
    Matches the parameterization of PlanetData:
      x = cos(lat) * cos(lon)
      y = sin(lat)
      z = cos(lat) * sin(lon)
    """
    if pos.ndim == 1:
        x, y, z = pos[0], pos[1], pos[2]
        lat = math.asin(max(-1.0, min(1.0, y)))
        lon = math.atan2(z, x)
        return lat, lon
    else:
        x = pos[..., 0]
        y = np.clip(pos[..., 1], -1.0, 1.0)
        z = pos[..., 2]
        lat = np.arcsin(y)
        lon = np.arctan2(z, x)
        return lat, lon


def sample_canonical_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling of a 2D equirectangular field (H, W) given arrays of lat and lon.
    Periodic wrapping for longitude [-pi, pi] and clamped for latitude [-pi/2, pi/2].
    """
    h, w = field.shape
    # Map lat [-pi/2, pi/2] -> [0, h - 1]
    fy = (lat + (math.pi / 2.0)) / math.pi * (h - 1)
    # Map lon [-pi, pi] -> [0, w]
    fx = (lon + math.pi) / (2.0 * math.pi) * w

    y0 = np.clip(np.floor(fy).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.mod(np.floor(fx).astype(np.int32), w)
    x1 = np.mod(x0 + 1, w)

    ty = (fy - y0).astype(np.float32)
    tx = (fx - np.floor(fx)).astype(np.float32)

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    top = v00 * (1.0 - tx) + v10 * tx
    bottom = v01 * (1.0 - tx) + v11 * tx
    return top * (1.0 - ty) + bottom * ty
