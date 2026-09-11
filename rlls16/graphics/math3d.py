import math
import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.copy()
    return v / norm


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b).astype(np.float32)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    fovy_rad = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    u_norm = normalize(up)
    s = normalize(cross(f, u_norm))
    u = cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -dot(s, eye)
    m[1, 3] = -dot(u, eye)
    m[2, 3] = dot(f, eye)
    return m


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_y(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_z(rad: float) -> np.ndarray:
    c = math.cos(rad)
    s = math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def rotate_axis(axis: np.ndarray, rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(rad)
    s = math.sin(rad)
    t = 1.0 - c

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = t * x * x + c
    m[0, 1] = t * x * y - s * z
    m[0, 2] = t * x * z + s * y
    m[1, 0] = t * x * y + s * z
    m[1, 1] = t * y * y + c
    m[1, 2] = t * y * z - s * x
    m[2, 0] = t * x * z - s * y
    m[2, 1] = t * y * z + s * x
    m[2, 2] = t * z * z + c
    return m


def mat4_mul(*matrices: np.ndarray) -> np.ndarray:
    res = matrices[0]
    for m in matrices[1:]:
        res = np.dot(res, m)
    return res.astype(np.float32)


def screen_to_ray(
    mouse_x: float,
    mouse_y: float,
    width: int,
    height: int,
    view_mat: np.ndarray,
    proj_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert screen pixel coordinates to a 3D ray in world space (origin, direction)."""
    ndc_x = (2.0 * mouse_x) / width - 1.0
    ndc_y = 1.0 - (2.0 * mouse_y) / height

    inv_pv = np.linalg.inv(np.dot(proj_mat, view_mat))

    near_point = np.dot(inv_pv, np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32))
    far_point = np.dot(inv_pv, np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32))

    near_point = near_point[:3] / near_point[3]
    far_point = far_point[:3] / far_point[3]

    ray_dir = normalize(far_point - near_point)
    return near_point, ray_dir


def ray_sphere_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> float | None:
    """Return distance along ray to intersection with sphere, or None if no hit."""
    oc = ray_origin - sphere_center
    b = 2.0 * float(np.dot(ray_dir, oc))
    c = float(np.dot(oc, oc)) - sphere_radius * sphere_radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0
    if t0 > 0:
        return t0
    if t1 > 0:
        return t1
    return None


def camera_relative_model_view(
    world_pos: np.ndarray,
    camera_pos: np.ndarray,
    rot_model: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute camera-relative model matrix in float32 from float64 world and camera positions.
    Subtracting in double precision (float64) before converting to float32 eliminates
    jitter when viewing fine surface features from huge astronomical world coordinates.
    """
    rel_pos = (np.asarray(world_pos, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)).astype(np.float32)
    m = np.eye(4, dtype=np.float32)
    if rot_model is not None:
        m[0:3, 0:3] = rot_model[0:3, 0:3]
    m[0:3, 3] = rel_pos
    return m


def extract_frustum_planes(view_proj: np.ndarray) -> np.ndarray:
    """
    Extract the 6 view-projection frustum planes in normalized form:
    planes[i] = [A, B, C, D] where A*x + B*y + C*z + D = 0.
    Order: 0: Left, 1: Right, 2: Bottom, 3: Top, 4: Near, 5: Far.
    Points inside the frustum have dot(plane[:3], p) + plane[3] >= 0.
    """
    m = np.asarray(view_proj, dtype=np.float32)
    planes = np.zeros((6, 4), dtype=np.float32)

    # Left: row 3 + row 0
    planes[0] = m[3] + m[0]
    # Right: row 3 - row 0
    planes[1] = m[3] - m[0]
    # Bottom: row 3 + row 1
    planes[2] = m[3] + m[1]
    # Top: row 3 - row 1
    planes[3] = m[3] - m[1]
    # Near: row 3 + row 2
    planes[4] = m[3] + m[2]
    # Far: row 3 - row 2
    planes[5] = m[3] - m[2]

    # Normalize plane normals
    lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    planes /= lengths
    return planes


def sphere_in_frustum(center: np.ndarray, radius: float, planes: np.ndarray | list) -> bool:
    """
    Test if a bounding sphere is inside or intersecting the view frustum.
    Returns False if strictly outside any of the 6 planes.
    """
    if planes is None or len(planes) == 0:
        return True
    c = np.asarray(center, dtype=np.float32)
    r = float(radius)
    for p in planes:
        dist = float(p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3])
        if dist < -r:
            return False
    return True


def sample_terrain_altitude(lat: float, lon: float, world_data: dict, radius: float = 5.0, terrain_amp: float = 0.35) -> float:
    """
    Sample continuous terrain surface radius at given latitude and longitude.
    """
    if "elevation" not in world_data:
        return radius
    elev_field = world_data["elevation"]
    h, w = elev_field.shape
    fy = (lat + (math.pi / 2.0)) / math.pi * (h - 1)
    fx = (lon + math.pi) / (2.0 * math.pi) * w

    y0 = max(0, min(h - 1, int(math.floor(fy))))
    y1 = max(0, min(h - 1, y0 + 1))
    x0 = int(math.floor(fx)) % w
    x1 = (x0 + 1) % w

    ty = fy - math.floor(fy)
    tx = fx - math.floor(fx)

    v00 = float(elev_field[y0, x0])
    v10 = float(elev_field[y0, x1])
    v01 = float(elev_field[y1, x0])
    v11 = float(elev_field[y1, x1])

    # If quantized uint16, convert to [0, 1]
    if v00 > 1.5 or v10 > 1.5:
        v00 /= 65535.0
        v10 /= 65535.0
        v01 /= 65535.0
        v11 /= 65535.0

    top = v00 * (1.0 - tx) + v10 * tx
    bottom = v01 * (1.0 - tx) + v11 * tx
    elev = top * (1.0 - ty) + bottom * ty

    return radius + (elev - 0.50) * terrain_amp


def unproject_terrain_hit(
    mouse_x: float,
    mouse_y: float,
    width: int,
    height: int,
    view_mat: np.ndarray,
    proj_mat: np.ndarray,
    earth_center: np.ndarray,
    earth_radius: float = 5.0,
    world_data: dict | None = None,
    earth_rot_rad: float = 0.0,
    axial_tilt_rad: float = math.radians(23.44),
) -> tuple[np.ndarray, float, float] | None:
    """
    Raycast from screen coordinates to intersect Earth's displaced surface.
    Returns:
        (hit_point_world, lat_rad, lon_rad) or None if the ray misses Earth.
    """
    ray_origin, ray_dir = screen_to_ray(mouse_x, mouse_y, width, height, view_mat, proj_mat)
    # 1. Base sphere intersection
    # Use conservative radius (radius + max elevation amplitude)
    max_r = earth_radius + 0.20
    t = ray_sphere_intersect(ray_origin, ray_dir, earth_center, max_r)
    if t is None or t <= 0:
        return None

    # Step along ray to find precise intersection point
    hit_pos = ray_origin + ray_dir * t
    # Refine hit point against displaced sphere
    rel_hit = hit_pos - earth_center
    dist_hit = np.linalg.norm(rel_hit)
    if dist_hit < 1e-6:
        return None
    dir_hit = rel_hit / dist_hit

    # Convert direction to Earth's local model coordinates factoring rotation & tilt
    cz = math.cos(axial_tilt_rad)
    sz = math.sin(axial_tilt_rad)
    rx = cz * dir_hit[0] - sz * dir_hit[1]
    ry = sz * dir_hit[0] + cz * dir_hit[1]
    rz = dir_hit[2]

    cy = math.cos(-earth_rot_rad)
    sy = math.sin(-earth_rot_rad)
    mx = cy * rx + sy * rz
    my = ry
    mz = -sy * rx + cy * rz

    lat = math.asin(max(-1.0, min(1.0, my)))
    lon = math.atan2(mx, mz)

    if world_data is not None:
        actual_r = sample_terrain_altitude(lat, lon, world_data, radius=earth_radius)
    else:
        actual_r = earth_radius

    refined_hit = earth_center + dir_hit * actual_r
    return refined_hit, lat, lon


def compute_surface_tangent_basis(
    focus_pos: np.ndarray,
    earth_center: np.ndarray,
    cam_forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local surface tangent orthonormal basis:
    - N: Surface normal pointing directly outward into space
    - F_tan: Camera forward vector projected onto the local tangent plane
    - R_tan: Tangent right vector (cross(N, F_tan))
    """
    diff = np.asarray(focus_pos, dtype=np.float32) - np.asarray(earth_center, dtype=np.float32)
    norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6:
        N = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        N = normalize(diff)

    fwd = normalize(np.asarray(cam_forward, dtype=np.float32))
    # Project camera forward onto tangent plane: F - N * dot(F, N)
    f_dot_n = float(np.dot(fwd, N))
    f_tan = fwd - N * f_dot_n
    norm_f_tan = np.linalg.norm(f_tan)

    if norm_f_tan < 1e-4:
        # Looking straight down or straight up: pick reference tangent
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(N[1]) > 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        f_tan = normalize(ref - N * float(np.dot(ref, N)))
    else:
        f_tan = f_tan / norm_f_tan

    r_tan = normalize(cross(N, f_tan))
    return N, f_tan, r_tan


