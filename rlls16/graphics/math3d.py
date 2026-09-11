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

