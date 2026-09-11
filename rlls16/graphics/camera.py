"""
RLLS 16 — Continuous Multi-Scale Logarithmic Camera.

Supports multi-scale navigation across:
1. Solar System Scale (~100–500 units)
2. Planetary Orbit Scale (~15–50 units)
3. Continental / Regional Scale (~6–15 units)
4. Surface / Settlement Scale (~5.05–6.0 units, where Earth radius = 5.0)

Features:
- Float64 double-precision internal tracking
- Altitude-proportional logarithmic zoom rate
- Dynamic near/far frustum planes to prevent near-clipping near terrain
- Ground collision avoidance clamping
- Camera-relative floating-origin view generation
"""

import math
import numpy as np
from .math3d import (
    look_at, perspective, normalize, cross,
    unproject_terrain_hit, compute_surface_tangent_basis
)


class OrbitCamera:
    """
    Continuous Multi-Scale Logarithmic Camera with Inertial Dynamics,
    Zoom-to-Cursor, Surface-Tangent Navigation, and Horizon Frustum Tracking.
    """

    def __init__(self, aspect: float = 16.0 / 9.0):
        # Focus targets: 'SOLAR', 'SUN', 'EARTH', 'MOON', 'FREE'
        self.focus_target_name = "SOLAR"
        self.target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.current_focus_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Spherical angles
        self.yaw = 0.65       # azimuth angle (radians)
        self.pitch = 0.42     # polar elevation angle (radians)
        self.distance = 180.0 # distance from current focus target

        self.target_yaw = self.yaw
        self.target_pitch = self.pitch
        self.target_distance = self.distance

        # Inertial dynamics velocities
        self.zoom_velocity = 0.0
        self.cam_velocity = np.zeros(3, dtype=np.float64)
        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.pan_velocity = np.zeros(3, dtype=np.float64)

        # Astronomical and planetary state tracking
        self.earth_center = np.zeros(3, dtype=np.float64)
        self.earth_rot_angle = 0.0
        self.axial_tilt = math.radians(23.44)

        # Min/Max distance bounds
        self.min_dist = 5.005
        self.max_dist = 800.0
        self.world_data = None
        self.pan_offset = np.zeros(3, dtype=np.float64)

        # Field of view & clipping
        self.fovy = 45.0
        self.aspect = aspect
        self.near = 0.5
        self.far = 3000.0

        # Mouse tracking
        self.is_orbiting = False
        self.is_panning = False
        self.last_mouse_pos = (0, 0)

    def set_world_data(self, world_data: dict):
        """Provide world data for continuous ground clearance sensing."""
        self.world_data = world_data

    def set_aspect(self, aspect: float):
        self.aspect = max(0.1, aspect)

    @property
    def earth_rot_rad(self) -> float:
        return self.earth_rot_angle

    @earth_rot_rad.setter
    def earth_rot_rad(self, value: float):
        self.earth_rot_angle = value

    @property
    def axial_tilt_rad(self) -> float:
        return self.axial_tilt

    @axial_tilt_rad.setter
    def axial_tilt_rad(self, value: float):
        self.axial_tilt = value

    def focus_object(self, name: str, position: np.ndarray, distance: float | None = None, snap: bool = False):
        self.focus_target_name = name
        self.target_pos = np.asarray(position, dtype=np.float64)
        self.pan_offset = np.zeros(3, dtype=np.float64)
        self.cam_velocity = np.zeros(3, dtype=np.float64)
        self.zoom_velocity = 0.0
        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.pan_velocity = np.zeros(3, dtype=np.float64)

        # Set appropriate min_dist depending on focused object
        if name == "EARTH":
            self.min_dist = 5.005
        elif name == "MOON":
            self.min_dist = 1.42
        elif name == "SUN":
            self.min_dist = 13.5
        else:
            self.min_dist = 8.0

        if distance is not None:
            self.target_distance = max(self.min_dist, float(distance))
            self.distance = self.target_distance
        else:
            if name == "SUN":
                self.target_distance = 65.0
                self.target_pitch = 0.35
            elif name == "EARTH":
                self.target_distance = 22.0
                self.target_pitch = 0.38
            elif name == "MOON":
                self.target_distance = 7.5
                self.target_pitch = 0.25
            elif name == "SOLAR":
                self.target_distance = 190.0
                self.target_pitch = 0.55
            self.distance = self.target_distance

        if snap:
            self.snap_to_target()

        self.is_flying = False
        self.fly_duration = 2.0
        self.fly_elapsed = 0.0

    def snap_to_target(self):
        """Immediately snap camera state to target values without lerping."""
        self.current_focus_pos = self.target_pos.copy()
        self.distance = self.target_distance
        self.yaw = self.target_yaw
        self.pitch = self.target_pitch
        self.zoom_velocity = 0.0
        self.cam_velocity = np.zeros(3, dtype=np.float64)

    def fly_to_lat_lon(
        self,
        lat_deg: float,
        lon_deg: float,
        earth_pos: np.ndarray | None = None,
        earth_rot_rad: float = 0.0,
        altitude: float = 2.2,
        duration: float = 2.0
    ):
        """
        Smooth fly-to animation to pinpoint a geographic settlement or point on Earth.
        """
        target = earth_pos if earth_pos is not None else self.current_focus_pos
        self.focus_object("EARTH", target, distance=5.0 + altitude)
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        self.target_pitch = max(-math.pi / 2.0 + 0.08, min(math.pi / 2.0 - 0.08, lat_rad))
        self.target_yaw = -(lon_rad + earth_rot_rad) + math.pi / 2.0
        self.is_flying = True
        self.fly_duration = max(0.1, duration)
        self.fly_elapsed = 0.0

    def handle_mouse_down(self, button: int, pos: tuple[int, int]):
        self.last_mouse_pos = pos
        if button == 1:  # Left click
            self.is_orbiting = True
        elif button in (2, 3):  # Middle or Right click
            self.is_panning = True

    def handle_mouse_up(self, button: int):
        if button == 1:
            self.is_orbiting = False
        elif button in (2, 3):
            self.is_panning = False

    def handle_mouse_motion(self, pos: tuple[int, int]):
        dx = pos[0] - self.last_mouse_pos[0]
        dy = pos[1] - self.last_mouse_pos[1]
        self.last_mouse_pos = pos

        altitude = max(0.002, self.distance - self.min_dist)

        if self.is_orbiting:
            sensitivity = 0.0042 * min(1.0, 0.15 + 0.85 * math.log10(max(1.0, altitude * 4.0 + 1.0)))
            self.yaw_velocity += dx * sensitivity * 12.0
            self.pitch_velocity += dy * sensitivity * 12.0
            self.target_yaw += dx * sensitivity
            self.target_pitch += dy * sensitivity
            self.target_pitch = max(-math.pi / 2.0 + 0.04, min(math.pi / 2.0 - 0.04, self.target_pitch))

        elif self.is_panning:
            pan_speed = 0.0007 * max(0.02, altitude)
            forward, right, up = self.get_basis()
            delta_pan = -right.astype(np.float64) * (dx * pan_speed) + up.astype(np.float64) * (dy * pan_speed)
            self.pan_velocity += delta_pan * 8.0
            self.pan_offset += delta_pan
            self.target_pos += delta_pan

    def handle_mouse_wheel(
        self,
        y: float,
        mouse_pos: tuple[int, int] | None = None,
        width: int = 1280,
        height: int = 720,
    ):
        """
        Inertial zoom impulse with altitude-scaled logarithmic response and planetary zoom-to-cursor.
        """
        # Normalize delta_y: handles both standard 1/-1 steps and precise_y or high-res wheel ticks
        if abs(y) > 10.0:
            norm_y = y / 120.0
        else:
            norm_y = math.copysign(min(3.0, max(0.1, abs(y))), y)

        altitude = max(0.001, self.distance - self.min_dist)
        # Logarithmic scale: small steps near ground, large steps in deep space
        impulse_mag = max(0.003, altitude * 0.18)
        self.zoom_velocity -= norm_y * impulse_mag * 10.0

        # Zoom-to-cursor support:
        # If cursor is provided and pointing at Earth, shift target/pan towards that surface point
        if norm_y > 0 and mouse_pos is not None and self.focus_target_name == "EARTH":
            try:
                view_mat = self.get_view_matrix()
                proj_mat = self.get_projection_matrix()
                earth_c = self.earth_center if hasattr(self, 'earth_center') else self.current_focus_pos
                hit_info = unproject_terrain_hit(
                    mouse_pos[0], mouse_pos[1], width, height,
                    view_mat, proj_mat,
                    earth_center=earth_c,
                    earth_radius=5.0,
                    world_data=self.world_data,
                    earth_rot_rad=getattr(self, 'earth_rot_angle', 0.0),
                    axial_tilt_rad=getattr(self, 'axial_tilt', math.radians(23.44))
                )
                if hit_info is not None:
                    hit_world, _, _ = hit_info
                    shift_frac = min(0.30, max(0.02, (impulse_mag / max(0.1, self.distance)) * 0.85))
                    delta_target = (hit_world - self.target_pos) * shift_frac
                    self.pan_offset += delta_target
                    self.target_pos += delta_target
            except Exception:
                pass

    def handle_keyboard(self, key_state, dt: float):
        import pygame
        altitude = max(0.002, self.distance - self.min_dist)
        base_speed = max(0.02, altitude * 0.75)

        def is_down(k):
            if hasattr(key_state, 'get'):
                return bool(key_state.get(k, False))
            try:
                return bool(key_state[k])
            except (IndexError, KeyError):
                return False

        is_shift = is_down(pygame.K_LSHIFT) or is_down(pygame.K_RSHIFT)
        speed = base_speed * (2.5 if is_shift else 1.0)

        forward, right, up = self.get_basis()

        # Surface-tangent basis when navigating Earth
        if self.focus_target_name == "EARTH":
            earth_c = getattr(self, 'earth_center', self.current_focus_pos)
            N, f_tan, r_tan = compute_surface_tangent_basis(self.current_focus_pos, earth_c, forward)
            N = N.astype(np.float64)
            f_tan = f_tan.astype(np.float64)
            r_tan = r_tan.astype(np.float64)
        else:
            N = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            f_tan = forward.astype(np.float64)
            r_tan = right.astype(np.float64)

        accel = np.zeros(3, dtype=np.float64)
        if is_down(pygame.K_w):
            accel += f_tan
        if is_down(pygame.K_s):
            accel -= f_tan
        if is_down(pygame.K_a):
            accel -= r_tan
        if is_down(pygame.K_d):
            accel += r_tan
        if is_down(pygame.K_q):
            accel += N
        if is_down(pygame.K_e):
            accel -= N

        # Reset camera orientation / level horizon
        if is_down(pygame.K_r):
            self.target_pitch = 0.35
            self.pitch_velocity = 0.0

        norm_accel = np.linalg.norm(accel)
        if norm_accel > 1e-5:
            accel = accel / norm_accel
            self.cam_velocity += accel * (speed * 16.0) * dt

        zoom_step = max(0.004, altitude * 0.12)
        if is_down(pygame.K_PLUS) or is_down(pygame.K_EQUALS):
            self.zoom_velocity -= zoom_step * 10.0
        if is_down(pygame.K_MINUS):
            self.zoom_velocity += zoom_step * 10.0

    def update(self, dynamic_target_pos: np.ndarray | None, dt: float):
        """
        Frame-rate-independent inertial dynamics with surface clearance and horizon frustum.
        """
        if dynamic_target_pos is not None:
            self.target_pos = np.asarray(dynamic_target_pos, dtype=np.float64) + self.pan_offset

        # Dynamic ground clearance calculation for Earth
        if self.focus_target_name == "EARTH" and self.world_data is not None:
            from .math3d import sample_terrain_altitude
            rel = self.get_eye_pos_f64() - self.current_focus_pos
            norm_r = np.linalg.norm(rel)
            if norm_r > 1e-6:
                norm_dir = rel / norm_r
                lat = math.asin(max(-1.0, min(1.0, norm_dir[1])))
                lon = math.atan2(norm_dir[0], norm_dir[2])
                surf_r = sample_terrain_altitude(lat, lon, self.world_data, radius=5.0, terrain_amp=0.35)
                self.min_dist = surf_r + 0.004
                self.target_distance = max(self.min_dist, self.target_distance)

        # Integrate zoom velocity into target_distance
        self.target_distance += self.zoom_velocity * dt
        self.zoom_velocity *= math.exp(-8.0 * max(1e-4, dt))
        if abs(self.zoom_velocity) < 1e-6:
            self.zoom_velocity = 0.0

        # Integrate mouse orbit inertia into target yaw and pitch
        self.target_yaw += self.yaw_velocity * dt
        self.target_pitch += self.pitch_velocity * dt
        self.target_pitch = max(-math.pi / 2.0 + 0.04, min(math.pi / 2.0 - 0.04, self.target_pitch))
        self.yaw_velocity *= math.exp(-9.0 * max(1e-4, dt))
        self.pitch_velocity *= math.exp(-9.0 * max(1e-4, dt))

        # Integrate keyboard camera velocity with exponential damping
        self.pan_offset += self.cam_velocity * dt
        self.target_pos += self.cam_velocity * dt
        self.cam_velocity *= math.exp(-7.0 * max(1e-4, dt))
        if np.linalg.norm(self.cam_velocity) < 1e-6:
            self.cam_velocity = np.zeros(3, dtype=np.float64)

        self.pan_offset += self.pan_velocity * dt
        self.target_pos += self.pan_velocity * dt
        self.pan_velocity *= math.exp(-8.0 * max(1e-4, dt))

        # Clamp target_distance against min_dist & max_dist
        if self.target_distance < self.min_dist:
            self.target_distance = self.min_dist
            self.zoom_velocity = max(0.0, self.zoom_velocity)
        elif self.target_distance > self.max_dist:
            self.target_distance = self.max_dist
            self.zoom_velocity = min(0.0, self.zoom_velocity)

        # Smooth critically damped motion towards targets
        decay_pos = 1.0 - math.exp(-8.5 * max(1e-4, dt))
        decay_rot = 1.0 - math.exp(-12.0 * max(1e-4, dt))
        decay_dist = 1.0 - math.exp(-9.0 * max(1e-4, dt))

        self.current_focus_pos += (self.target_pos - self.current_focus_pos) * decay_pos
        self.yaw += (self.target_yaw - self.yaw) * decay_rot
        self.pitch += (self.target_pitch - self.pitch) * decay_rot
        self.distance += (self.target_distance - self.distance) * decay_dist

        # Dynamic near clipping plane: safe fraction of local altitude
        altitude = max(0.002, self.distance - self.min_dist)
        self.near = max(0.0005, min(0.5, altitude * 0.04))

        # Dynamic far clipping plane: visible horizon + margin
        if self.focus_target_name == "EARTH" and self.distance > 5.0:
            d_sq = self.distance * self.distance
            r_sq = 5.0 * 5.0
            horizon_d = math.sqrt(max(0.01, d_sq - r_sq))
            self.far = max(25.0, min(3000.0, horizon_d * 2.2 + 10.0))
        elif self.focus_target_name == "SOLAR":
            self.far = 3000.0
        else:
            self.far = max(50.0, self.distance * 3.5)

        if getattr(self, "is_flying", False):
            self.fly_elapsed += dt
            if self.fly_elapsed >= self.fly_duration:
                self.is_flying = False

    def get_eye_pos_f64(self) -> np.ndarray:
        cos_p = math.cos(self.pitch)
        sin_p = math.sin(self.pitch)
        sin_y = math.sin(self.yaw)
        cos_y = math.cos(self.yaw)

        rel_pos = np.array([
            self.distance * cos_p * sin_y,
            self.distance * sin_p,
            self.distance * cos_p * cos_y
        ], dtype=np.float64)

        return self.current_focus_pos + rel_pos

    def get_eye_pos(self) -> np.ndarray:
        return self.get_eye_pos_f64().astype(np.float32)

    def get_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye = self.get_eye_pos()
        target = self.current_focus_pos.astype(np.float32)
        forward = normalize(target - eye)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(cross(forward, world_up))
        up = normalize(cross(right, forward))
        return forward, right, up

    def get_view_matrix(self) -> np.ndarray:
        eye = self.get_eye_pos()
        target = self.current_focus_pos.astype(np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(eye, target, up)

    def get_projection_matrix(self) -> np.ndarray:
        return perspective(self.fovy, self.aspect, self.near, self.far)


# Point 20: PlanetCamera alias
PlanetCamera = OrbitCamera

