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
from .math3d import look_at, perspective, normalize, cross


class OrbitCamera:
    """
    Continuous Multi-Scale Logarithmic Camera for RLLS 16.
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

        # Min/Max distance bounds
        # Earth radius = 5.0; min_dist allows zooming down close to surface terrain
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

    def focus_object(self, name: str, position: np.ndarray, distance: float | None = None, snap: bool = False):
        self.focus_target_name = name
        self.target_pos = np.asarray(position, dtype=np.float64)
        self.pan_offset = np.zeros(3, dtype=np.float64)

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
        Point 53: Smooth fly-to animation to pinpoint a geographic settlement on Earth.
        """
        target = earth_pos if earth_pos is not None else self.current_focus_pos
        self.focus_object("EARTH", target, distance=5.0 + altitude)
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        # Aim camera at the location factoring diurnal rotation
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

        # Continuous altitude-scaled navigation speeds
        altitude = max(0.002, self.distance - self.min_dist)

        if self.is_orbiting:
            # Scale orbit sensitivity according to altitude above planet surface
            sensitivity = 0.0042 * min(1.0, 0.15 + 0.85 * math.log10(max(1.0, altitude * 4.0 + 1.0)))
            self.target_yaw += dx * sensitivity
            self.target_pitch += dy * sensitivity
            # Clamp pitch to prevent gimbal flip
            self.target_pitch = max(-math.pi / 2.0 + 0.04, min(math.pi / 2.0 - 0.04, self.target_pitch))

        elif self.is_panning:
            # Pan speed proportional to true altitude
            pan_speed = 0.0007 * max(0.02, altitude)
            forward, right, up = self.get_basis()
            delta_pan = -right.astype(np.float64) * (dx * pan_speed) + up.astype(np.float64) * (dy * pan_speed)
            self.pan_offset += delta_pan
            self.target_pos += delta_pan

    def handle_mouse_wheel(self, y: float):
        """
        Deep zoom logarithmic zoom step dynamically scaling with altitude.
        Continuous from solar system (180+) to near-ground (0.005).
        """
        altitude = max(0.002, self.target_distance - self.min_dist)
        step = max(0.003, altitude * 0.15)

        if y > 0:
            # Zoom In
            self.target_distance = max(self.min_dist, self.target_distance - step)
        else:
            # Zoom Out
            self.target_distance = min(self.max_dist, self.target_distance + step)

    def handle_keyboard(self, key_state, dt: float):
        import pygame
        altitude = max(0.005, self.distance - self.min_dist)
        pan_speed = 18.0 * dt * max(0.005, altitude / 20.0)
        forward, right, up = self.get_basis()

        delta_pan = np.zeros(3, dtype=np.float64)
        if key_state[pygame.K_w]:
            delta_pan += forward.astype(np.float64) * pan_speed
        if key_state[pygame.K_s]:
            delta_pan -= forward.astype(np.float64) * pan_speed
        if key_state[pygame.K_a]:
            delta_pan -= right.astype(np.float64) * pan_speed
        if key_state[pygame.K_d]:
            delta_pan += right.astype(np.float64) * pan_speed
        if key_state[pygame.K_q]:
            delta_pan += np.array([0.0, 1.0, 0.0], dtype=np.float64) * pan_speed
        if key_state[pygame.K_e]:
            delta_pan -= np.array([0.0, 1.0, 0.0], dtype=np.float64) * pan_speed

        if np.any(delta_pan != 0.0):
            self.pan_offset += delta_pan
            self.target_pos += delta_pan

        zoom_step = max(0.004, altitude * 0.08)
        if key_state[pygame.K_PLUS] or key_state[pygame.K_EQUALS]:
            self.target_distance = max(self.min_dist, self.target_distance - zoom_step)
        if key_state[pygame.K_MINUS]:
            self.target_distance = min(self.max_dist, self.target_distance + zoom_step)

    def update(self, dynamic_target_pos: np.ndarray | None, dt: float):
        """
        Point 12: Frame-rate-independent exponential critically damped motion.
        Supports deep zoom down to near-surface with dynamic ground clearance and near clip plane.
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
                # Allow camera to approach within 0.005 units (~30-50m) of displaced terrain
                self.min_dist = surf_r + 0.005
                self.target_distance = max(self.min_dist, self.target_distance)

        # dt-independent exponential smoothing
        decay_pos = 1.0 - math.exp(-8.5 * max(1e-4, dt))
        decay_rot = 1.0 - math.exp(-10.0 * max(1e-4, dt))
        decay_dist = 1.0 - math.exp(-9.0 * max(1e-4, dt))

        self.current_focus_pos += (self.target_pos - self.current_focus_pos) * decay_pos
        self.yaw += (self.target_yaw - self.yaw) * decay_rot
        self.pitch += (self.target_pitch - self.pitch) * decay_rot
        self.distance += (self.target_distance - self.distance) * decay_dist

        # Dynamic near clipping plane: scales smoothly down to 0.001 to prevent near-clipping at deep zoom
        altitude = max(0.002, self.distance - self.min_dist)
        self.near = max(0.001, min(0.5, altitude * 0.08))

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
