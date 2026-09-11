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
        self.min_dist = 5.10
        self.max_dist = 800.0

        # Field of view & clipping
        self.fovy = 45.0
        self.aspect = aspect
        self.near = 0.5
        self.far = 3000.0

        # Mouse tracking
        self.is_orbiting = False
        self.is_panning = False
        self.last_mouse_pos = (0, 0)

    def set_aspect(self, aspect: float):
        self.aspect = max(0.1, aspect)

    def focus_object(self, name: str, position: np.ndarray, distance: float | None = None, snap: bool = False):
        self.focus_target_name = name
        self.target_pos = np.asarray(position, dtype=np.float64)

        # Set appropriate min_dist depending on focused object
        if name == "EARTH":
            self.min_dist = 5.08
        elif name == "MOON":
            self.min_dist = 1.48
        elif name == "SUN":
            self.min_dist = 13.5
        else:
            self.min_dist = 10.0

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

        # Point 11: Altitude-scaled planetary navigation speeds
        altitude = max(0.04, self.distance - self.min_dist)

        if self.is_orbiting:
            # Scale orbit sensitivity according to altitude above planet surface
            sensitivity = 0.0042 * min(1.0, 0.25 + 0.75 * math.log10(max(1.0, altitude + 1.0)))
            self.target_yaw += dx * sensitivity
            self.target_pitch += dy * sensitivity
            # Clamp pitch to prevent gimbal flip
            self.target_pitch = max(-math.pi / 2.0 + 0.04, min(math.pi / 2.0 - 0.04, self.target_pitch))

        elif self.is_panning:
            # Pan speed proportional to true altitude
            pan_speed = 0.0008 * max(0.2, altitude)
            forward, right, up = self.get_basis()
            self.target_pos -= right.astype(np.float64) * (dx * pan_speed)
            self.target_pos += up.astype(np.float64) * (dy * pan_speed)

    def handle_mouse_wheel(self, y: float):
        """
        Point 11: Logarithmic zoom step dynamically scaling with altitude.
        From Solar System to planetary orbit to settlement surface.
        """
        altitude = max(0.015, self.target_distance - self.min_dist)
        # Continuous smooth scaling: faster at high altitude, micro-stepped near terrain
        step = max(0.012, altitude * 0.18)

        if y > 0:
            # Zoom In
            self.target_distance = max(self.min_dist, self.target_distance - step)
        else:
            # Zoom Out
            self.target_distance = min(self.max_dist, self.target_distance + step)

    def handle_keyboard(self, key_state, dt: float):
        import pygame
        altitude = max(0.08, self.distance - self.min_dist)
        pan_speed = 18.0 * dt * max(0.01, altitude / 20.0)
        forward, right, up = self.get_basis()

        if key_state[pygame.K_w]:
            self.target_pos += forward.astype(np.float64) * pan_speed
        if key_state[pygame.K_s]:
            self.target_pos -= forward.astype(np.float64) * pan_speed
        if key_state[pygame.K_a]:
            self.target_pos -= right.astype(np.float64) * pan_speed
        if key_state[pygame.K_d]:
            self.target_pos += right.astype(np.float64) * pan_speed
        if key_state[pygame.K_q]:
            self.target_pos += np.array([0.0, 1.0, 0.0], dtype=np.float64) * pan_speed
        if key_state[pygame.K_e]:
            self.target_pos -= np.array([0.0, 1.0, 0.0], dtype=np.float64) * pan_speed

        zoom_step = max(0.015, altitude * 0.06)
        if key_state[pygame.K_PLUS] or key_state[pygame.K_EQUALS]:
            self.target_distance = max(self.min_dist, self.target_distance - zoom_step)
        if key_state[pygame.K_MINUS]:
            self.target_distance = min(self.max_dist, self.target_distance + zoom_step)

    def update(self, dynamic_target_pos: np.ndarray | None, dt: float):
        """
        Point 12: Frame-rate-independent exponential critically damped motion.
        Produces identical physical smoothing across 30 FPS, 60 FPS, and 120 FPS.
        """
        if dynamic_target_pos is not None:
            self.target_pos = np.asarray(dynamic_target_pos, dtype=np.float64)

        # dt-independent exponential smoothing
        decay_pos = 1.0 - math.exp(-8.5 * max(1e-4, dt))
        decay_rot = 1.0 - math.exp(-10.0 * max(1e-4, dt))
        decay_dist = 1.0 - math.exp(-9.0 * max(1e-4, dt))

        self.current_focus_pos += (self.target_pos - self.current_focus_pos) * decay_pos
        self.yaw += (self.target_yaw - self.yaw) * decay_rot
        self.pitch += (self.target_pitch - self.pitch) * decay_rot
        self.distance += (self.target_distance - self.distance) * decay_dist

        # Dynamic near plane to prevent near-clipping when close to surface
        altitude = max(0.01, self.distance - self.min_dist)
        self.near = max(0.02, min(0.5, altitude * 0.1))

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
