"""
RLLS 16 — Cube-Sphere Quadtree LOD Manager.

Manages 6 quadtrees (one per cardinal face).
Evaluates camera distance, horizon culling, and level of detail,
ensuring adaptive subdivision from planetary orbit down to regional/settlement scale.
"""

import math
import numpy as np
from .cubesphere import cube_to_sphere


class QuadtreeNode:
    """A node in the cube-sphere quadtree representing a spatial terrain tile."""
    __slots__ = (
        'face', 'level', 'u_min', 'v_min', 'u_max', 'v_max',
        'center_sphere', 'center_u', 'center_v', 'bounding_radius',
        'children', 'is_leaf', 'key', 'parent'
    )

    def __init__(self, face: int, level: int, u_min: float, v_min: float, u_max: float, v_max: float, parent=None):
        self.face = face
        self.level = level
        self.u_min = u_min
        self.v_min = v_min
        self.u_max = u_max
        self.v_max = v_max
        self.parent = parent

        self.center_u = (u_min + u_max) * 0.5
        self.center_v = (v_min + v_max) * 0.5
        self.center_sphere = cube_to_sphere(face, self.center_u, self.center_v)

        # Approximate bounding radius on unit sphere
        corner = cube_to_sphere(face, u_min, v_min)
        self.bounding_radius = float(np.linalg.norm(self.center_sphere - corner))

        self.children = []
        self.is_leaf = True
        self.key = f"{face}_{level}_{u_min:.5f}_{v_min:.5f}"

    def subdivide(self):
        """Create 4 child quadtree nodes."""
        if self.children:
            return

        mid_u = self.center_u
        mid_v = self.center_v
        next_lvl = self.level + 1

        self.children = [
            QuadtreeNode(self.face, next_lvl, self.u_min, self.v_min, mid_u, mid_v, parent=self),     # Bottom-Left
            QuadtreeNode(self.face, next_lvl, mid_u, self.v_min, self.u_max, mid_v, parent=self),     # Bottom-Right
            QuadtreeNode(self.face, next_lvl, self.u_min, mid_v, mid_u, self.v_max, parent=self),     # Top-Left
            QuadtreeNode(self.face, next_lvl, mid_u, mid_v, self.u_max, self.v_max, parent=self),     # Top-Right
        ]
        self.is_leaf = False

    def collapse(self):
        """Remove children, reverting this node back to a leaf."""
        self.children = []
        self.is_leaf = True


class LODManager:
    """
    Coordinates planetary quadtree subdivision with Screen-Space Error (Point 3),
    LOD Hysteresis (Point 4), Frustum Culling (Point 5), and Model-Space Horizon Culling (Point 6).
    """

    def __init__(self, max_lod: int = 6, error_threshold: float = 3.5):
        self.max_lod = max_lod
        self.split_threshold = 1.0 * error_threshold   # Point 4: 1.0x threshold
        self.merge_threshold = 0.7 * error_threshold   # Point 4: 0.7x threshold
        # Root nodes for each of the 6 cardinal faces
        self.roots = [
            QuadtreeNode(face_id, 0, -1.0, -1.0, 1.0, 1.0)
            for face_id in range(6)
        ]

    def update(
        self,
        camera_pos: np.ndarray,
        planet_pos: np.ndarray,
        radius: float = 5.0,
        earth_rot_angle: float = 0.0,
        axial_tilt: float = math.radians(23.44),
        fovy_deg: float = 45.0,
        viewport_height: int = 768,
        frustum_planes: list[np.ndarray] | None = None,
    ) -> list[QuadtreeNode]:
        """
        Evaluate LOD across all faces given camera state, Earth orientation, and projection parameters.
        Returns a list of visible leaf QuadtreeNodes to render.
        """
        cam_pos_f64 = np.asarray(camera_pos, dtype=np.float64)
        planet_pos_f64 = np.asarray(planet_pos, dtype=np.float64)

        rel_cam = cam_pos_f64 - planet_pos_f64
        cam_dist = float(np.linalg.norm(rel_cam))
        if cam_dist < 1e-6:
            cam_dir_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            cam_dir_world = rel_cam / cam_dist

        # Point 6: Transform camera direction into Earth's local model space
        # World to Model: rotate_y(-earth_rot_angle) * rotate_z(axial_tilt)
        cz = math.cos(axial_tilt)
        sz = math.sin(axial_tilt)
        # R_z(axial_tilt) on rel_cam
        rx = cz * cam_dir_world[0] - sz * cam_dir_world[1]
        ry = sz * cam_dir_world[0] + cz * cam_dir_world[1]
        rz = cam_dir_world[2]

        # R_y(-earth_rot_angle)
        cy = math.cos(-earth_rot_angle)
        sy = math.sin(-earth_rot_angle)
        mx = cy * rx + sy * rz
        my = ry
        mz = -sy * rx + cy * rz
        cam_dir_model = np.array([mx, my, mz], dtype=np.float64)

        # Horizon culling angle: max angle visible from planet center
        if cam_dist > radius:
            horizon_dot_limit = -(math.sqrt(max(0.0, 1.0 - (radius * radius) / (cam_dist * cam_dist))))
            horizon_margin = 0.20
        else:
            horizon_dot_limit = -1.0
            horizon_margin = 0.0

        # Screen-space projection scale factor: H / (2 * tan(fovy / 2))
        fovy_rad = math.radians(fovy_deg)
        proj_scale = (viewport_height * 0.5) / max(1e-4, math.tan(fovy_rad * 0.5))

        visible_leaves = []

        def traverse(node: QuadtreeNode):
            # 1. Horizon Culling Check in Model Space (Point 6)
            dot_cam = float(np.dot(node.center_sphere, cam_dir_model))
            if dot_cam < (horizon_dot_limit - node.bounding_radius - horizon_margin):
                return

            # 2. Distance from camera to node surface point in world space
            # Model space center on sphere
            sc = node.center_sphere
            # Model to World
            # R_y(earth_rot_angle) * R_z(-axial_tilt) * sc
            # First R_z(-axial_tilt)
            cz_neg = math.cos(-axial_tilt)
            sz_neg = math.sin(-axial_tilt)
            px = cz_neg * sc[0] - sz_neg * sc[1]
            py = sz_neg * sc[0] + cz_neg * sc[1]
            pz = sc[2]
            # Then R_y(earth_rot_angle)
            cy_pos = math.cos(earth_rot_angle)
            sy_pos = math.sin(earth_rot_angle)
            wx = cy_pos * px + sy_pos * pz
            wy = py
            wz = -sy_pos * px + cy_pos * pz
            node_world = planet_pos_f64 + np.array([wx, wy, wz], dtype=np.float64) * radius
            node_radius_world = radius * node.bounding_radius * 1.15 + 0.5

            # 3. Frustum Culling Check (Point 5)
            if frustum_planes:
                outside = False
                for plane in frustum_planes:
                    dist_to_plane = float(np.dot(plane[:3], node_world) + plane[3])
                    if dist_to_plane < -node_radius_world:
                        outside = True
                        break
                if outside:
                    return

            dist_to_node = float(np.linalg.norm(node_world - cam_pos_f64))

            # 4. Screen-Space Error Calculation (Point 3)
            # Physical chord length
            node_chord = radius * (2.0 / (2 ** node.level))
            # Surface geometric error estimate (curvature sagitta + terrain relief)
            sagitta = (node_chord * node_chord) / (8.0 * radius)
            geometric_error = sagitta + (0.35 / (2 ** node.level))
            # Screen-space pixel error
            projected_error = (geometric_error / max(0.05, dist_to_node)) * proj_scale

            # 5. LOD Hysteresis (Point 4)
            if node.is_leaf:
                # Approaching: split if projected error exceeds split threshold
                if projected_error > self.split_threshold and node.level < self.max_lod:
                    node.subdivide()
                    for child in node.children:
                        traverse(child)
                else:
                    visible_leaves.append(node)
            else:
                # Moving away: collapse only if error falls below merge threshold (Point 4)
                if projected_error < self.merge_threshold:
                    node.collapse()
                    visible_leaves.append(node)
                else:
                    for child in node.children:
                        traverse(child)

        for root in self.roots:
            traverse(root)

        return visible_leaves
