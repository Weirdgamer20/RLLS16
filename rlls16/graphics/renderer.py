import math
import numpy as np
import moderngl

from .shaders import (
    STARFIELD_VS, STARFIELD_FS,
    SUN_VS, SUN_FS,
    PLANET_VS, EARTH_FS, MOON_FS,
    LINE_VS, LINE_FS,
    OVERLAY_VS, OVERLAY_FS,
    CUBESPHERE_TERRAIN_VS, CUBESPHERE_TERRAIN_FS,
)
from .math3d import (
    identity, translate, scale, rotate_x, rotate_y, rotate_z,
    mat4_mul, normalize
)
from .quadtree import LODManager
from .terrain_tile import TerrainTilePool, build_tile_mesh_data
from .gpu_config import get_gpu_hardware_info, optimize_texture


def mat4_bytes(m: np.ndarray) -> bytes:
    """Format row-major 4x4 matrix into column-major bytes for OpenGL GLSL uniform mat4."""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def create_sphere_mesh(lat_segments: int = 64, lon_segments: int = 128, radius: float = 1.0):
    """
    Generate a smooth UV sphere mesh with positions, normals, and UVs.
    Returns: vertices (N, 8) [x, y, z, nx, ny, nz, u, v], indices (M,) uint32
    """
    verts = []
    for i in range(lat_segments + 1):
        lat = math.pi * (i / lat_segments - 0.5)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        v = 1.0 - (i / lat_segments)

        for j in range(lon_segments + 1):
            lon = 2.0 * math.pi * (j / lon_segments - 0.5)
            sin_lon = math.sin(lon)
            cos_lon = math.cos(lon)
            u = j / lon_segments

            x = cos_lat * sin_lon
            y = sin_lat
            z = cos_lat * cos_lon

            verts.append([
                x * radius, y * radius, z * radius,
                x, y, z,
                u, v
            ])

    vertices = np.array(verts, dtype=np.float32)

    indices = []
    for i in range(lat_segments):
        for j in range(lon_segments):
            first = i * (lon_segments + 1) + j
            second = first + lon_segments + 1

            indices.append(first)
            indices.append(second)
            indices.append(first + 1)

            indices.append(second)
            indices.append(second + 1)
            indices.append(first + 1)

    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices


def generate_earth_albedo_texture(world_data: dict) -> np.ndarray:
    """
    Build a high-definition RGBA texture representing the canonical Earth:
    RGB: terrain relief + ocean depth + biome coloration.
    Alpha: land mask (1.0 = land, 0.0 = ocean) for shader specular reflections.
    """
    elevation = world_data['elevation']
    land_mask = world_data['land_mask']
    biome = world_data['biome']
    precipitation = world_data['precipitation']
    temperature = world_data['temperature']

    h, w = elevation.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Ocean depth coloring
    ocean_depth = np.clip(1.0 - elevation / 0.50, 0.0, 1.0)
    ocean_r = (10 + 20 * (1.0 - ocean_depth)).astype(np.uint8)
    ocean_g = (35 + 50 * (1.0 - ocean_depth)).astype(np.uint8)
    ocean_b = (85 + 95 * (1.0 - ocean_depth)).astype(np.uint8)

    rgba[..., 0] = ocean_r
    rgba[..., 1] = ocean_g
    rgba[..., 2] = ocean_b
    rgba[..., 3] = 0  # ocean alpha = 0

    # Land biome coloration
    # Biome IDs: 0=ocean, 1=ice, 2=desert, 3=grassland, 4=forest, 5=rainforest, 6=tundra
    biome_palette = np.array([
        [25, 75, 140],    # 0 ocean fallback
        [230, 240, 245],  # 1 ice / glaciers
        [210, 180, 105],  # 2 arid / desert
        [135, 175, 75],   # 3 temperate grassland
        [48, 120, 52],    # 4 temperate forest
        [20, 95, 45],     # 5 tropical rainforest
        [140, 145, 125],  # 6 tundra / highlands
    ], dtype=np.float32)

    land_indices = np.where(land_mask)
    land_biomes = np.clip(biome[land_indices], 0, 6)
    land_colors = biome_palette[land_biomes]

    # Add subtle elevation shading & mountain peaks
    elev_relief = np.clip((elevation[land_indices] - 0.50) * 2.2, 0.0, 1.0)[:, None]
    land_colors = land_colors * (0.85 + 0.35 * elev_relief)

    # High altitude snow caps
    high_alt = elevation[land_indices] > 0.82
    land_colors[high_alt] = land_colors[high_alt] * 0.4 + np.array([240, 245, 255]) * 0.6

    land_colors = np.clip(land_colors, 0, 255).astype(np.uint8)

    rgba[land_indices[0], land_indices[1], 0] = land_colors[:, 0]
    rgba[land_indices[0], land_indices[1], 1] = land_colors[:, 1]
    rgba[land_indices[0], land_indices[1], 2] = land_colors[:, 2]
    rgba[land_indices[0], land_indices[1], 3] = 255  # land alpha = 255

    # Flip vertically for standard OpenGL UV mapping (v=0 at bottom)
    rgba = np.flipud(rgba)
    return rgba


def generate_cloud_texture(h: int = 192, w: int = 384, seed: int = 16001) -> np.ndarray:
    """Generate dynamic cloud cover field using multi-frequency sinusoids and noise."""
    lat = np.linspace(-math.pi / 2, math.pi / 2, h, dtype=np.float32)[:, None]
    lon = np.linspace(-math.pi, math.pi, w, dtype=np.float32)[None, :]

    # Tropical convergence zone & mid-latitude storm bands
    band1 = np.exp(-((lat - 0.1) ** 2) / 0.06) * 0.7
    band2 = np.exp(-((np.abs(lat) - 0.85) ** 2) / 0.12) * 0.55

    # Wave patterns
    waves = (
        np.sin(lon * 4.0 + lat * 3.0) * 0.25
        + np.cos(lon * 7.0 - lat * 5.0) * 0.20
        + np.sin(lon * 11.0 + lat * 8.0) * 0.15
    )

    clouds = np.clip(band1 + band2 + waves + 0.15, 0.0, 1.0)
    cloud_img = (clouds * 255).astype(np.uint8)
    cloud_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cloud_rgba[..., 0] = cloud_img
    cloud_rgba[..., 1] = cloud_img
    cloud_rgba[..., 2] = cloud_img
    cloud_rgba[..., 3] = cloud_img
    return np.flipud(cloud_rgba)


def generate_lunar_texture(h: int = 128, w: int = 256) -> np.ndarray:
    """Generate lunar regolith texture with dark basaltic mare basins and highlands."""
    lat = np.linspace(-math.pi / 2, math.pi / 2, h, dtype=np.float32)[:, None]
    lon = np.linspace(-math.pi, math.pi, w, dtype=np.float32)[None, :]

    # Low frequency mare patterns
    mare = (
        np.sin(lon * 2.0) * np.cos(lat * 2.0) * 0.25
        + np.cos(lon * 4.0 + 0.5) * np.sin(lat * 3.0) * 0.20
    )
    regolith = 0.55 + mare * 0.25
    regolith = np.clip(regolith, 0.2, 0.85)

    lunar_gray = (regolith * 255).astype(np.uint8)
    lunar_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    lunar_rgb[..., 0] = lunar_gray
    lunar_rgb[..., 1] = lunar_gray
    lunar_rgb[..., 2] = lunar_gray
    return lunar_rgb


class EarthTier:
    HIGH_DETAIL_CUBESPHERE = "HIGH_DETAIL_CUBESPHERE"
    LOW_DETAIL_CUBESPHERE = "LOW_DETAIL_CUBESPHERE"
    EMERGENCY_SPHERE = "EMERGENCY_SPHERE"


DEBUG_MODES = [
    "Normal",
    "Earth Solid",
    "Wireframe",
    "Heightmap",
    "Biome",
    "Land Mask",
    "Ocean Mask",
    "Normals",
    "LOD Level",
    "Tile Bounds"
]


class EarthRenderer:
    """
    Guaranteed Three-Tier Planetary Earth Renderer (Points 1, 2, 44).
    Hierarchy:
      1. HIGH_DETAIL_CUBESPHERE (Screen-space error quadtree + skirts)
      2. LOW_DETAIL_CUBESPHERE (LOD0 cube-sphere base)
      3. EMERGENCY_SPHERE (Continuous UV sphere fallback, guaranteed to render)
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        world_data: dict,
        prog_cubesphere: moderngl.Program,
        prog_earth: moderngl.Program,
        gpu_info: dict,
    ):
        self.ctx = ctx
        self.world_data = world_data
        self.prog_cubesphere = prog_cubesphere
        self.prog_earth = prog_earth
        self.gpu_info = gpu_info

        max_lod = 6 if gpu_info.get("is_dedicated", False) else 5
        self.lod_manager = LODManager(max_lod=max_lod, error_threshold=3.5)
        self.tile_pool = TerrainTilePool(self.ctx, max_gpu_cached=384, max_cpu_cached=768)

        # Pre-bake Emergency Fallback UV Sphere mesh (Point 2)
        s_verts, s_idxs = create_sphere_mesh(lat_segments=48, lon_segments=96, radius=1.0)
        self.emergency_vbo = ctx.buffer(s_verts.tobytes())
        self.emergency_ibo = ctx.buffer(s_idxs.tobytes())
        self.emergency_vao = ctx.vertex_array(
            self.prog_earth,
            [(self.emergency_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
            index_buffer=self.emergency_ibo,
            index_element_size=4
        )

        # Active Tier & Visual Debug Mode
        self.tier = EarthTier.HIGH_DETAIL_CUBESPHERE
        self.debug_mode = 0  # 0: Normal

        # Live performance telemetry
        self.telemetry = {
            "rendered_tiles": 0,
            "visible_nodes": 0,
            "tier_in_use": EarthTier.HIGH_DETAIL_CUBESPHERE,
            "max_active_lod": 0,
            "upload_queue": 0,
        }

        # Point 1: Mandatory Earth render diagnostic mode
        self.run_startup_diagnostic()

    def run_startup_diagnostic(self):
        """Execute and display the required 5-stage Earth diagnostic checks (Point 1)."""
        # 1. Earth Data Check
        elev = self.world_data.get("elevation")
        land = self.world_data.get("land_mask")
        data_ok = (elev is not None and len(elev) > 0 and land is not None and len(land) > 0)

        # 2. Terrain Mesh Check
        root_node = self.lod_manager.roots[0]
        v, idx = build_tile_mesh_data(root_node, self.world_data, grid_size=8, radius=5.0)
        mesh_ok = (len(v) > 0 and len(idx) > 0)

        # 3. GPU Buffer Check
        gpu_buf_ok = False
        test_vao = None
        test_buf = None
        test_ibuf = None
        fbo = None
        try:
            test_buf = self.ctx.buffer(v.tobytes())
            test_ibuf = self.ctx.buffer(idx.tobytes())
            test_vao = self.ctx.vertex_array(
                self.prog_cubesphere,
                [(test_buf, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_scalars')],
                index_buffer=test_ibuf,
                index_element_size=4
            )
            gpu_buf_ok = True
        except Exception:
            gpu_buf_ok = False

        # 4. Earth Shader Check
        shader_ok = (self.prog_cubesphere is not None and self.prog_earth is not None)

        # 5. Draw Call Check
        draw_ok = False
        try:
            fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((16, 16), 4)])
            fbo.use()
            if test_vao:
                test_vao.render()
                draw_ok = True
        except Exception:
            draw_ok = False
        finally:
            self.ctx.screen.use()
            try:
                if test_vao: test_vao.release()
                if test_buf: test_buf.release()
                if test_ibuf: test_ibuf.release()
                if fbo: fbo.release()
            except Exception:
                pass

        print("\n" + "=" * 35)
        print(" EARTH RENDER PIPELINE DIAGNOSTIC")
        print("=" * 35)
        print(f" EARTH DATA       {'OK' if data_ok else 'FAIL'}")
        print(f" TERRAIN MESH     {'OK' if mesh_ok else 'FAIL'}")
        print(f" GPU BUFFER       {'OK' if gpu_buf_ok else 'FAIL'}")
        print(f" EARTH SHADER     {'OK' if shader_ok else 'FAIL'}")
        print("=" * 35 + "\n")
        return {"data": data_ok, "mesh": mesh_ok, "gpu_buf": gpu_buf_ok, "shader": shader_ok, "draw": draw_ok}

    def render(
        self,
        camera,
        sim_time_sec: float,
        earth_pos: np.ndarray,
        earth_rot_angle: float,
        solar_irradiance: float,
        show_clouds: bool,
        show_atmo: bool,
        tex_albedo,
        tex_clouds,
        width: int,
        height: int,
    ):
        # 1. Process worker thread GPU uploads (Points 9, 10)
        self.tile_pool.process_ready_uploads(self.prog_cubesphere, max_uploads=6)
        self.tile_pool.step_frame()

        earth_radius = 5.0
        axial_tilt = math.radians(23.44)

        # Matrices
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        eye_pos = camera.get_eye_pos()

        # Wireframe toggle (Point 44)
        if self.debug_mode == 2:
            self.ctx.wireframe = True

        rendered_count = 0
        current_max_lod = 0
        tier_used = self.tier
        visible_nodes = []

        # Try HIGH_DETAIL_CUBESPHERE or LOW_DETAIL_CUBESPHERE
        if self.tier in (EarthTier.HIGH_DETAIL_CUBESPHERE, EarthTier.LOW_DETAIL_CUBESPHERE):
            try:
                m_earth = mat4_mul(
                    translate(earth_pos[0], earth_pos[1], earth_pos[2]),
                    rotate_z(-axial_tilt),
                    rotate_y(earth_rot_angle)
                )

                self.prog_cubesphere['u_model'].write(mat4_bytes(m_earth))
                self.prog_cubesphere['u_view'].write(mat4_bytes(view))
                self.prog_cubesphere['u_proj'].write(mat4_bytes(proj))
                self.prog_cubesphere['u_camera_pos'].value = tuple(eye_pos)
                self.prog_cubesphere['u_sun_pos'].value = (0.0, 0.0, 0.0)
                self.prog_cubesphere['u_cloud_offset'].value = (sim_time_sec * 0.0003) % 1.0
                self.prog_cubesphere['u_show_clouds'].value = 1.0 if show_clouds else 0.0
                self.prog_cubesphere['u_show_atmo'].value = 1.0 if show_atmo else 0.0
                self.prog_cubesphere['u_solar_irradiance'].value = float(solar_irradiance)
                self.prog_cubesphere['u_debug_mode'].value = int(self.debug_mode)

                tex_albedo.use(location=0)
                self.prog_cubesphere['u_tex_albedo'].value = 0
                tex_clouds.use(location=1)
                self.prog_cubesphere['u_tex_clouds'].value = 1

                if self.tier == EarthTier.HIGH_DETAIL_CUBESPHERE:
                    cam_pos_f64 = camera.get_eye_pos_f64() if hasattr(camera, 'get_eye_pos_f64') else np.asarray(eye_pos, dtype=np.float64)
                    visible_nodes = self.lod_manager.update(
                        cam_pos_f64, earth_pos, radius=earth_radius,
                        earth_rot_angle=earth_rot_angle, axial_tilt=axial_tilt,
                        fovy_deg=camera.fovy, viewport_height=height
                    )
                else:
                    visible_nodes = self.lod_manager.roots

                rendered_keys = set()
                for node in visible_nodes:
                    tile = self.tile_pool.get_or_create(node, self.world_data, self.prog_cubesphere, radius=earth_radius)
                    # If child tile is still building on worker thread, fall back to resident ancestor (Point 2)
                    if tile is None:
                        curr = node.parent
                        while curr is not None:
                            if curr.key in self.tile_pool.gpu_cache:
                                tile = self.tile_pool.gpu_cache[curr.key]
                                break
                            curr = curr.parent
                        if tile is None:
                            root = self.lod_manager.roots[node.face]
                            tile = self.tile_pool.get_or_create(root, self.world_data, self.prog_cubesphere, radius=earth_radius)

                    if tile is not None and tile.node_key not in rendered_keys:
                        rendered_keys.add(tile.node_key)
                        self.prog_cubesphere['u_lod_level'].value = float(node.level)
                        tile.vao.render()
                        rendered_count += 1
                        current_max_lod = max(current_max_lod, node.level)

            except Exception as e:
                print(f"[EarthRenderer] CubeSphere pass failed: {e}")
                rendered_count = 0

        # Point 2: Guaranteed Fallback
        # If no tiles rendered or explicitly in emergency tier -> render EMERGENCY_SPHERE
        if rendered_count == 0 or self.tier == EarthTier.EMERGENCY_SPHERE:
            tier_used = EarthTier.EMERGENCY_SPHERE
            m_earth = mat4_mul(
                translate(earth_pos[0], earth_pos[1], earth_pos[2]),
                rotate_z(-axial_tilt),
                rotate_y(earth_rot_angle),
                scale(earth_radius, earth_radius, earth_radius)
            )

            self.prog_earth['u_model'].write(mat4_bytes(m_earth))
            self.prog_earth['u_view'].write(mat4_bytes(view))
            self.prog_earth['u_proj'].write(mat4_bytes(proj))
            self.prog_earth['u_camera_pos'].value = tuple(eye_pos)
            self.prog_earth['u_sun_pos'].value = (0.0, 0.0, 0.0)
            self.prog_earth['u_cloud_offset'].value = (sim_time_sec * 0.0003) % 1.0
            self.prog_earth['u_show_clouds'].value = 1.0 if show_clouds else 0.0
            self.prog_earth['u_show_atmo'].value = 1.0 if show_atmo else 0.0
            self.prog_earth['u_solar_irradiance'].value = float(solar_irradiance)
            self.prog_earth['u_debug_mode'].value = int(self.debug_mode)

            tex_albedo.use(location=0)
            self.prog_earth['u_tex_albedo'].value = 0
            tex_clouds.use(location=1)
            self.prog_earth['u_tex_clouds'].value = 1

            self.emergency_vao.render()
            rendered_count = 1

        if self.debug_mode == 2:
            self.ctx.wireframe = False

        self.telemetry = {
            "rendered_tiles": rendered_count,
            "visible_nodes": len(visible_nodes) if visible_nodes else 1,
            "tier_in_use": tier_used,
            "max_active_lod": current_max_lod,
            "upload_queue": self.tile_pool.ready_cpu_meshes.qsize(),
        }


class SceneRenderer:
    """
    Hardware-accelerated ModernGL 3D scene renderer.
    Renders Deep Space, Sun, Earth (via EarthRenderer), Moon, Orbits, Grid, and 2D UI Overlay.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int, world_data: dict):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.world_data = world_data

        # Configure OpenGL state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = 'back'

        # Shaders
        self.prog_stars = self.ctx.program(vertex_shader=STARFIELD_VS, fragment_shader=STARFIELD_FS)
        self.prog_sun = self.ctx.program(vertex_shader=SUN_VS, fragment_shader=SUN_FS)
        self.prog_earth = self.ctx.program(vertex_shader=PLANET_VS, fragment_shader=EARTH_FS)
        self.prog_moon = self.ctx.program(vertex_shader=PLANET_VS, fragment_shader=MOON_FS)
        self.prog_line = self.ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.prog_overlay = self.ctx.program(vertex_shader=OVERLAY_VS, fragment_shader=OVERLAY_FS)
        self.prog_cubesphere = self.ctx.program(vertex_shader=CUBESPHERE_TERRAIN_VS, fragment_shader=CUBESPHERE_TERRAIN_FS)

        # GPU Hardware Detection & Optimization
        self.gpu_info = get_gpu_hardware_info(self.ctx)

        # Earth Renderer Subsystem (Points 1, 2, 44)
        self.earth_renderer = EarthRenderer(
            self.ctx,
            self.world_data,
            self.prog_cubesphere,
            self.prog_earth,
            self.gpu_info
        )

        # Build geometry
        self._init_starfield()
        self._init_spheres()
        self._init_orbit_lines()
        self._init_grid_lines()
        self._init_overlay_quad()

        # Textures
        self._init_textures()

        # Visual toggles (Point 43: Grid defaults to OFF)
        self.show_orbits = True
        self.show_grid = False
        self.show_clouds = True
        self.show_atmo = True

    def resize(self, width: int, height: int):
        self.width = max(100, width)
        self.height = max(100, height)
        self.ctx.viewport = (0, 0, self.width, self.height)
        if hasattr(self, 'tex_overlay'):
            self.tex_overlay.release()
        self.tex_overlay = self.ctx.texture((self.width, self.height), 4)
        self.tex_overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _init_starfield(self):
        rng = np.random.default_rng(16001)
        n_stars = 3000
        # Distribute on outer sphere
        u = rng.uniform(0.0, 1.0, n_stars)
        v = rng.uniform(0.0, 1.0, n_stars)
        theta = 2.0 * math.pi * u
        phi = np.arccos(2.0 * v - 1.0)
        r = 1200.0

        x = (r * np.sin(phi) * np.cos(theta)).astype(np.float32)
        y = (r * np.sin(phi) * np.sin(theta)).astype(np.float32)
        z = (r * np.cos(phi)).astype(np.float32)
        brightness = rng.uniform(0.3, 1.0, n_stars).astype(np.float32)

        star_data = np.stack([x, y, z, brightness], axis=-1).astype(np.float32)
        self.vbo_stars = self.ctx.buffer(star_data.tobytes())
        self.vao_stars = self.ctx.vertex_array(
            self.prog_stars,
            [(self.vbo_stars, '3f 1f', 'in_position', 'in_brightness')]
        )
        self.star_count = n_stars

    def _init_spheres(self):
        # Earth mesh
        earth_verts, earth_idx = create_sphere_mesh(lat_segments=72, lon_segments=144, radius=1.0)
        self.earth_vbo = self.ctx.buffer(earth_verts.tobytes())
        self.earth_ibo = self.ctx.buffer(earth_idx.tobytes())
        self.earth_vao = self.ctx.vertex_array(
            self.prog_earth,
            [(self.earth_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
            self.earth_ibo
        )

        # Sun mesh (slightly lower poly count since it's emissive)
        sun_verts, sun_idx = create_sphere_mesh(lat_segments=48, lon_segments=96, radius=1.0)
        self.sun_vbo = self.ctx.buffer(sun_verts.tobytes())
        self.sun_ibo = self.ctx.buffer(sun_idx.tobytes())
        self.sun_vao = self.ctx.vertex_array(
            self.prog_sun,
            [(self.sun_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
            self.sun_ibo
        )

        # Moon mesh
        moon_verts, moon_idx = create_sphere_mesh(lat_segments=36, lon_segments=72, radius=1.0)
        self.moon_vbo = self.ctx.buffer(moon_verts.tobytes())
        self.moon_ibo = self.ctx.buffer(moon_idx.tobytes())
        self.moon_vao = self.ctx.vertex_array(
            self.prog_moon,
            [(self.moon_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
            self.moon_ibo
        )

    def _init_orbit_lines(self):
        # Earth orbit around Sun (radius = 90.0)
        pts = []
        n_seg = 256
        for i in range(n_seg + 1):
            angle = 2.0 * math.pi * (i / n_seg)
            pts.append([math.cos(angle) * 90.0, 0.0, math.sin(angle) * 90.0])
        earth_orbit_pts = np.array(pts, dtype=np.float32)
        self.vbo_earth_orbit = self.ctx.buffer(earth_orbit_pts.tobytes())
        self.vao_earth_orbit = self.ctx.vertex_array(
            self.prog_line,
            [(self.vbo_earth_orbit, '3f', 'in_position')]
        )
        self.earth_orbit_points = n_seg + 1

        # Moon orbit around Earth (radius = 12.0)
        m_pts = []
        for i in range(n_seg + 1):
            angle = 2.0 * math.pi * (i / n_seg)
            # 5 degree inclination for lunar orbit
            inc = math.radians(5.14)
            x = math.cos(angle) * 12.0
            y = math.sin(angle) * math.sin(inc) * 12.0
            z = math.sin(angle) * math.cos(inc) * 12.0
            m_pts.append([x, y, z])
        moon_orbit_pts = np.array(m_pts, dtype=np.float32)
        self.vbo_moon_orbit = self.ctx.buffer(moon_orbit_pts.tobytes())
        self.vao_moon_orbit = self.ctx.vertex_array(
            self.prog_line,
            [(self.vbo_moon_orbit, '3f', 'in_position')]
        )
        self.moon_orbit_points = n_seg + 1

    def _init_grid_lines(self):
        # Spatial reference grid on ecliptic plane Y=0
        lines = []
        grid_size = 140.0
        step = 14.0
        r = int(grid_size / step)
        for i in range(-r, r + 1):
            pos = i * step
            # Line parallel to Z
            lines.append([-grid_size, 0.0, pos])
            lines.append([grid_size, 0.0, pos])
            # Line parallel to X
            lines.append([pos, 0.0, -grid_size])
            lines.append([pos, 0.0, grid_size])

        grid_data = np.array(lines, dtype=np.float32)
        self.vbo_grid = self.ctx.buffer(grid_data.tobytes())
        self.vao_grid = self.ctx.vertex_array(
            self.prog_line,
            [(self.vbo_grid, '3f', 'in_position')]
        )
        self.grid_point_count = len(lines)

    def _init_overlay_quad(self):
        # Fullscreen quad in NDC [-1, 1]
        quad_verts = np.array([
            # pos(x, y), uv(u, v)
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
             1.0, -1.0,  1.0, 1.0,
        ], dtype=np.float32)
        self.vbo_overlay = self.ctx.buffer(quad_verts.tobytes())
        self.vao_overlay = self.ctx.vertex_array(
            self.prog_overlay,
            [(self.vbo_overlay, '2f 2f', 'in_pos', 'in_uv')]
        )
        self.tex_overlay = self.ctx.texture((self.width, self.height), 4)
        self.tex_overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _init_textures(self):
        # 1. Earth albedo texture
        albedo_rgba = generate_earth_albedo_texture(self.world_data)
        ah, aw = albedo_rgba.shape[:2]
        self.tex_earth_albedo = self.ctx.texture((aw, ah), 4, albedo_rgba.tobytes())
        self.tex_earth_albedo.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # 2. Dynamic clouds texture
        cloud_rgba = generate_cloud_texture(h=192, w=384)
        ch, cw = cloud_rgba.shape[:2]
        self.tex_clouds = self.ctx.texture((cw, ch), 4, cloud_rgba.tobytes())
        self.tex_clouds.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_clouds.repeat_x = True

        # 3. Lunar surface texture
        lunar_rgb = generate_lunar_texture(h=128, w=256)
        lh, lw = lunar_rgb.shape[:2]
        self.tex_lunar = self.ctx.texture((lw, lh), 3, lunar_rgb.tobytes())
        self.tex_lunar.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Apply high-performance anisotropic texture filtering
        optimize_texture(self.tex_earth_albedo, max_anisotropy=16.0)
        optimize_texture(self.tex_clouds, max_anisotropy=8.0)
        optimize_texture(self.tex_lunar, max_anisotropy=8.0)

    def render(
        self,
        camera,
        sim_time_sec: float,
        earth_pos: np.ndarray,
        earth_rot_angle: float,
        moon_pos: np.ndarray,
        moon_rot_angle: float,
        solar_irradiance: float = 1.0,
        ui_surface = None,
    ):
        # Clear framebuffer
        self.ctx.clear(0.01, 0.015, 0.03, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<='

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        eye_pos = camera.get_eye_pos()

        # -------------------------------------------------------------
        # 1. Starfield Background
        # -------------------------------------------------------------
        self.ctx.disable(moderngl.CULL_FACE)
        self.prog_stars['u_view'].write(mat4_bytes(view))
        self.prog_stars['u_proj'].write(mat4_bytes(proj))
        self.vao_stars.render(mode=moderngl.POINTS)
        self.ctx.enable(moderngl.CULL_FACE)

        # -------------------------------------------------------------
        # 2. Spatial Reference Grid
        # -------------------------------------------------------------
        if self.show_grid:
            self.prog_line['u_model'].write(mat4_bytes(identity()))
            self.prog_line['u_view'].write(mat4_bytes(view))
            self.prog_line['u_proj'].write(mat4_bytes(proj))
            self.prog_line['u_line_color'].value = (0.12, 0.28, 0.45, 0.25)
            self.vao_grid.render(mode=moderngl.LINES, vertices=self.grid_point_count)

        # -------------------------------------------------------------
        # 3. Orbital Paths
        # -------------------------------------------------------------
        if self.show_orbits:
            # Earth orbit around Sun
            self.prog_line['u_model'].write(mat4_bytes(identity()))
            self.prog_line['u_view'].write(mat4_bytes(view))
            self.prog_line['u_proj'].write(mat4_bytes(proj))
            self.prog_line['u_line_color'].value = (0.22, 0.65, 0.95, 0.45)
            self.vao_earth_orbit.render(mode=moderngl.LINE_STRIP, vertices=self.earth_orbit_points)

            # Moon orbit around Earth
            m_model = translate(earth_pos[0], earth_pos[1], earth_pos[2])
            self.prog_line['u_model'].write(mat4_bytes(m_model))
            self.prog_line['u_view'].write(mat4_bytes(view))
            self.prog_line['u_proj'].write(mat4_bytes(proj))
            self.prog_line['u_line_color'].value = (0.65, 0.75, 0.85, 0.35)
            self.vao_moon_orbit.render(mode=moderngl.LINE_STRIP, vertices=self.moon_orbit_points)

        # -------------------------------------------------------------
        # 4. Sun (at Origin)
        # -------------------------------------------------------------
        sun_radius = 12.0
        sun_model = scale(sun_radius, sun_radius, sun_radius)

        self.prog_sun['u_model'].write(mat4_bytes(sun_model))
        self.prog_sun['u_view'].write(mat4_bytes(view))
        self.prog_sun['u_proj'].write(mat4_bytes(proj))
        self.prog_sun['u_camera_pos'].value = tuple(eye_pos)
        self.prog_sun['u_time'].value = sim_time_sec * 0.1
        self.sun_vao.render()

        # -------------------------------------------------------------
        # 5. Earth (via Guaranteed EarthRenderer System, Points 1, 2, 44)
        # -------------------------------------------------------------
        self.earth_renderer.render(
            camera=camera,
            sim_time_sec=sim_time_sec,
            earth_pos=earth_pos,
            earth_rot_angle=earth_rot_angle,
            solar_irradiance=solar_irradiance,
            show_clouds=self.show_clouds,
            show_atmo=self.show_atmo,
            tex_albedo=self.tex_earth_albedo,
            tex_clouds=self.tex_clouds,
            width=self.width,
            height=self.height,
        )

        # -------------------------------------------------------------
        # 6. Moon
        # -------------------------------------------------------------
        moon_radius = 1.4
        m_moon = mat4_mul(
            translate(moon_pos[0], moon_pos[1], moon_pos[2]),
            rotate_y(moon_rot_angle),
            scale(moon_radius, moon_radius, moon_radius)
        )

        self.prog_moon['u_model'].write(mat4_bytes(m_moon))
        self.prog_moon['u_view'].write(mat4_bytes(view))
        self.prog_moon['u_proj'].write(mat4_bytes(proj))
        self.prog_moon['u_sun_pos'].value = (0.0, 0.0, 0.0)

        self.tex_lunar.use(location=0)
        self.prog_moon['u_tex_lunar'].value = 0

        self.moon_vao.render()

        # -------------------------------------------------------------
        # 7. 2D Workstation UI Overlay Pass
        # -------------------------------------------------------------
        if ui_surface is not None:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            import pygame
            raw_data = pygame.image.tostring(ui_surface, 'RGBA', False)
            self.tex_overlay.write(raw_data)
            self.tex_overlay.use(location=0)
            self.prog_overlay['u_ui_texture'].value = 0
            self.vao_overlay.render(mode=moderngl.TRIANGLE_STRIP)

            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
