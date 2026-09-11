"""
RLLS 16 — Terrain Tile Mesh Generation, Displacements, Skirts, and GPU Pool.

Generates NxN vertex mesh grids for CubeSphere quadtree nodes.
Samples authoritative canonical elevation, applies deterministic procedural octave
refinement at higher LOD levels, adds defensive perimeter skirts, and manages an
LRU GPU VBO/VAO cache to prevent memory allocation churn.
"""

import math
from collections import OrderedDict
import numpy as np
import moderngl

from .cubesphere import cube_to_sphere, sphere_to_latlon, sample_canonical_field
from ..noise import fbm_2d


# Standard biome color palette (matched to canonical specifications)
# 0=ocean, 1=ice, 2=desert, 3=grassland, 4=forest, 5=rainforest, 6=tundra
BIOME_PALETTE = np.array([
    [25, 75, 140],    # 0 ocean fallback
    [230, 240, 245],  # 1 ice / glaciers
    [210, 180, 105],  # 2 arid / desert
    [135, 175, 75],   # 3 temperate grassland
    [48, 120, 52],    # 4 temperate forest
    [20, 95, 45],     # 5 tropical rainforest
    [140, 145, 125],  # 6 tundra / highlands
], dtype=np.float32) / 255.0


def build_tile_mesh_data(
    node,
    world_data: dict,
    grid_size: int = 16,
    radius: float = 5.0,
    terrain_amp: float = 0.35,
    skirt_depth: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the vertex and index arrays for a quadtree node.
    Vertex layout (10 floats per vertex):
      x, y, z (position relative to planet center)
      nx, ny, nz (normal vector)
      u, v (spherical lon/lat mapped to [0, 1])
      elev, land_mask (terrain scalars)
    """
    u_vals = np.linspace(node.u_min, node.u_max, grid_size, dtype=np.float32)
    v_vals = np.linspace(node.v_min, node.v_max, grid_size, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_vals, v_vals)

    # 1. Normalized sphere coordinates
    sphere_pts = cube_to_sphere(node.face, u_grid, v_grid) # (grid_size, grid_size, 3)

    # 2. Convert to lat/lon for canonical field sampling
    lat, lon = sphere_to_latlon(sphere_pts)

    # 3. Sample canonical elevation and land mask
    elev_canonical = sample_canonical_field(world_data["elevation"], lat, lon)
    if np.issubdtype(elev_canonical.dtype, np.integer) or np.max(elev_canonical) > 1.5:
        elev_canonical = elev_canonical.astype(np.float32) / 65535.0
    land_mask = sample_canonical_field(world_data["land_mask"].astype(np.float32), lat, lon) > 0.5

    # 4. Optional deterministic procedural refinement at deeper LOD levels
    elev = elev_canonical.copy()
    if node.level >= 2:
        # Micro-scale fractal detail scaled by LOD depth
        sx = sphere_pts[..., 0] * (4.0 * (2 ** (node.level - 2)))
        sy = sphere_pts[..., 1] * (4.0 * (2 ** (node.level - 2)))
        detail = fbm_2d(sx, sy, seed=16001 + node.level, base_grid=1.5, octaves=3)
        # Refinement is active primarily on land
        land_factor = np.where(land_mask, 1.0, 0.05).astype(np.float32)
        refinement_scale = (0.04 / (2 ** (node.level - 1)))
        elev += (detail - 0.5) * refinement_scale * land_factor

        # Deep zoom: add high-frequency micro crags and ridges on land
        if node.level >= 5:
            hx = sphere_pts[..., 0] * (16.0 * (2 ** (node.level - 4)))
            hy = sphere_pts[..., 1] * (16.0 * (2 ** (node.level - 4)))
            micro = fbm_2d(hx, hy, seed=32003 + node.level, base_grid=2.0, octaves=2)
            micro_scale = (0.012 / (2 ** (node.level - 1)))
            elev += (micro - 0.5) * micro_scale * land_factor

    # 5. Radial vertex displacement
    # Sea level is at elevation 0.50
    # Oceans have slight bathymetry, land rises above radius
    displaced_radius = radius + (elev - 0.50) * terrain_amp
    positions = sphere_pts * displaced_radius[..., None] # (grid_size, grid_size, 3)

    # 6. High-detail surface normals via central differences of displaced mesh
    dv = np.zeros_like(positions)
    du = np.zeros_like(positions)

    dv[1:-1, :] = positions[2:, :] - positions[:-2, :]
    dv[0, :] = (positions[1, :] - positions[0, :]) * 2.0
    dv[-1, :] = (positions[-1, :] - positions[-2, :]) * 2.0

    du[:, 1:-1] = positions[:, 2:] - positions[:, :-2]
    du[:, 0] = (positions[:, 1] - positions[:, 0]) * 2.0
    du[:, -1] = (positions[:, -1] - positions[:, -2]) * 2.0

    computed_normals = np.cross(du, dv)
    dots = np.sum(computed_normals * sphere_pts, axis=-1, keepdims=True)
    sign = np.where(dots < 0.0, -1.0, 1.0)
    computed_normals = computed_normals * sign

    norm_len = np.linalg.norm(computed_normals, axis=-1, keepdims=True)
    computed_normals = np.where(norm_len > 1e-12, computed_normals / norm_len, sphere_pts)

    if node.level <= 1:
        blend = 0.35 + 0.35 * node.level
        normals = sphere_pts * (1.0 - blend) + computed_normals * blend
        n_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.where(n_len > 1e-12, normals / n_len, sphere_pts)
    else:
        normals = computed_normals

    # 7. Normalized UVs for texture / shader mapping
    u_tex = (lon + math.pi) / (2.0 * math.pi)
    v_tex = (lat + (math.pi / 2.0)) / math.pi

    # Combine into vertex attributes
    # [x, y, z, nx, ny, nz, u, v, elev, land]
    flat_pos = positions.reshape(-1, 3)
    flat_norm = normals.reshape(-1, 3)
    flat_uv = np.stack([u_tex.reshape(-1), v_tex.reshape(-1)], axis=-1)
    flat_scalars = np.stack([elev.reshape(-1), land_mask.astype(np.float32).reshape(-1)], axis=-1)

    grid_vertices = np.concatenate([flat_pos, flat_norm, flat_uv, flat_scalars], axis=-1).astype(np.float32)

    # 8. Indices for the interior grid with mathematically guaranteed outward CCW winding
    v_tl = positions[0, 0]
    v_tr = positions[0, 1]
    v_bl = positions[1, 0]
    # In right-handed space, check if (v_bl - v_tl) x (v_tr - v_tl) points in or out
    test_normal = np.cross(v_bl - v_tl, v_tr - v_tl)
    invert_winding = (np.dot(test_normal, v_tl) < 0)

    indices = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            tl = i * grid_size + j
            tr = tl + 1
            bl = (i + 1) * grid_size + j
            br = bl + 1

            if invert_winding:
                indices.extend([tl, tr, bl])
                indices.extend([tr, br, bl])
            else:
                indices.extend([tl, bl, tr])
                indices.extend([tr, bl, br])

    # 9. Perimeter Skirts to eliminate cracks across LOD boundaries (Point 8)
    num_grid_verts = len(grid_vertices)
    skirt_verts = []

    def get_skirt_vert(orig_v):
        skirt_v = orig_v.copy()
        r_current = float(np.linalg.norm(skirt_v[:3]))
        skirt_v[:3] = (skirt_v[:3] / max(r_current, 1e-6)) * (r_current - skirt_depth)
        return skirt_v

    edge_indices_top = [j for j in range(grid_size)]
    edge_indices_bottom = [(grid_size - 1) * grid_size + j for j in range(grid_size)]
    edge_indices_left = [i * grid_size for i in range(grid_size)]
    edge_indices_right = [i * grid_size + (grid_size - 1) for i in range(grid_size)]

    all_edges = [
        edge_indices_top,
        edge_indices_right,
        edge_indices_bottom[::-1],
        edge_indices_left[::-1]
    ]

    curr_skirt_idx = num_grid_verts
    tile_center = positions[grid_size // 2, grid_size // 2]

    for edge in all_edges:
        for k in range(len(edge) - 1):
            i0 = edge[k]
            i1 = edge[k + 1]

            v0 = grid_vertices[i0]
            v1 = grid_vertices[i1]

            skirt_v0 = get_skirt_vert(v0)
            skirt_v1 = get_skirt_vert(v1)

            skirt_verts.append(skirt_v0)
            skirt_verts.append(skirt_v1)

            s0 = curr_skirt_idx
            s1 = curr_skirt_idx + 1
            curr_skirt_idx += 2

            p0 = v0[:3]
            p1 = v1[:3]
            ps0 = skirt_v0[:3]
            wall_norm = np.cross(p1 - p0, ps0 - p0)
            outward_vec = p0 - tile_center
            if np.dot(wall_norm, outward_vec) > 0:
                indices.extend([i0, i1, s0])
                indices.extend([i1, s1, s0])
            else:
                indices.extend([i0, s0, i1])
                indices.extend([i1, s0, s1])

    if skirt_verts:
        all_vertices = np.concatenate([grid_vertices, np.array(skirt_verts, dtype=np.float32)], axis=0)
    else:
        all_vertices = grid_vertices

    return all_vertices, np.array(indices, dtype=np.uint32)


class TerrainTile:
    """Represents an active GPU-buffered terrain tile with support for buffer reuse."""
    __slots__ = ('node_key', 'vbo', 'ibo', 'vao', 'vertex_count', 'index_count', 'last_used_frame')

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program, vertices: np.ndarray, indices: np.ndarray, key: str):
        self.node_key = key
        self.last_used_frame = 0
        self.vertex_count = len(vertices)
        self.index_count = len(indices)

        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())

        self.vao = ctx.vertex_array(
            prog,
            [
                (self.vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_scalars')
            ],
            index_buffer=self.ibo,
            index_element_size=4,
        )

    def update_data(self, vertices: np.ndarray, indices: np.ndarray, key: str):
        """Reuse existing GPU buffer without reallocation if sizes match."""
        self.node_key = key
        self.last_used_frame = 0
        self.vertex_count = len(vertices)
        self.index_count = len(indices)

        v_bytes = vertices.tobytes()
        if len(v_bytes) == self.vbo.size:
            self.vbo.write(v_bytes)
        else:
            self.vbo.orphan(len(v_bytes))
            self.vbo.write(v_bytes)

        i_bytes = indices.tobytes()
        if len(i_bytes) == self.ibo.size:
            self.ibo.write(i_bytes)
        else:
            self.ibo.orphan(len(i_bytes))
            self.ibo.write(i_bytes)

    def release(self):
        try:
            self.vao.release()
            self.vbo.release()
            self.ibo.release()
        except Exception:
            pass


from concurrent.futures import ThreadPoolExecutor
import queue


class TerrainTilePool:
    """
    Minecraft-inspired chunk streaming pool with multi-tier caching and GPU buffer recycling:
    1. Active GPU Set: Currently rendered tiles in view frustum.
    2. GPU Cache & Recycled Buffer Pool: High-speed GPU cache with buffer reuse on eviction.
    3. CPU RAM Cache: Authoritative chunk geometry cached in RAM to restore visited areas with zero regeneration.
    4. Strict Per-Frame Budgeting: Never blocks the render loop.
    """

    def __init__(self, ctx: moderngl.Context, max_gpu_cached: int = 512, max_cpu_cached: int = 2048):
        self.ctx = ctx
        self.max_gpu_cached = max_gpu_cached
        self.max_cpu_cached = max_cpu_cached
        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.free_buffer_pool = []  # Recycled TerrainTile objects ready for reuse
        self.max_pool_size = 128
        self.pending_tasks = set()
        self.ready_cpu_meshes = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="TerrainChunkWorker")
        self.current_frame = 0

        # Telemetry metrics
        self.cache_hits_gpu = 0
        self.cache_hits_cpu = 0
        self.reused_buffers_count = 0
        self.total_generated_count = 0
        self.frame_uploads = 0

    def _worker_build_mesh(self, node, world_data: dict, grid_size: int, radius: float):
        try:
            verts, indices = build_tile_mesh_data(node, world_data, grid_size=grid_size, radius=radius)
            self.ready_cpu_meshes.put((node.key, verts, indices))
        except Exception as e:
            print(f"[TerrainTilePool] Worker error on chunk {node.key}: {e}")

    def _acquire_or_create_tile(self, prog: moderngl.Program, verts: np.ndarray, indices: np.ndarray, key: str) -> TerrainTile:
        """Acquire a recycled GPU tile from pool if available, else instantiate a new one."""
        if self.free_buffer_pool:
            tile = self.free_buffer_pool.pop()
            tile.update_data(verts, indices, key)
            self.reused_buffers_count += 1
            return tile
        return TerrainTile(self.ctx, prog, verts, indices, key)

    def _evict_oldest_gpu_tile(self):
        """Evict least-recently-used GPU tile into free buffer pool for recycling."""
        _, oldest_tile = self.gpu_cache.popitem(last=False)
        if len(self.free_buffer_pool) < self.max_pool_size:
            self.free_buffer_pool.append(oldest_tile)
        else:
            oldest_tile.release()

    def process_ready_uploads(self, prog: moderngl.Program, max_uploads: int = 6):
        """Process ready CPU meshes on the main render thread with strict work budget."""
        count = 0
        while count < max_uploads and not self.ready_cpu_meshes.empty():
            try:
                key, verts, indices = self.ready_cpu_meshes.get_nowait()
                if key in self.pending_tasks:
                    self.pending_tasks.remove(key)

                # Store in CPU RAM cache (never discard data on render distance exit)
                self.cpu_cache[key] = (verts, indices)
                if len(self.cpu_cache) > self.max_cpu_cached:
                    self.cpu_cache.popitem(last=False)

                # Upload to GPU using recycled buffer if possible
                if key not in self.gpu_cache:
                    while len(self.gpu_cache) >= self.max_gpu_cached:
                        self._evict_oldest_gpu_tile()

                    tile = self._acquire_or_create_tile(prog, verts, indices, key)
                    tile.last_used_frame = self.current_frame
                    self.gpu_cache[key] = tile

                count += 1
            except queue.Empty:
                break
        self.frame_uploads = count

    def get_or_create(self, node, world_data: dict, prog: moderngl.Program, radius: float = 5.0) -> TerrainTile | None:
        key = node.key

        # 1. Hit in GPU cache (Tier 1 & 2)
        if key in self.gpu_cache:
            tile = self.gpu_cache[key]
            tile.last_used_frame = self.current_frame
            self.gpu_cache.move_to_end(key)
            self.cache_hits_gpu += 1
            return tile

        # 2. Hit in CPU RAM cache (Tier 3: zero regeneration!)
        if key in self.cpu_cache:
            verts, indices = self.cpu_cache[key]
            while len(self.gpu_cache) >= self.max_gpu_cached:
                self._evict_oldest_gpu_tile()

            tile = self._acquire_or_create_tile(prog, verts, indices, key)
            tile.last_used_frame = self.current_frame
            self.gpu_cache[key] = tile
            self.cache_hits_cpu += 1
            return tile

        # 3. If LOD 0 (root tiles), generate synchronously to guarantee baseline rendering
        if node.level == 0:
            verts, indices = build_tile_mesh_data(node, world_data, grid_size=16, radius=radius)
            self.total_generated_count += 1
            self.cpu_cache[key] = (verts, indices)
            while len(self.gpu_cache) >= self.max_gpu_cached:
                self._evict_oldest_gpu_tile()
            tile = self._acquire_or_create_tile(prog, verts, indices, key)
            tile.last_used_frame = self.current_frame
            self.gpu_cache[key] = tile
            return tile

        # 4. Deep LOD -> enqueue to worker thread (asynchronous streaming)
        if key not in self.pending_tasks:
            self.pending_tasks.add(key)
            self.total_generated_count += 1
            self.executor.submit(self._worker_build_mesh, node, world_data, 16, radius)

        return None

    def step_frame(self):
        self.current_frame += 1

    def release_all(self):
        self.executor.shutdown(wait=False)
        for tile in self.gpu_cache.values():
            tile.release()
        for tile in self.free_buffer_pool:
            tile.release()
        self.gpu_cache.clear()
        self.free_buffer_pool.clear()
        self.cpu_cache.clear()

