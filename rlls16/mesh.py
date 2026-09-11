import numpy as np

def build_cube_sphere_mesh(world, radius=10.0, terrain_amplitude=0.42):
    """
    Convert the latitude/longitude fields into a continuous spherical mesh.

    This is the visual layer. The canonical data remains field-based.
    """
    lat = world["latitude"]
    lon = world["longitude"]
    elevation = world["elevation"]

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.sin(lon_grid)

    r = radius + (elevation - 0.5) * 2.0 * terrain_amplitude

    vertices = np.stack(
        [x * r, y * r, z * r],
        axis=-1,
    ).astype(np.float32)

    # Two triangles per grid cell. Wrap longitude.
    rows, cols = elevation.shape
    tris = []
    for i in range(rows - 1):
        a = i * cols
        b = (i + 1) * cols
        for j in range(cols):
            nj = (j + 1) % cols
            tris.append((a + j, b + j, b + nj))
            tris.append((a + j, b + nj, a + nj))

    indices = np.asarray(tris, dtype=np.uint32).reshape(-1)

    # Approximate radial normals are adequate for this first renderer.
    normals = vertices / np.maximum(
        np.linalg.norm(vertices, axis=2, keepdims=True), 1e-8
    )

    return vertices.reshape(-1, 3), normals.reshape(-1, 3), indices
