import numpy as np

# Minecraft-inspired idea:
# terrain is treated as a continuous scalar field rather than a collection
# of independent blocks. We combine broad continental structure, erosion/
# ridge terms, and fine detail, then interpolate the field smoothly.

def _hash2(ix, iy, seed):
    x = np.asarray(ix, dtype=np.int64)
    y = np.asarray(iy, dtype=np.int64)
    h = x * np.int64(374761393) + y * np.int64(668265263) + np.int64(seed) * np.int64(1442695041)
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & np.int64(0x7fffffff)).astype(np.float64) / 1073741823.5 - 1.0

def value_noise_2d(x, y, seed, grid):
    """Smooth deterministic value noise sampled at arbitrary x/y."""
    gx = x * grid
    gy = y * grid
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    tx = gx - x0
    ty = gy - y0

    # Quintic interpolation gives smooth derivatives at cell boundaries.
    sx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
    sy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)

    n00 = _hash2(x0,     y0,     seed)
    n10 = _hash2(x0 + 1, y0,     seed)
    n01 = _hash2(x0,     y0 + 1, seed)
    n11 = _hash2(x0 + 1, y0 + 1, seed)

    nx0 = n00 + (n10 - n00) * sx
    nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy

def fbm_2d(x, y, seed, base_grid=2.0, octaves=5, lacunarity=2.0, gain=0.5):
    total = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = base_grid
    norm = 0.0

    for octave in range(octaves):
        total += amplitude * value_noise_2d(
            x, y, seed + octave * 1013, frequency
        )
        norm += amplitude
        frequency *= lacunarity
        amplitude *= gain

    return total / norm

def ridged_fbm(x, y, seed, octaves=4):
    n = fbm_2d(x, y, seed, base_grid=2.5, octaves=octaves)
    return 1.0 - np.abs(n)
