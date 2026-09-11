import math
import sys
import numpy as np

from .storage import load_world


def _bilinear(field, lat, lon):
    h, w = field.shape
    fy = (lat + math.pi / 2) / math.pi * (h - 1)
    fx = (lon + math.pi) / (2 * math.pi) * w

    y0 = np.clip(np.floor(fy).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0f = np.floor(fx).astype(np.int32)
    x0 = np.mod(x0f, w)
    x1 = np.mod(x0f + 1, w)
    ty = (fy - y0).astype(np.float32)
    tx = (fx - x0f).astype(np.float32)

    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty)


def _nearest(field, lat, lon):
    h, w = field.shape
    fy = (lat + math.pi / 2) / math.pi * (h - 1)
    fx = (lon + math.pi) / (2 * math.pi) * w
    yi = np.clip(np.rint(fy).astype(np.int32), 0, h - 1)
    xi = np.mod(np.rint(fx).astype(np.int32), w)
    return field[yi, xi]


def render_frame_rgb(world, size=700, lon_offset=0.0, lat_tilt=0.0, zoom=1.0):
    """Pure NumPy renderer. Returns an RGB uint8 image and visible mask.

    Keeping this function independent from Pygame makes the rendering path
    testable without a graphics driver or windowing system.
    """
    elevation = world['elevation']
    land_field = world['land_mask'].astype(np.float32)
    precipitation = world['precipitation']
    biome_field = world['biome'].astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    r = size * 0.42 * zoom
    nx = (xx - cx) / r
    ny = (yy - cy) / r
    rr2 = nx * nx + ny * ny
    visible = rr2 <= 1.0
    nz = np.zeros_like(nx, dtype=np.float32)
    nz[visible] = np.sqrt(np.maximum(0.0, 1.0 - rr2[visible]))

    lon = np.arctan2(nx, nz) + lon_offset
    lat0 = np.arcsin(np.clip(ny, -1.0, 1.0))
    y3 = np.sin(lat0)
    z3 = np.cos(lat0) * nz
    lat = np.arcsin(np.clip(y3 * math.cos(lat_tilt) - z3 * math.sin(lat_tilt), -1.0, 1.0))

    elev = _bilinear(elevation, lat, lon)
    land = _nearest(land_field, lat, lon) > 0.5
    precip = _bilinear(precipitation, lat, lon)
    biome = _nearest(biome_field, lat, lon).astype(np.uint8)

    img = np.zeros((size, size, 3), dtype=np.float32)

    # Ocean depth / elevation gradient.
    depth = np.clip(1.0 - elev / 0.5, 0.0, 1.0)
    img[..., 0] = 7 + 22 * (1 - depth)
    img[..., 1] = 28 + 60 * (1 - depth)
    img[..., 2] = 75 + 105 * (1 - depth)

    palette = np.array([
        [20, 65, 120],
        [225, 235, 235],
        [180, 145, 75],
        [125, 155, 65],
        [45, 105, 48],
        [25, 120, 55],
        [125, 130, 105],
    ], dtype=np.float32)
    img[land] = palette[np.clip(biome, 0, 6)][land]

    # Relief and cloud/haze contribution.
    relief = np.clip((elev - 0.5) * 2.0, 0.0, 1.0)
    img += relief[..., None] * 38.0

    cloud = np.clip(
        0.55 * precip
        + 0.45 * (np.sin(lon * 2.7 + lat * 4.0) * 0.5)
        + 0.25 - 0.65,
        0.0,
        1.0,
    )
    img = img * (1.0 - cloud[..., None] * 0.20) + 235.0 * cloud[..., None] * 0.20

    # Directional sunlight using the sphere normal.
    lx, ly, lz = -0.55, 0.30, 0.78
    light = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    img *= (0.18 + 0.92 * light)[..., None]

    # Thin atmospheric rim.
    rim = np.clip(1.0 - nz, 0.0, 1.0) ** 2
    img += rim[..., None] * np.array([12, 28, 45], dtype=np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img[~visible] = 0
    return img, visible


class SphereViewer:
    def __init__(self, world_path):
        self.world = load_world(world_path)
        self.width, self.height = 1100, 760
        self.zoom = 1.0
        self.lon_offset = 0.0
        self.lat_tilt = 0.0
        self.render_size = 700
        self._pygame = None

    def start(self):
        import pygame
        self._pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption('RLLS 16 - Canonical World Viewer')
        self.clock = pygame.time.Clock()
        rng = np.random.default_rng(16001)
        self.stars = [
            (int(rng.integers(0, self.width)), int(rng.integers(0, self.height)), int(rng.integers(1, 3)))
            for _ in range(700)
        ]

    def draw(self):
        pygame = self._pygame
        self.screen.fill((2, 5, 12))
        for x, y, s in self.stars:
            if x < self.width and y < self.height:
                pygame.draw.circle(self.screen, (150, 165, 185), (x, y), s)

        rgb, _ = render_frame_rgb(
            self.world,
            size=self.render_size,
            lon_offset=self.lon_offset,
            lat_tilt=self.lat_tilt,
            zoom=1.0,
        )
        sphere = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        diameter = int(min(self.width, self.height) * 0.72 * self.zoom)
        sphere = pygame.transform.smoothscale(sphere, (diameter, diameter))
        sphere.set_colorkey((0, 0, 0))
        self.screen.blit(sphere, (self.width // 2 - diameter // 2, self.height // 2 - diameter // 2))

        font = pygame.font.SysFont('consolas', 18)
        small = pygame.font.SysFont('consolas', 14)
        self.screen.blit(font.render('RLLS 16  |  CANONICAL WORLD', True, (190, 220, 245)), (24, 22))
        self.screen.blit(small.render(
            f"Field: {self.world['elevation'].shape[1]} x {self.world['elevation'].shape[0]}   Seed: {self.world['metadata'].get('generator_seed', '?')}",
            True, (135, 160, 180)), (24, 48))
        self.screen.blit(small.render(
            'LEFT/RIGHT rotate   UP/DOWN tilt   +/- zoom   ESC exit',
            True, (135, 160, 180)), (24, self.height - 30))
        pygame.display.flip()

    def run(self):
        self.start()
        pygame = self._pygame
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.zoom = min(1.5, self.zoom + 0.08)
                    elif e.key == pygame.K_MINUS:
                        self.zoom = max(0.65, self.zoom - 0.08)

            k = pygame.key.get_pressed()
            if k[pygame.K_LEFT]: self.lon_offset -= 0.025
            if k[pygame.K_RIGHT]: self.lon_offset += 0.025
            if k[pygame.K_UP]: self.lat_tilt = min(0.7, self.lat_tilt + 0.012)
            if k[pygame.K_DOWN]: self.lat_tilt = max(-0.7, self.lat_tilt - 0.012)

            self.draw()
            self.clock.tick(30)
        pygame.quit()


def run(path='worlds/canonical_world.npz'):
    SphereViewer(path).run()


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'worlds/canonical_world.npz')
