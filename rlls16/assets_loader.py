"""
RLLS16 2D Asset Loader & Cache.
Loads 16-bit SVG icons and cosmetics from the asset package,
caches rasterized surfaces, and applies nearest-neighbor scaling.
"""

from pathlib import Path
import io
import pygame

_ASSET_DIR = Path(__file__).parent / "assets" / "RLLS16_2D_World_Assets"
_SURFACE_CACHE: dict[tuple[str, int, int], pygame.Surface] = {}

def get_asset_path(subfolder: str, name: str) -> Path:
    if not name.endswith(".svg"):
        name = f"{name}.svg"
    return _ASSET_DIR / subfolder / name

def get_icon(name: str, size: tuple[int, int] = (24, 24)) -> pygame.Surface:
    """
    Load and return a cached 16-bit icon surface scaled to the requested size.
    Uses nearest-neighbor scaling for crisp pixel edges.
    """
    key = ("icon", name, size[0], size[1])
    if key in _SURFACE_CACHE:
        return _SURFACE_CACHE[key]

    path = get_asset_path("icons", name)
    if not path.exists():
        # Fallback to a procedural colored surface if icon not found
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill((46, 216, 232, 180))
        _SURFACE_CACHE[key] = surf
        return surf

    try:
        raw_surf = pygame.image.load(str(path))
        if raw_surf.get_size() != size:
            # Nearest-neighbor scale for 16-bit aesthetics
            scaled = pygame.transform.scale(raw_surf, size)
        else:
            scaled = raw_surf
    except Exception:
        scaled = pygame.Surface(size, pygame.SRCALPHA)
        scaled.fill((46, 216, 232, 180))

    _SURFACE_CACHE[key] = scaled
    return scaled

def get_cosmetic(name: str, size: tuple[int, int] | None = None) -> pygame.Surface:
    """Load and return a cached cosmetic surface."""
    w = size[0] if size else 0
    h = size[1] if size else 0
    key = ("cosmetic", name, w, h)
    if key in _SURFACE_CACHE:
        return _SURFACE_CACHE[key]

    path = get_asset_path("cosmetics", name)
    if not path.exists():
        fallback_size = size or (32, 32)
        surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
        _SURFACE_CACHE[key] = surf
        return surf

    try:
        raw_surf = pygame.image.load(str(path))
        if size is not None and raw_surf.get_size() != size:
            scaled = pygame.transform.scale(raw_surf, size)
        else:
            scaled = raw_surf
    except Exception:
        fallback_size = size or (32, 32)
        scaled = pygame.Surface(fallback_size, pygame.SRCALPHA)

    _SURFACE_CACHE[key] = scaled
    return scaled

def preload_all_icons(sizes: list[tuple[int, int]] = [(16, 16), (24, 24), (32, 32)]):
    """Preload all icons into cache to avoid any runtime disk read or scaling."""
    icons_dir = _ASSET_DIR / "icons"
    if not icons_dir.exists():
        return
    for p in icons_dir.glob("*.svg"):
        for sz in sizes:
            get_icon(p.stem, sz)
