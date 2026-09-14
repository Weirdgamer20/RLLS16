"""
RLLS16 2D Asset Loader & Hardened Surface Cache.
Implements one-time SVG rasterization -> native Surface cache -> nearest-neighbor scaling.
Eliminates repeated SVG parsing during simulation frames and provides crisp 16-bit procedural fallbacks.
"""

from pathlib import Path
import pygame

_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "RLLS16_2D_World_Assets"

# Master caches:
# 1. Base native raster surfaces loaded once from SVG sources
_BASE_SURFACES: dict[str, pygame.Surface] = {}
# 2. Resolution-specific scaled surfaces (nearest-neighbor scaled)
_SCALED_SURFACES: dict[tuple[str, str, int, int], pygame.Surface] = {}


def _create_procedural_fallback(name: str, size: tuple[int, int]) -> pygame.Surface:
    """Crisp procedural 16-bit fallback sprite if SVG asset is missing or malformed."""
    surf = pygame.Surface(size, pygame.SRCALPHA)
    w, h = size

    # Palette
    color_map = {
        "human": (245, 180, 70),
        "tree": (46, 185, 75),
        "animal": (220, 120, 60),
        "water": (46, 175, 240),
        "shelter": (180, 140, 95),
        "fire": (245, 85, 45),
        "food": (215, 190, 60),
        "mountain": (150, 155, 165),
        "rain": (70, 140, 220),
        "river": (42, 140, 168),
        "cursor_select": (46, 216, 232),
        "weather": (200, 220, 240),
    }
    col = color_map.get(name, (46, 216, 232))

    # Draw crisp 16-bit icon silhouette
    margin = max(1, w // 8)
    inner_rect = pygame.Rect(margin, margin, w - 2 * margin, h - 2 * margin)
    pygame.draw.rect(surf, col, inner_rect, border_radius=max(2, w // 6))
    pygame.draw.rect(surf, (255, 255, 255, 180), inner_rect, width=1, border_radius=max(2, w // 6))

    return surf


def _load_base_surface(subfolder: str, name: str) -> pygame.Surface:
    """One-time load of SVG asset into a native hardware/software Surface."""
    key = f"{subfolder}/{name}"
    if key in _BASE_SURFACES:
        return _BASE_SURFACES[key]

    ext_name = name if name.endswith(".svg") else f"{name}.svg"
    path = _ASSET_DIR / subfolder / ext_name

    if not path.exists():
        fallback = _create_procedural_fallback(name, (32, 32))
        _BASE_SURFACES[key] = fallback
        return fallback

    try:
        raw_surf = pygame.image.load(str(path))
        # Ensure surface has per-pixel alpha channel
        base_surf = raw_surf.convert_alpha() if pygame.display.get_surface() else raw_surf
    except Exception:
        base_surf = _create_procedural_fallback(name, (32, 32))

    _BASE_SURFACES[key] = base_surf
    return base_surf


def get_icon(name: str, size: tuple[int, int] = (24, 24)) -> pygame.Surface:
    """
    Retrieves resolution-specific cached 16-bit icon surface.
    Never parses SVG during render frames.
    Uses nearest-neighbor scaling to maintain authentic pixel art crispness.
    """
    key = ("icon", name, size[0], size[1])
    if key in _SCALED_SURFACES:
        return _SCALED_SURFACES[key]

    base = _load_base_surface("icons", name)
    if base.get_size() != size:
        scaled = pygame.transform.scale(base, size)
    else:
        scaled = base

    _SCALED_SURFACES[key] = scaled
    return scaled


def get_cosmetic(name: str, size: tuple[int, int] | None = None) -> pygame.Surface:
    """Retrieves cached cosmetic surface."""
    w = size[0] if size else 0
    h = size[1] if size else 0
    key = ("cosmetic", name, w, h)
    if key in _SCALED_SURFACES:
        return _SCALED_SURFACES[key]

    base = _load_base_surface("cosmetics", name)
    if size is not None and base.get_size() != size:
        scaled = pygame.transform.scale(base, size)
    else:
        scaled = base

    _SCALED_SURFACES[key] = scaled
    return scaled


def preload_all_icons(sizes: list[tuple[int, int]] = [(16, 16), (20, 20), (24, 24), (28, 28), (32, 32), (40, 40)]):
    """
    Preload and pre-scale all icons into memory at startup.
    Ensures zero file I/O and zero SVG parsing during interactive simulation.
    """
    icons_dir = _ASSET_DIR / "icons"
    if not icons_dir.exists():
        return
    for p in icons_dir.glob("*.svg"):
        icon_name = p.stem
        # Pre-rasterize base
        _load_base_surface("icons", icon_name)
        # Pre-scale for standard sizes
        for sz in sizes:
            get_icon(icon_name, sz)
