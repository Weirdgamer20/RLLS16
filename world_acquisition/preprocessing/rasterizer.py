"""
RLLS16 Geographical Rasterizer.
Converts Natural Earth vector GeoJSON datasets into simulation-ready 2D raster grids.
"""

from pathlib import Path
import json
import numpy as np
import pygame

def load_geojson(filepath: Path | str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def rasterize_polygons(
    geojson_data: dict,
    width: int = 512,
    height: int = 256,
    invert_holes: bool = True
) -> np.ndarray:
    """
    Rasterizes GeoJSON Polygon and MultiPolygon features onto a 2D grid
    spanning Longitude [-180, 180] and Latitude [-90, 90].
    Returns a 2D boolean numpy array (height, width).
    """
    pygame.init()
    surf = pygame.Surface((width, height))
    surf.fill((0, 0, 0))

    def to_pixel(lon: float, lat: float) -> tuple[float, float]:
        px = (lon + 180.0) / 360.0 * (width - 1)
        py = (90.0 - lat) / 180.0 * (height - 1)
        return px, py

    for feat in geojson_data.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "")

        if gtype == "Polygon":
            poly_list = [coords]
        elif gtype == "MultiPolygon":
            poly_list = coords
        else:
            continue

        for poly in poly_list:
            if not poly or len(poly) == 0:
                continue
            # Exterior ring (fill with 255)
            ext_pts = [to_pixel(p[0], p[1]) for p in poly[0]]
            if len(ext_pts) >= 3:
                pygame.draw.polygon(surf, (255, 255, 255), ext_pts)

            # Interior rings / holes (fill with 0)
            if invert_holes and len(poly) > 1:
                for hole in poly[1:]:
                    hole_pts = [to_pixel(p[0], p[1]) for p in hole]
                    if len(hole_pts) >= 3:
                        pygame.draw.polygon(surf, (0, 0, 0), hole_pts)

    # Convert surface to 2D numpy array [height, width]
    # pygame surfarray is (width, height)
    arr = pygame.surfarray.array2d(surf)
    arr_2d = np.transpose(arr) > 0
    return arr_2d.astype(bool)

def rasterize_lines(
    geojson_data: dict,
    width: int = 512,
    height: int = 256,
    line_thickness: int = 1
) -> np.ndarray:
    """
    Rasterizes GeoJSON LineString and MultiLineString features (rivers, coastlines).
    Returns a 2D boolean numpy array (height, width).
    """
    pygame.init()
    surf = pygame.Surface((width, height))
    surf.fill((0, 0, 0))

    def to_pixel(lon: float, lat: float) -> tuple[float, float]:
        px = (lon + 180.0) / 360.0 * (width - 1)
        py = (90.0 - lat) / 180.0 * (height - 1)
        return px, py

    for feat in geojson_data.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "")

        if gtype == "LineString":
            lines = [coords]
        elif gtype == "MultiLineString":
            lines = coords
        else:
            continue

        for line in lines:
            if len(line) < 2:
                continue
            pts = [to_pixel(p[0], p[1]) for p in line]
            pygame.draw.lines(surf, (255, 255, 255), False, pts, line_thickness)

    arr = pygame.surfarray.array2d(surf)
    arr_2d = np.transpose(arr) > 0
    return arr_2d.astype(bool)
