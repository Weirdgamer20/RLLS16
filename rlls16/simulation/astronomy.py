import math
import numpy as np

# Immutable Canonical Astronomical Constants
SUN_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
EARTH_ORBIT_RADIUS = 90.0          # Render distance units
EARTH_ORBIT_PERIOD_DAYS = 365.25   # Days for 1 revolution around Sun
EARTH_ROTATION_PERIOD_HOURS = 24.0 # Hours for 1 full diurnal rotation
EARTH_AXIAL_TILT_DEG = 23.44       # Earth obliquity to ecliptic
EARTH_AXIAL_TILT_RAD = math.radians(EARTH_AXIAL_TILT_DEG)

MOON_ORBIT_RADIUS = 12.0           # Render distance units from Earth center
MOON_ORBIT_PERIOD_DAYS = 27.32166  # Sidereal orbit period
MOON_ORBIT_INCLINATION_DEG = 5.14  # Inclination to ecliptic


from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class AstronomicalState(Mapping):
    """
    Immutable, fully typed astronomical state contract for Earth, Moon, and Sun.
    Implements the Mapping protocol for 100% transparent dictionary backward-compatibility,
    while providing canonical typed attributes and normalized aliases.
    """
    days: float
    sun_pos: np.ndarray
    earth_pos: np.ndarray
    earth_rot_angle: float
    moon_pos: np.ndarray
    moon_rot_angle: float
    earth_sun_dist_au: float
    moon_earth_dist_km: float
    season: str
    axial_tilt_rad: float = EARTH_AXIAL_TILT_RAD
    axial_tilt_deg: float = EARTH_AXIAL_TILT_DEG

    # Normalized aliases for attribute access
    @property
    def earth_rot_rad(self) -> float:
        """Alias for earth_rot_angle (radians)."""
        return self.earth_rot_angle

    @property
    def axial_tilt(self) -> float:
        """Alias for axial_tilt_rad (radians)."""
        return self.axial_tilt_rad

    # Mapping protocol implementation for transparent dictionary access
    def __getitem__(self, key: str) -> Any:
        if key in ("earth_rot_rad", "earth_rot_angle"):
            return self.earth_rot_angle
        elif key in ("axial_tilt", "axial_tilt_rad"):
            return self.axial_tilt_rad
        elif key == "axial_tilt_deg":
            return self.axial_tilt_deg
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if key in ("earth_rot_rad", "axial_tilt"):
            return True
        return hasattr(self, str(key)) and not str(key).startswith("_")

    def __len__(self) -> int:
        return 11

    def __iter__(self) -> Iterator[str]:
        yield from [
            "days", "sun_pos", "earth_pos", "earth_rot_angle",
            "moon_pos", "moon_rot_angle", "earth_sun_dist_au",
            "moon_earth_dist_km", "season", "axial_tilt_rad",
            "axial_tilt_deg"
        ]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def compute_astronomical_state(sim_time_sec: float) -> AstronomicalState:
    """
    Deterministically computes the spatial state of Earth and Moon
    given accumulated simulation time in seconds.
    Returns an immutable, typed AstronomicalState (with Mapping compatibility).
    """
    days = sim_time_sec / 86400.0

    # 1. Earth orbital revolution around Sun (counter-clockwise looking from North)
    earth_orbit_phase = (days / EARTH_ORBIT_PERIOD_DAYS) * 2.0 * math.pi
    earth_x = math.cos(earth_orbit_phase) * EARTH_ORBIT_RADIUS
    earth_y = 0.0
    earth_z = math.sin(earth_orbit_phase) * EARTH_ORBIT_RADIUS
    earth_pos = np.array([earth_x, earth_y, earth_z], dtype=np.float32)

    # 2. Earth diurnal rotation on its tilted axis
    earth_rot_angle = ((sim_time_sec / (EARTH_ROTATION_PERIOD_HOURS * 3600.0)) * 2.0 * math.pi) % (2.0 * math.pi)

    # 3. Moon orbit around Earth
    moon_orbit_phase = (days / MOON_ORBIT_PERIOD_DAYS) * 2.0 * math.pi
    inc_rad = math.radians(MOON_ORBIT_INCLINATION_DEG)
    rel_moon_x = math.cos(moon_orbit_phase) * MOON_ORBIT_RADIUS
    rel_moon_y = math.sin(moon_orbit_phase) * math.sin(inc_rad) * MOON_ORBIT_RADIUS
    rel_moon_z = math.sin(moon_orbit_phase) * math.cos(inc_rad) * MOON_ORBIT_RADIUS
    moon_pos = earth_pos + np.array([rel_moon_x, rel_moon_y, rel_moon_z], dtype=np.float32)

    # 4. Moon tidal locking (synchronous rotation with its orbit)
    moon_rot_angle = moon_orbit_phase % (2.0 * math.pi)

    # Season calculation (derived from Earth orbit phase)
    # Phase 0 = Vernal Equinox; 0.5 pi = Summer Solstice (Northern); pi = Autumnal Equinox; 1.5 pi = Winter Solstice
    season_names = ["Spring Equinox", "Summer Solstice", "Autumn Equinox", "Winter Solstice"]
    quadrant = int((earth_orbit_phase % (2.0 * math.pi)) / (math.pi / 2.0))
    season = season_names[quadrant]

    return AstronomicalState(
        days=days,
        sun_pos=SUN_POSITION,
        earth_pos=earth_pos,
        earth_rot_angle=earth_rot_angle,
        moon_pos=moon_pos,
        moon_rot_angle=moon_rot_angle,
        earth_sun_dist_au=1.0,
        moon_earth_dist_km=384400,
        season=season,
        axial_tilt_rad=EARTH_AXIAL_TILT_RAD,
        axial_tilt_deg=EARTH_AXIAL_TILT_DEG,
    )


@dataclass(frozen=True)
class CanonicalCelestialSystem:
    """
    Immutable Canonical Celestial System (Point 17).
    Deterministic orbital mechanics for Sun, Earth, and Moon.
    Celestial parameters cannot be altered by users or simulations.
    """
    sun_position: np.ndarray = field(default_factory=lambda: SUN_POSITION.copy())
    earth_orbit_radius: float = EARTH_ORBIT_RADIUS
    earth_orbit_period_days: float = EARTH_ORBIT_PERIOD_DAYS
    earth_rotation_period_hours: float = EARTH_ROTATION_PERIOD_HOURS
    earth_axial_tilt_deg: float = EARTH_AXIAL_TILT_DEG
    moon_orbit_radius: float = MOON_ORBIT_RADIUS
    moon_orbit_period_days: float = MOON_ORBIT_PERIOD_DAYS
    moon_orbit_inclination_deg: float = MOON_ORBIT_INCLINATION_DEG
    solar_constant_w_m2: float = 1361.0  # W/m^2 canonical solar irradiance

    @staticmethod
    def compute_state(sim_time_sec: float) -> AstronomicalState:
        return compute_astronomical_state(sim_time_sec)


# Global singleton instance
CELESTIAL_SYSTEM = CanonicalCelestialSystem()


