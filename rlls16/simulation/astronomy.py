import math
import numpy as np

# Immutable Canonical Astronomical Constants
SUN_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
EARTH_ORBIT_RADIUS = 90.0          # Render distance units
EARTH_ORBIT_PERIOD_DAYS = 365.25   # Days for 1 revolution around Sun
EARTH_ROTATION_PERIOD_HOURS = 24.0 # Hours for 1 full diurnal rotation
EARTH_AXIAL_TILT_DEG = 23.44       # Earth obliquity to ecliptic

MOON_ORBIT_RADIUS = 12.0           # Render distance units from Earth center
MOON_ORBIT_PERIOD_DAYS = 27.32166  # Sidereal orbit period
MOON_ORBIT_INCLINATION_DEG = 5.14  # Inclination to ecliptic


def compute_astronomical_state(sim_time_sec: float) -> dict:
    """
    Deterministically computes the spatial state of Earth and Moon
    given accumulated simulation time in seconds.
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

    return {
        "days": days,
        "sun_pos": SUN_POSITION,
        "earth_pos": earth_pos,
        "earth_rot_angle": earth_rot_angle,
        "moon_pos": moon_pos,
        "moon_rot_angle": moon_rot_angle,
        "earth_sun_dist_au": 1.0,
        "moon_earth_dist_km": 384400,
        "season": season,
    }


from dataclasses import dataclass, field


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
    def compute_state(sim_time_sec: float) -> dict:
        return compute_astronomical_state(sim_time_sec)


# Global singleton instance
CELESTIAL_SYSTEM = CanonicalCelestialSystem()


