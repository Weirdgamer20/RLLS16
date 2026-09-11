import math
import numpy as np
from dataclasses import dataclass


@dataclass
class LocalEnvironmentObservation:
    """Instantaneous environmental state at a specific coordinate."""
    latitude_deg: float
    longitude_deg: float
    elevation: float
    is_land: bool
    temperature_c: float
    pressure_kpa: float
    wind_u: float
    wind_v: float
    wind_speed: float
    wind_dir_deg: float
    humidity: float
    cloud_cover: float
    precipitation_rate_mm_h: float
    is_snow: bool
    soil_moisture: float
    vegetation_biomass: float
    solar_flux_w_m2: float


class EnvironmentEngine:
    """
    RLLS 16 Reduced-Order Coupled Earth-System Environmental Simulator.
    Implements:
      - Surface Energy Balance (insolation, outgoing longwave, albedo, thermal inertia)
      - Atmospheric Pressure & Reduced Horizontal Momentum (Coriolis, pressure gradient, friction)
      - Moisture Advection & Orographic Condensation (Clausius-Clapeyron, cloud formation, rain/snow)
      - Soil Moisture & Emergent Vegetation Biomass Dynamics
    """

    def __init__(self, world_data: dict):
        self.elevation = world_data["elevation"].astype(np.float32)       # [H, W] normalized 0..1
        self.land_mask = world_data["land_mask"].astype(bool)             # [H, W]
        self.base_temp = world_data["temperature"].astype(np.float32)     # [H, W] normalized 0..1
        self.base_precip = world_data["precipitation"].astype(np.float32) # [H, W] normalized 0..1
        self.biome = world_data["biome"].astype(np.int32)
        self.lat_rad = world_data["latitude"].astype(np.float32)          # [H] -pi/2..pi/2
        self.lon_rad = world_data["longitude"].astype(np.float32)         # [W] -pi..pi

        self.H, self.W = self.elevation.shape
        self.dlat = float(self.lat_rad[1] - self.lat_rad[0])
        self.dlon = float(self.lon_rad[1] - self.lon_rad[0])

        # Precompute spatial geometric arrays
        # 2D latitude grid in radians
        self.lat2d = np.repeat(self.lat_rad[:, np.newaxis], self.W, axis=1)
        self.lon2d = np.repeat(self.lon_rad[np.newaxis, :], self.H, axis=0)

        # Coriolis parameter: f = 2 * Omega * sin(phi), Earth Omega ~ 7.2921e-5 rad/s
        self.coriolis = (2.0 * 7.2921e-5 * np.sin(self.lat2d)).astype(np.float32)

        # Heat capacity C: Ocean has ~5x heat capacity of land
        self.heat_capacity = np.where(self.land_mask, 1.2e6, 6.0e6).astype(np.float32) # J/(m^2 K)

        # Base physical elevation in meters (~0 to 8000m)
        self.elev_meters = (np.maximum(0.0, self.elevation - 0.5) * 16000.0).astype(np.float32)

        # Calculate terrain gradients (slopes) for orographic lift
        self.grad_elev_y, self.grad_elev_x = np.gradient(self.elev_meters)
        self.grad_elev_x /= 111000.0 * np.maximum(0.1, np.cos(self.lat2d))
        self.grad_elev_y /= 111000.0

        # --- Dynamic State Buffers (Double-Buffered: State A / State B) ---
        # Temperature in Celsius: convert normalized base (0..1) to ~ -35°C to +35°C
        t_init = (self.base_temp * 70.0 - 35.0).astype(np.float32)
        self.temp = t_init.copy()
        self.temp_next = t_init.copy()

        # Atmospheric pressure in kPa (standard sea-level ~101.3 kPa, hydrostatic drop with elevation)
        p_init = (101.325 * np.exp(-self.elev_meters / 8400.0)).astype(np.float32)
        self.pressure = p_init.copy()
        self.pressure_next = p_init.copy()

        # Horizontal wind field (u: zonal/eastward m/s, v: meridional/northward m/s)
        self.wind_u = np.zeros((self.H, self.W), dtype=np.float32)
        self.wind_v = np.zeros((self.H, self.W), dtype=np.float32)
        self.wind_u_next = np.zeros_like(self.wind_u)
        self.wind_v_next = np.zeros_like(self.wind_v)

        # Specific humidity q (kg/kg), normalized roughly 0.001 to 0.025
        q_init = (self.base_precip * 0.015 + 0.002).astype(np.float32)
        self.humidity = q_init.copy()
        self.humidity_next = q_init.copy()

        # Cloud cover fraction [0.0, 1.0]
        self.cloud_cover = (self.base_precip * 0.6).astype(np.float32)

        # Precipitation rate in mm/hour
        self.precip_rate = np.zeros((self.H, self.W), dtype=np.float32)
        self.is_snow = np.zeros((self.H, self.W), dtype=bool)

        # Soil moisture fraction [0.0, 1.0]
        self.soil_moisture = np.where(self.land_mask, self.base_precip, 1.0).astype(np.float32)

        # Emergent vegetation biomass index [0.0, 1.0]
        self.vegetation_biomass = np.where(self.land_mask, self.base_precip * 0.8 + 0.1, 0.0).astype(np.float32)

        # Radiation flux tracker for diagnostics (W/m^2)
        self.solar_flux = np.zeros((self.H, self.W), dtype=np.float32)

        # Snow cover accumulation (depth in cm)
        self.snow_depth_cm = np.zeros((self.H, self.W), dtype=np.float32)

    def step_hourly(
        self,
        sim_time_sec: float,
        solar_irradiance_factor: float = 1.0,
        co2_ppm: float = 415.0,
        global_temp_offset: float = 0.0,
        cloud_cover_multiplier: float = 1.0,
    ):
        """
        Advance the coupled environmental physics by 1 simulation hour (dt = 3600s).
        Reads self.* (State t) and writes into self.*_next (State t+1).
        """
        dt = 3600.0  # 1 hour in seconds

        # 1. Astronomical Subsolar Geometry
        # Earth rotates once every 86400s (24h)
        # Earth axial tilt = 23.44° = 0.409 rad
        # Annual orbit ~ 365.25 days = 31557600s
        year_frac = (sim_time_sec % 31557600.0) / 31557600.0
        orbital_angle = 2.0 * math.pi * year_frac
        solar_declination = 0.409 * math.sin(orbital_angle - 1.39) # Solstice offset
        earth_rot_angle = (2.0 * math.pi * (sim_time_sec % 86400.0) / 86400.0) - math.pi

        # Solar zenith angle cos(theta) for every cell
        hour_angle = self.lon2d - earth_rot_angle
        hour_angle = (hour_angle + math.pi) % (2.0 * math.pi) - math.pi

        cos_zenith = np.sin(self.lat2d) * math.sin(solar_declination) + \
                     np.cos(self.lat2d) * math.cos(solar_declination) * np.cos(hour_angle)
        sunlit = np.maximum(0.0, cos_zenith)

        # 2. Surface Albedo
        # Base albedo: Ocean ~ 0.08, Land ~ 0.22, Snow/Ice ~ 0.75
        base_albedo = np.where(self.land_mask, 0.22, 0.08)
        snow_albedo = np.clip(self.snow_depth_cm / 10.0, 0.0, 1.0) * 0.53
        albedo = np.clip(base_albedo + snow_albedo + self.cloud_cover * 0.20, 0.05, 0.85)

        # Solar constant S0 ~ 1361 W/m^2 modulated by user solar irradiance
        s0 = 1361.0 * solar_irradiance_factor
        q_solar = s0 * sunlit * (1.0 - albedo)
        self.solar_flux[:] = q_solar

        # 3. Radiative Longwave Cooling & Greenhouse Effect
        t_kelvin = self.temp + 273.15
        co2_forcing = 5.35 * math.log(max(100.0, co2_ppm) / 280.0) # W/m^2 forcing relative to pre-industrial
        emissivity = np.clip(0.68 - 0.03 * np.log(max(100.0, co2_ppm) / 280.0) - 0.08 * self.humidity * 100.0, 0.40, 0.95)
        q_longwave = emissivity * 5.67e-8 * (t_kelvin ** 4) - co2_forcing

        # 4. Latent & Sensible Heat Fluxes
        q_latent = np.where(self.land_mask, 35.0 * self.soil_moisture, 80.0) * np.maximum(0.0, (t_kelvin - 260.0) / 40.0)
        q_sensible = np.where(self.land_mask, 25.0, 10.0) * (self.temp - 15.0)

        # Temperature increment
        net_flux = q_solar - q_longwave - q_latent - q_sensible
        dtemp = (net_flux / self.heat_capacity) * dt

        target_temp = self.temp + dtemp + (global_temp_offset * (dt / 86400.0))
        self.temp_next[:] = np.clip(target_temp, -75.0, 58.0)

        # 5. Pressure Solver
        p_hydro = 101.325 * np.exp(-self.elev_meters / 8400.0)
        p_thermal = -0.045 * (self.temp_next - 15.0)
        self.pressure_next[:] = np.clip(p_hydro + p_thermal, 35.0, 108.0)

        # 6. Wind Momentum (Zonal U and Meridional V)
        dp_dy, dp_dx = np.gradient(self.pressure_next * 1000.0) # Pa
        rho = 1.225 # Air density kg/m^3
        dp_dx /= 111000.0 * np.maximum(0.1, np.cos(self.lat2d))
        dp_dy /= 111000.0

        # Pressure gradient acceleration: -1/rho * grad(p)
        acc_u = - (1.0 / rho) * dp_dx
        acc_v = - (1.0 / rho) * dp_dy

        # Coriolis acceleration
        acc_u += self.coriolis * self.wind_v
        acc_v -= self.coriolis * self.wind_u

        # Terrain friction and atmospheric drag
        drag = np.where(self.land_mask, 0.00015, 0.00004)
        u_new = self.wind_u + (acc_u - drag * self.wind_u) * dt
        v_new = self.wind_v + (acc_v - drag * self.wind_v) * dt

        planetary_u = -8.0 * np.cos(self.lat2d * 3.0)
        self.wind_u_next[:] = np.clip(0.85 * u_new + 0.15 * planetary_u, -45.0, 45.0)
        self.wind_v_next[:] = np.clip(0.90 * v_new, -35.0, 35.0)

        # 7. Moisture, Clouds, and Precipitation
        t_c = self.temp_next
        e_sat_kpa = 0.61078 * np.exp((17.27 * t_c) / np.maximum(1.0, t_c + 237.3))
        q_sat = np.clip(0.622 * (e_sat_kpa / np.maximum(20.0, self.pressure_next)), 0.0005, 0.040)

        wind_spd = np.sqrt(self.wind_u_next**2 + self.wind_v_next**2)
        evap_potential = np.where(self.land_mask, 0.0000003 * self.soil_moisture, 0.0000012)
        evaporation = evap_potential * (1.0 + 0.15 * wind_spd) * np.maximum(0.0, q_sat - self.humidity)

        dq_x = np.roll(self.humidity, -1, axis=1) - np.roll(self.humidity, 1, axis=1)
        dq_y = np.roll(self.humidity, -1, axis=0) - np.roll(self.humidity, 1, axis=0)
        advection = - (self.wind_u_next * dq_x / (2.0 * self.dlon * 111000.0) + \
                       self.wind_v_next * dq_y / (2.0 * self.dlat * 111000.0))

        orographic_lift = np.maximum(0.0, self.wind_u_next * self.grad_elev_x + self.wind_v_next * self.grad_elev_y)
        orographic_cooling_effect = orographic_lift * 0.00001

        q_tentative = np.maximum(0.0001, self.humidity + (advection + evaporation) * dt)

        relative_humidity = q_tentative / q_sat
        excess_q = np.maximum(0.0, q_tentative - q_sat * 0.95) + orographic_cooling_effect

        raw_clouds = np.clip((relative_humidity - 0.55) / 0.40, 0.0, 1.0)
        self.cloud_cover[:] = np.clip(raw_clouds * cloud_cover_multiplier, 0.0, 1.0)

        precip_mm_h = excess_q * 1000.0 * 3.6
        self.precip_rate[:] = np.clip(precip_mm_h, 0.0, 120.0)
        self.is_snow[:] = (self.temp_next <= 0.0) & (self.precip_rate > 0.05)

        self.snow_depth_cm = np.where(
            self.is_snow,
            self.snow_depth_cm + self.precip_rate * 0.1,
            np.maximum(0.0, self.snow_depth_cm - np.maximum(0.0, self.temp_next) * 0.08)
        )

        self.humidity_next[:] = np.maximum(0.0002, q_tentative - excess_q * 0.75)

        # 8. Swap Buffers
        self.temp[:] = self.temp_next
        self.pressure[:] = self.pressure_next
        self.wind_u[:] = self.wind_u_next
        self.wind_v[:] = self.wind_v_next
        self.humidity[:] = self.humidity_next

    def step_daily(self):
        """
        Daily environmental cycle (~once every 24 simulated hours).
        Updates hydrology, soil moisture, and emergent vegetation biomass.
        """
        recharge = self.precip_rate * 0.02
        discharge = 0.05 + np.maximum(0.0, self.temp / 35.0) * 0.08
        self.soil_moisture = np.where(
            self.land_mask,
            np.clip(self.soil_moisture + recharge - discharge, 0.02, 1.0),
            1.0
        )

        thermal_suitability = np.clip(1.0 - np.abs(self.temp - 22.0) / 22.0, 0.0, 1.0)
        moisture_suitability = np.clip((self.soil_moisture - 0.15) / 0.70, 0.0, 1.0)
        growth_potential = thermal_suitability * moisture_suitability

        growth_rate = 0.04 * growth_potential
        decay_rate = np.where(self.temp < -2.0, 0.08, 0.02)
        self.vegetation_biomass = np.where(
            self.land_mask,
            np.clip(self.vegetation_biomass + growth_rate - decay_rate, 0.01, 1.0),
            0.0
        )

    def get_observation_at(self, lat_deg: float, lon_deg: float) -> LocalEnvironmentObservation:
        """Query instantaneous physical and weather telemetry for any Earth coordinate."""
        lat_r = math.radians(lat_deg)
        lon_r = math.radians(lon_deg)

        r = int(np.clip(int((lat_r - self.lat_rad[0]) / self.dlat), 0, self.H - 1))
        c = int(np.clip(int((lon_r - self.lon_rad[0]) / self.dlon), 0, self.W - 1))

        u = float(self.wind_u[r, c])
        v = float(self.wind_v[r, c])
        spd = float(math.hypot(u, v))
        dir_deg = float(math.degrees(math.atan2(v, u))) % 360.0

        return LocalEnvironmentObservation(
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            elevation=float(self.elevation[r, c]),
            is_land=bool(self.land_mask[r, c]),
            temperature_c=float(self.temp[r, c]),
            pressure_kpa=float(self.pressure[r, c]),
            wind_u=u,
            wind_v=v,
            wind_speed=spd,
            wind_dir_deg=dir_deg,
            humidity=float(self.humidity[r, c]),
            cloud_cover=float(self.cloud_cover[r, c]),
            precipitation_rate_mm_h=float(self.precip_rate[r, c]),
            is_snow=bool(self.is_snow[r, c]),
            soil_moisture=float(self.soil_moisture[r, c]),
            vegetation_biomass=float(self.vegetation_biomass[r, c]),
            solar_flux_w_m2=float(self.solar_flux[r, c]),
        )
