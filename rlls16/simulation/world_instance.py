import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np

from .environment_engine import EnvironmentEngine, LocalEnvironmentObservation
from .human_population import HumanSettlementCohort
from .scheduler import MultiRateScheduler
from .rl_environment import RLLSEnvironmentInterface
from ..map.layers import MapLayers


ENVIRONMENT_TYPES = [
    "Forest",
    "Grassland",
    "Savannah",
    "Desert",
    "Tundra",
    "Mountain",
    "Coastal",
    "River Valley",
    "Tropical",
    "Temperate",
]


@dataclass
class IntelligentAgent:
    """Individual intelligent being with vitals, cognition, and RL state."""
    id: int
    name: str
    wx: float
    wy: float
    latitude_deg: float
    longitude_deg: float
    elevation: float
    biome_id: int
    age_years: float = 22.0
    health: float = 100.0
    energy: float = 88.0
    hydration: float = 92.0
    hunger: float = 12.0
    comfort: float = 85.0
    alive: bool = True
    perception_radius: float = 0.05
    rl_policy: str = "PPO_Survive_v1"
    memory: list = field(default_factory=list)
    recent_action: str = "FORAGE_FOOD"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntelligentAgent":
        return cls(**d)


@dataclass
class EnvironmentalState:
    solar_irradiance: float = 1.0       # 0.5x to 1.5x factor
    global_temp_offset: float = 0.0     # -10°C to +10°C offset
    cloud_cover_factor: float = 1.0     # 0.0x to 2.0x
    humidity_factor: float = 1.0        # 0.2x to 2.0x
    co2_ppm: float = 415.0              # Parts per million
    atmo_pressure_kpa: float = 101.3    # Standard atmosphere ~101.3 kPa
    soil_moisture_factor: float = 1.0   # Hydrological indicator


class WorldInstance:
    """
    Simulation World Instance.
    Coordinates:
      - MultiRateScheduler (1 tick = 1 hour, multi-rate decoupling)
      - EnvironmentEngine (surface energy balance, wind, orographic rain)
      - HumanSettlementCohort & Intelligent Agents
      - Immutable reference to canonical Earth dataset
    """

    def __init__(
        self,
        world_name: str = "World - 001",
        canonical_path: str = "worlds/canonical_world.npz",
        human_population: int = 1000,
        initial_environment: str = "Temperate",
        world_data: dict | None = None,
    ):
        self.world_name = world_name
        self.canonical_path = canonical_path

        # 1. Multi-Rate Scheduler
        self.scheduler = MultiRateScheduler(initial_time_sec=0.0)

        # 2. User Environmental Forcings (from UI accordions/sliders)
        self.environment = EnvironmentalState()

        # 3. Dynamic Environmental Physics Engine & 2D Map Layers
        self.world_data = world_data
        if world_data is not None:
            self.env_engine = EnvironmentEngine(world_data)
            self.layers = MapLayers(world_data)
        else:
            self.env_engine = None
            self.layers = None

        # Baseline snapshot for reset
        self.initial_population = human_population
        self.initial_env_type = initial_environment

        # 4. Human Settlement Deployment
        lat, lon, elev, b_id = self._locate_environment(world_data, initial_environment)
        self.human = HumanSettlementCohort(
            population=human_population,
            environment_type=initial_environment,
            latitude_deg=lat,
            longitude_deg=lon,
            elevation=elev,
            biome_id=b_id,
        )

        # 5. Spawn Initial Intelligent Beings
        self.agents: list[IntelligentAgent] = self._spawn_initial_agents(lat, lon, elev, b_id, count=6)

        # 6. Reinforcement Learning Environment Interface
        self.rl_interface = RLLSEnvironmentInterface(self, goal_type="Survive")

        # 7. Wire Scheduler Callbacks
        self.scheduler.subscribe_hourly(self._on_hourly_tick)
        self.scheduler.subscribe_daily(self._on_daily_tick)

    @property
    def sim_time_sec(self) -> float:
        return self.scheduler.sim_time_sec

    @sim_time_sec.setter
    def sim_time_sec(self, val: float):
        self.scheduler.sim_time_sec = val

    @property
    def time_speed(self) -> float:
        return self.scheduler.time_speed

    @time_speed.setter
    def time_speed(self, val: float):
        self.scheduler.time_speed = val

    @property
    def is_paused(self) -> bool:
        return self.scheduler.is_paused

    @is_paused.setter
    def is_paused(self, val: bool):
        self.scheduler.is_paused = val

    def _spawn_initial_agents(
        self,
        base_lat: float,
        base_lon: float,
        base_elev: float,
        base_biome: int,
        count: int = 6,
    ) -> list[IntelligentAgent]:
        """Spawn the first generation of individual intelligent beings near the settlement."""
        names = ["Adam", "Eve", "Enki", "Inanna", "Manu", "Noah", "Gilgamesh", "Ishtar"]
        agents = []
        for i in range(count):
            name = names[i % len(names)]
            # Slight geographic spread around settlement (~0.05 to 0.2 degrees)
            angle = (2.0 * math.pi / count) * i
            dist = 0.08 * (0.8 + 0.4 * (i % 3))
            lat = base_lat + dist * math.cos(angle)
            lon = base_lon + dist * math.sin(angle)

            wx = (lon + 180.0) / 360.0
            wy = (90.0 - lat) / 180.0 * 0.5

            agents.append(
                IntelligentAgent(
                    id=i + 1,
                    name=name,
                    wx=wx,
                    wy=wy,
                    latitude_deg=lat,
                    longitude_deg=lon,
                    elevation=base_elev,
                    biome_id=base_biome,
                    age_years=20.0 + (i * 3.5) % 15.0,
                    health=100.0,
                    energy=90.0 - (i * 5.0) % 20.0,
                    hydration=95.0 - (i * 7.0) % 20.0,
                    hunger=10.0 + (i * 4.0) % 25.0,
                    comfort=88.0,
                    rl_policy="PPO_Survive_v1",
                    memory=[f"Spawned in {self.initial_env_type} biome"],
                    recent_action="OBSERVE_ENVIRONMENT",
                )
            )
        return agents

    def get_all_agents(self) -> list:
        """Returns the settlement cohort followed by all individual intelligent beings."""
        return [self.human] + self.agents

    def _on_hourly_tick(self, sim_time: float):
        """Execute hourly environmental physics and individual agent / settlement metabolism."""
        if self.env_engine is not None:
            self.env_engine.step_hourly(
                sim_time_sec=sim_time,
                solar_irradiance_factor=self.environment.solar_irradiance,
                co2_ppm=self.environment.co2_ppm,
                global_temp_offset=self.environment.global_temp_offset,
                cloud_cover_multiplier=self.environment.cloud_cover_factor,
            )
            # Update settlement local vitals
            obs = self.env_engine.get_observation_at(self.human.latitude_deg, self.human.longitude_deg)
            self.human.step_hourly(obs, self.env_engine)

            # Update individual intelligent beings
            action_pool = ["FORAGE_FOOD", "GATHER_WATER", "EXPLORE", "REST", "SEEK_SHELTER"]
            for agent in self.agents:
                if not agent.alive:
                    continue
                ag_obs = self.env_engine.get_observation_at(agent.latitude_deg, agent.longitude_deg)

                # Metabolism
                agent.energy = max(0.0, min(100.0, agent.energy - 0.45))
                agent.hydration = max(0.0, min(100.0, agent.hydration - 0.60))
                agent.hunger = max(0.0, min(100.0, agent.hunger + 0.50))

                # Thermal exposure
                temp_diff = abs(ag_obs.temperature_c - 22.0)
                agent.comfort = max(0.0, min(100.0, 100.0 - temp_diff * 2.2))

                # Simple autonomous action & vitals recovery
                chosen_act = action_pool[(agent.id + int(sim_time / 3600.0)) % len(action_pool)]
                agent.recent_action = chosen_act

                if chosen_act == "FORAGE_FOOD" and ag_obs.vegetation_biomass > 0.2:
                    agent.hunger = max(0.0, agent.hunger - 15.0)
                    agent.energy = min(100.0, agent.energy + 8.0)
                elif chosen_act == "GATHER_WATER" and (ag_obs.soil_moisture > 0.3 or not ag_obs.is_land):
                    agent.hydration = min(100.0, agent.hydration + 20.0)
                elif chosen_act == "REST":
                    agent.energy = min(100.0, agent.energy + 10.0)
                elif chosen_act == "EXPLORE":
                    # Slight movement in local area
                    dlat = (math.sin(agent.id * 1.7 + sim_time * 0.001)) * 0.005
                    dlon = (math.cos(agent.id * 2.3 + sim_time * 0.001)) * 0.005
                    agent.latitude_deg += dlat
                    agent.longitude_deg += dlon
                    agent.wx = (agent.longitude_deg + 180.0) / 360.0
                    agent.wy = (90.0 - agent.latitude_deg) / 180.0 * 0.5

                # Health decay if starved or dehydrated
                if agent.hunger > 90.0 or agent.hydration < 10.0:
                    agent.health = max(0.0, agent.health - 2.0)
                elif agent.energy > 50.0 and agent.hydration > 50.0 and agent.health < 100.0:
                    agent.health = min(100.0, agent.health + 0.5)

                if agent.health <= 0.0:
                    agent.alive = False

    def _on_daily_tick(self):
        """Execute daily environmental hydrology and demographic dynamics."""
        if self.env_engine is not None:
            self.env_engine.step_daily()
            obs = self.env_engine.get_observation_at(self.human.latitude_deg, self.human.longitude_deg)
            self.human.step_daily(obs)

        # Increment age of agents
        for agent in self.agents:
            if agent.alive:
                agent.age_years += 1.0 / 365.0

    def get_current_observation(self) -> LocalEnvironmentObservation:
        """Get environmental observation at human settlement coordinates."""
        if self.env_engine is not None:
            return self.env_engine.get_observation_at(self.human.latitude_deg, self.human.longitude_deg)
        return LocalEnvironmentObservation(
            latitude_deg=self.human.latitude_deg,
            longitude_deg=self.human.longitude_deg,
            elevation=self.human.elevation,
            is_land=True,
            temperature_c=22.0,
            pressure_kpa=101.3,
            wind_u=2.0,
            wind_v=1.0,
            wind_speed=2.2,
            wind_dir_deg=45.0,
            humidity=0.010,
            cloud_cover=0.3,
            precipitation_rate_mm_h=0.0,
            is_snow=False,
            soil_moisture=0.5,
            vegetation_biomass=0.6,
            solar_flux_w_m2=450.0,
        )

    def _locate_environment(self, world_data: dict | None, env_type: str) -> tuple[float, float, float, int]:
        """Find a canonical coordinate matching the desired environment class."""
        if world_data is None:
            return 28.5, 42.0, 0.55, 3

        elevation = world_data["elevation"]
        land_mask = world_data["land_mask"]
        temp = world_data["temperature"]
        precip = world_data["precipitation"]
        biome = world_data["biome"]
        lat = world_data["latitude"]
        lon = world_data["longitude"]

        # Selection criteria masks
        if env_type == "Forest":
            mask = land_mask & (biome == 4)
        elif env_type == "Grassland":
            mask = land_mask & (biome == 3)
        elif env_type == "Savannah":
            mask = land_mask & (temp > 0.6) & (precip > 0.25) & (precip < 0.50)
        elif env_type == "Desert":
            mask = land_mask & (biome == 2)
        elif env_type == "Tundra":
            mask = land_mask & (biome == 6)
        elif env_type == "Mountain":
            mask = land_mask & (elevation > 0.72)
        elif env_type == "Coastal":
            mask = land_mask & (elevation >= 0.50) & (elevation <= 0.53)
        elif env_type == "River Valley":
            mask = land_mask & (elevation >= 0.51) & (elevation <= 0.58) & (precip > 0.55)
        elif env_type == "Tropical":
            mask = land_mask & (biome == 5)
        else:  # Temperate
            mask = land_mask & (temp >= 0.40) & (temp <= 0.65) & (precip >= 0.35)

        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            candidates = np.argwhere(land_mask)
            if len(candidates) == 0:
                return 0.0, 0.0, 0.5, 0

        # Deterministic selection based on world_name
        idx = (abs(hash(self.world_name)) + 42) % len(candidates)
        r, c = candidates[idx]

        lat_deg = float(math.degrees(lat[r]))
        lon_deg = float(math.degrees(lon[c]))
        elev_val = float(elevation[r, c])
        biome_val = int(biome[r, c])
        return lat_deg, lon_deg, elev_val, biome_val

    def update(self, dt: float):
        """Advance multi-rate scheduler by real delta time."""
        self.scheduler.advance(dt)

    def get_formatted_time(self) -> tuple[str, str, str]:
        return self.scheduler.get_formatted_time()

    def reset(self, world_data: dict | None):
        """Restore world to its initial state without modifying canonical Earth."""
        self.scheduler.reset(0.0)
        self.environment = EnvironmentalState()
        if world_data is not None:
            self.env_engine = EnvironmentEngine(world_data)
            self.layers = MapLayers(world_data)

        lat, lon, elev, b_id = self._locate_environment(world_data, self.initial_env_type)
        self.human = HumanSettlementCohort(
            population=self.initial_population,
            environment_type=self.initial_env_type,
            latitude_deg=lat,
            longitude_deg=lon,
            elevation=elev,
            biome_id=b_id,
        )
        self.agents = self._spawn_initial_agents(lat, lon, elev, b_id, count=6)
        self.rl_interface = RLLSEnvironmentInterface(self, goal_type="Survive")

    def save(self, base_dir: str = "simulations"):
        sim_dir = Path(base_dir) / self.world_name.replace(" ", "_")
        sim_dir.mkdir(parents=True, exist_ok=True)
        save_file = sim_dir / "world_state.json"

        data = {
            "world_name": self.world_name,
            "canonical_path": self.canonical_path,
            "sim_time_sec": self.sim_time_sec,
            "time_speed": self.time_speed,
            "is_paused": self.is_paused,
            "environment": asdict(self.environment),
            "human": self.human.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "initial_population": self.initial_population,
            "initial_env_type": self.initial_env_type,
        }
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(save_file)

    @classmethod
    def load(cls, file_path: str, world_data: dict | None = None) -> "WorldInstance":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        inst = cls(
            world_name=data["world_name"],
            canonical_path=data.get("canonical_path", "worlds/canonical_world.npz"),
            human_population=data.get("initial_population", 1000),
            initial_environment=data.get("initial_env_type", "Temperate"),
            world_data=world_data,
        )
        inst.sim_time_sec = data.get("sim_time_sec", 0.0)
        inst.time_speed = data.get("time_speed", 1.0)
        inst.is_paused = data.get("is_paused", False)

        env_dict = data.get("environment", {})
        inst.environment = EnvironmentalState(**env_dict)

        if "human" in data:
            inst.human = HumanSettlementCohort.from_dict(data["human"])
        if "agents" in data:
            inst.agents = [IntelligentAgent.from_dict(a) for a in data["agents"]]
        return inst
