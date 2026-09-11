import math
import random
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment_engine import EnvironmentEngine, LocalEnvironmentObservation


@dataclass
class CognitiveState:
    """Cognitive capacities and internal mental representation."""
    learning_rate: float = 0.05
    memory_capacity: int = 50
    perception_radius_km: float = 15.0
    reasoning_capacity: float = 0.20  # Baseline primitive cognition


@dataclass
class KnowledgeBase:
    """Acquired and culturally transmitted skills and discoveries."""
    edible_flora_identified: float = 0.10   # Fraction of local flora known
    freshwater_sourcing_skill: float = 0.15 # Efficiency in locating / digging water
    shelter_crafting_skill: float = 0.05    # Protection against thermal extremes
    fire_utilization_skill: float = 0.0     # Discovered through experiment or transmission
    tool_efficacy: float = 0.05             # Foraging and hunting multiplier
    accumulated_cultural_lore: float = 0.0  # Shared oral traditions across generations


@dataclass
class BiologicalState:
    """Physiological vitals of the population cohort."""
    health: float = 100.0             # 0 to 100%
    energy_hunger: float = 85.0       # 0 (starving) to 100% (satiated)
    hydration_thirst: float = 90.0    # 0 (dehydrated) to 100% (hydrated)
    thermal_comfort: float = 80.0     # 0 (extreme stress) to 100% (comfortable)
    average_age_years: float = 24.5
    generation: int = 1


class HumanSettlementCohort:
    """
    RLLS 16 Human Population & Cognitive Evolution Engine.
    Represents an evolving human settlement initialized at low capability.
    Coordinates:
      - Biological homeostasis (hydration, hunger, thermal exposure)
      - Environmental extraction & bidirectional ecological modification
      - Experience gathering, learning, and cultural transmission
      - Generational demographic cycles and knowledge retention/decay
    """

    def __init__(
        self,
        population: int = 1000,
        environment_type: str = "Savannah",
        latitude_deg: float = 15.0,
        longitude_deg: float = 35.0,
        elevation: float = 0.52,
        biome_id: int = 3,
    ):
        self.population = max(10, population)
        self.initial_population = self.population
        self.environment_type = environment_type
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.elevation = elevation
        self.biome_id = biome_id

        # Cognitive, Biological, and Knowledge states
        self.cognition = CognitiveState()
        self.knowledge = KnowledgeBase()
        self.vitals = BiologicalState()

        # Spatial Memory: list of remembered coordinates of discovered resources
        # e.g., [{"type": "water", "lat": 15.1, "lon": 35.2, "quality": 0.9}]
        self.remembered_sites: list[dict] = []

        # Tracking metrics for telemetry and analytics
        self.births_total = 0
        self.deaths_total = 0
        self.knowledge_discoveries = 0
        self.shelter_level = 0.05

    def step_hourly(self, env_obs: "LocalEnvironmentObservation", env_engine: "EnvironmentEngine"):
        """Hourly physiological metabolism and immediate environmental interaction."""
        if self.population <= 0:
            return

        # 1. Thermal stress evaluation
        temp = env_obs.temperature_c
        # Comfortable range ~ 18°C to 28°C
        thermal_strain = max(0.0, 18.0 - temp) if temp < 18.0 else max(0.0, temp - 28.0)
        # Shelter and knowledge mitigate thermal exposure
        protection = min(0.85, self.shelter_level + self.knowledge.shelter_crafting_skill * 0.4)
        effective_strain = thermal_strain * (1.0 - protection)
        self.vitals.thermal_comfort = max(0.0, 100.0 - effective_strain * 3.0)

        # 2. Water / Hydration Metabolism
        # Higher temperature or exertion accelerates water loss
        water_loss_rate = 0.45 + (max(0.0, temp - 22.0) / 10.0) * 0.35
        self.vitals.hydration_thirst = max(0.0, self.vitals.hydration_thirst - water_loss_rate)

        # 3. Energy / Hunger Metabolism
        # Base metabolic burn ~ 0.3% per hour, increased in freezing temperatures
        caloric_burn = 0.30 + (max(0.0, 5.0 - temp) / 15.0) * 0.20
        self.vitals.energy_hunger = max(0.0, self.vitals.energy_hunger - caloric_burn)

        # 4. Immediate Foraging / Drinking if resources are locally available
        # Drinking: from local rainfall, rivers, or soil moisture
        if env_obs.precipitation_rate_mm_h > 0.1 or env_obs.soil_moisture > 0.30 or not env_obs.is_land:
            water_intake = (1.5 + self.knowledge.freshwater_sourcing_skill * 2.0)
            self.vitals.hydration_thirst = min(100.0, self.vitals.hydration_thirst + water_intake)

        # Foraging: harvest local vegetation biomass
        if env_obs.vegetation_biomass > 0.05:
            # Efficiency boosted by knowledge of edible flora and tools
            foraging_eff = 0.8 + self.knowledge.edible_flora_identified * 1.5 + self.knowledge.tool_efficacy * 0.5
            food_obtained = min(2.5, env_obs.vegetation_biomass * 10.0 * foraging_eff)
            self.vitals.energy_hunger = min(100.0, self.vitals.energy_hunger + food_obtained)

            # Bidirectional ecological feedback: humans deplete a tiny fraction of local biomass
            consumed_biomass = (self.population * 0.000002) * (foraging_eff / 2.0)
            # Find grid indices and deduct
            r = int(np_clip_grid(env_obs.latitude_deg, env_engine.lat_rad, env_engine.dlat, env_engine.H))
            c = int(np_clip_grid(env_obs.longitude_deg, env_engine.lon_rad, env_engine.dlon, env_engine.W))
            env_engine.vegetation_biomass[r, c] = max(0.01, env_engine.vegetation_biomass[r, c] - consumed_biomass)

        # 5. Health impact from severe deprivation
        health_delta = 0.0
        if self.vitals.hydration_thirst < 15.0:
            health_delta -= (15.0 - self.vitals.hydration_thirst) * 0.08  # Dehydration is lethal
        if self.vitals.energy_hunger < 10.0:
            health_delta -= (10.0 - self.vitals.energy_hunger) * 0.03    # Starvation damage
        if self.vitals.thermal_comfort < 20.0:
            health_delta -= (20.0 - self.vitals.thermal_comfort) * 0.04  # Hypothermia / heatstroke

        # Natural recovery when satiated and well hydrated
        if self.vitals.hydration_thirst > 70.0 and self.vitals.energy_hunger > 70.0 and self.vitals.thermal_comfort > 60.0:
            health_delta += 0.25

        self.vitals.health = max(0.0, min(100.0, self.vitals.health + health_delta))

        # 6. Empirical Discovery & Observation
        # Humans observing novel seasonal conditions discover edible plants or tool uses
        if random.random() < self.cognition.learning_rate * 0.02:
            self._attempt_discovery(env_obs)

    def step_daily(self, env_obs: "LocalEnvironmentObservation"):
        """Daily demographic progression, social knowledge transmission, and mortality."""
        if self.population <= 0:
            return

        # 1. Demographic Mortality
        daily_mortality_rate = 0.0001 # Base natural mortality ~ 3.6% per year
        if self.vitals.health < 40.0:
            daily_mortality_rate += (40.0 - self.vitals.health) * 0.0008
        if self.vitals.hydration_thirst < 5.0:
            daily_mortality_rate += 0.08  # Extreme acute dehydration deaths
        if self.vitals.energy_hunger < 5.0:
            daily_mortality_rate += 0.04  # Acute starvation deaths

        deaths = min(self.population, int(math.ceil(self.population * daily_mortality_rate)))
        self.population -= deaths
        self.deaths_total += deaths

        if self.population <= 0:
            self.vitals.health = 0.0
            return

        # 2. Demographic Reproduction (Births)
        # Births occur when health, hydration, and energy are abundant
        daily_birth_rate = 0.00015 # Base natural birth rate ~ 5.5% per year
        if self.vitals.health > 80.0 and self.vitals.energy_hunger > 75.0 and self.vitals.hydration_thirst > 75.0:
            daily_birth_rate *= 1.8

        births = int(self.population * daily_birth_rate)
        self.population += births
        self.births_total += births

        # 3. Shelter construction from local materials
        if self.vitals.energy_hunger > 50.0:
            shelter_gain = 0.002 * (self.knowledge.shelter_crafting_skill + 0.1)
            self.shelter_level = min(0.95, self.shelter_level + shelter_gain)

        # 4. Social Knowledge Transmission & Cultural Retention
        # Elders communicate survival lore to youth
        transmission_rate = 0.005 * (self.population / (self.population + 200.0))
        self.knowledge.accumulated_cultural_lore += transmission_rate
        self.knowledge.edible_flora_identified = min(1.0, self.knowledge.edible_flora_identified + transmission_rate * 0.05)
        self.knowledge.freshwater_sourcing_skill = min(1.0, self.knowledge.freshwater_sourcing_skill + transmission_rate * 0.04)

        # 5. Generational Knowledge Decay check:
        # If massive catastrophic mortality occurred (>20% of population died today), knowledge is partially lost!
        if deaths > 0.20 * (self.population + deaths):
            decay = 0.25 # Knowledge dies with the fallen elders
            self.knowledge.edible_flora_identified = max(0.05, self.knowledge.edible_flora_identified * (1.0 - decay))
            self.knowledge.shelter_crafting_skill = max(0.02, self.knowledge.shelter_crafting_skill * (1.0 - decay))
            self.knowledge.accumulated_cultural_lore = max(0.0, self.knowledge.accumulated_cultural_lore * (1.0 - decay))

    def _attempt_discovery(self, env_obs: "LocalEnvironmentObservation"):
        """Probabilistic empirical discovery when exploring local environmental conditions."""
        self.knowledge_discoveries += 1
        roll = random.random()
        if roll < 0.35:
            self.knowledge.edible_flora_identified = min(1.0, self.knowledge.edible_flora_identified + 0.04)
        elif roll < 0.65:
            self.knowledge.freshwater_sourcing_skill = min(1.0, self.knowledge.freshwater_sourcing_skill + 0.04)
        elif roll < 0.85:
            self.knowledge.shelter_crafting_skill = min(1.0, self.knowledge.shelter_crafting_skill + 0.03)
        else:
            self.knowledge.tool_efficacy = min(1.0, self.knowledge.tool_efficacy + 0.05)

    def to_dict(self) -> dict:
        return {
            "population": self.population,
            "initial_population": self.initial_population,
            "environment_type": self.environment_type,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "elevation": self.elevation,
            "biome_id": self.biome_id,
            "cognition": asdict(self.cognition),
            "knowledge": asdict(self.knowledge),
            "vitals": asdict(self.vitals),
            "shelter_level": self.shelter_level,
            "births_total": self.births_total,
            "deaths_total": self.deaths_total,
            "knowledge_discoveries": self.knowledge_discoveries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HumanSettlementCohort":
        inst = cls(
            population=data.get("population", 1000),
            environment_type=data.get("environment_type", "Savannah"),
            latitude_deg=data.get("latitude_deg", 15.0),
            longitude_deg=data.get("longitude_deg", 35.0),
            elevation=data.get("elevation", 0.52),
            biome_id=data.get("biome_id", 3),
        )
        inst.initial_population = data.get("initial_population", inst.population)
        if "cognition" in data:
            inst.cognition = CognitiveState(**data["cognition"])
        if "knowledge" in data:
            inst.knowledge = KnowledgeBase(**data["knowledge"])
        if "vitals" in data:
            inst.vitals = BiologicalState(**data["vitals"])
        inst.shelter_level = data.get("shelter_level", 0.05)
        inst.births_total = data.get("births_total", 0)
        inst.deaths_total = data.get("deaths_total", 0)
        inst.knowledge_discoveries = data.get("knowledge_discoveries", 0)
        return inst


def np_clip_grid(deg: float, rad_arr, d_rad: float, max_dim: int) -> int:
    rad = math.radians(deg)
    return int(max(0, min(max_dim - 1, int((rad - rad_arr[0]) / d_rad))))
