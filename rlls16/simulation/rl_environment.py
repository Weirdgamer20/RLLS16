import math
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .world_instance import WorldInstance


ACTIONS = [
    "STAY",
    "MOVE_N",
    "MOVE_NE",
    "MOVE_E",
    "MOVE_SE",
    "MOVE_S",
    "MOVE_SW",
    "MOVE_W",
    "MOVE_NW",
    "FORAGE_FOOD",
    "GATHER_WATER",
    "SEEK_OR_BUILD_SHELTER",
    "EXPLORE",
    "TEACH_COMMUNITY",
]

ACTION_NAME_TO_ID = {name: idx for idx, name in enumerate(ACTIONS)}


class RLLSEnvironmentInterface:
    """
    RLLS 16 Reinforcement Learning Interface.
    Standardized Gym-style environment wrapper around the Earth simulation.
    Exposes:
      - Observation vector: normalized physiological and environmental state
      - Hierarchical discrete action space with dynamic action masking
      - Multi-objective reward formulations selectable by the user
    """

    def __init__(self, world_instance: "WorldInstance", goal_type: str = "Survive"):
        self.world = world_instance
        self.goal_type = goal_type # "Survive", "Growth", "Exploration", "Culture"

        self.action_space_size = len(ACTIONS)
        self.observation_dim = 12

        # Step count tracking for episodes
        self.episode_steps = 0
        self.max_episode_steps = 1000

    def get_observation(self) -> np.ndarray:
        """
        Produce a normalized observation vector in [0.0, 1.0] representing
        local environmental conditions and internal physiological vitals.
        """
        obs_env = self.world.get_current_observation()
        human = self.world.human

        # 1. Normalized Temperature: map -40°C..+50°C to 0..1
        temp_norm = np.clip((obs_env.temperature_c + 40.0) / 90.0, 0.0, 1.0)
        # 2. Humidity: 0..0.025 kg/kg mapped to 0..1
        humidity_norm = np.clip(obs_env.humidity / 0.025, 0.0, 1.0)
        # 3. Precipitation: 0..30 mm/h mapped to 0..1
        precip_norm = np.clip(obs_env.precipitation_rate_mm_h / 30.0, 0.0, 1.0)
        # 4. Vegetation Biomass: 0..1
        biomass_norm = np.clip(obs_env.vegetation_biomass, 0.0, 1.0)
        # 5. Soil Moisture: 0..1
        water_avail = np.clip(obs_env.soil_moisture, 0.0, 1.0)
        # 6. Wind Speed: 0..30 m/s mapped to 0..1
        wind_norm = np.clip(obs_env.wind_speed / 30.0, 0.0, 1.0)

        # 7-10. Internal Physiological Vitals
        health_norm = np.clip(human.vitals.health / 100.0, 0.0, 1.0)
        energy_norm = np.clip(human.vitals.energy_hunger / 100.0, 0.0, 1.0)
        thirst_norm = np.clip(human.vitals.hydration_thirst / 100.0, 0.0, 1.0)
        comfort_norm = np.clip(human.vitals.thermal_comfort / 100.0, 0.0, 1.0)

        # 11. Shelter Level: 0..1
        shelter_norm = np.clip(human.shelter_level, 0.0, 1.0)
        # 12. Accumulated Cultural Lore: 0..10 mapped to 0..1
        culture_norm = np.clip(human.knowledge.accumulated_cultural_lore / 10.0, 0.0, 1.0)

        return np.array([
            temp_norm,
            humidity_norm,
            precip_norm,
            biomass_norm,
            water_avail,
            wind_norm,
            health_norm,
            energy_norm,
            thirst_norm,
            comfort_norm,
            shelter_norm,
            culture_norm,
        ], dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        """
        Dynamic action mask (1 = valid, 0 = invalid).
        Prevents wasting learning capacity on physically impossible actions.
        """
        mask = np.ones(self.action_space_size, dtype=np.int32)
        obs_env = self.world.get_current_observation()
        human = self.world.human

        # Cannot gather water if cell is completely arid with no rain and no high skill
        if obs_env.precipitation_rate_mm_h < 0.05 and obs_env.soil_moisture < 0.15 and obs_env.is_land:
            if human.knowledge.freshwater_sourcing_skill < 0.40:
                mask[ACTION_NAME_TO_ID["GATHER_WATER"]] = 0

        # Cannot forage food if vegetation is completely depleted
        if obs_env.vegetation_biomass < 0.02:
            mask[ACTION_NAME_TO_ID["FORAGE_FOOD"]] = 0

        # Cannot teach community if no cultural lore has been gathered yet
        if human.knowledge.accumulated_cultural_lore < 0.05 and human.knowledge.edible_flora_identified < 0.2:
            mask[ACTION_NAME_TO_ID["TEACH_COMMUNITY"]] = 0

        # If population is dead, all actions are invalid except stay
        if human.population <= 0:
            mask[:] = 0
            mask[ACTION_NAME_TO_ID["STAY"]] = 1

        return mask

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action, advance the environment by 1 simulation hour, and compute reward.
        Returns: (observation, reward, terminated, truncated, info)
        """
        human = self.world.human
        obs_env_pre = self.world.get_current_observation()
        pop_pre = human.population
        knowledge_pre = human.knowledge.edible_flora_identified + human.knowledge.freshwater_sourcing_skill

        # 1. Execute chosen intent/action
        action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "STAY"

        # Movement offsets (approx 0.1° ~ 11 km)
        move_offsets = {
            "MOVE_N": (0.1, 0.0),
            "MOVE_NE": (0.07, 0.07),
            "MOVE_E": (0.0, 0.1),
            "MOVE_SE": (-0.07, 0.07),
            "MOVE_S": (-0.1, 0.0),
            "MOVE_SW": (-0.07, -0.07),
            "MOVE_W": (0.0, -0.1),
            "MOVE_NW": (0.07, -0.07),
        }

        if action_name in move_offsets:
            dlat, dlon = move_offsets[action_name]
            human.latitude_deg = float(np.clip(human.latitude_deg + dlat, -85.0, 85.0))
            human.longitude_deg = float((human.longitude_deg + dlon + 180.0) % 360.0 - 180.0)
            human.vitals.energy_hunger = max(0.0, human.vitals.energy_hunger - 0.5)

        elif action_name == "FORAGE_FOOD":
            if obs_env_pre.vegetation_biomass > 0.05:
                human.vitals.energy_hunger = min(100.0, human.vitals.energy_hunger + 5.0)

        elif action_name == "GATHER_WATER":
            human.vitals.hydration_thirst = min(100.0, human.vitals.hydration_thirst + 8.0)

        elif action_name == "SEEK_OR_BUILD_SHELTER":
            human.shelter_level = min(0.95, human.shelter_level + 0.05)
            human.vitals.thermal_comfort = min(100.0, human.vitals.thermal_comfort + 6.0)

        elif action_name == "EXPLORE":
            human._attempt_discovery(obs_env_pre)

        elif action_name == "TEACH_COMMUNITY":
            human.knowledge.accumulated_cultural_lore += 0.08
            human.knowledge.edible_flora_identified = min(1.0, human.knowledge.edible_flora_identified + 0.01)

        # 2. Advance world simulation clock by 1 hour (3600s)
        self.world.scheduler.advance(3600.0 / max(1.0, self.world.scheduler.time_speed))
        self.episode_steps += 1

        # 3. Compute Reward based on selected experiment Goal
        obs_post = self.get_observation()
        reward = self._calculate_reward(action_name, pop_pre, knowledge_pre)

        # 4. Termination conditions
        terminated = (human.population <= 0) or (human.vitals.health <= 0.0)
        truncated = (self.episode_steps >= self.max_episode_steps)

        info = {
            "action": action_name,
            "population": human.population,
            "health": human.vitals.health,
            "hunger": human.vitals.energy_hunger,
            "thirst": human.vitals.hydration_thirst,
            "culture_lore": human.knowledge.accumulated_cultural_lore,
        }

        return obs_post, reward, terminated, truncated, info

    def _calculate_reward(self, action_name: str, pop_pre: int, knowledge_pre: float) -> float:
        human = self.world.human

        # Fatal penalty
        if human.population <= 0 or human.vitals.health <= 0.0:
            return -100.0

        if self.goal_type == "Growth":
            # Rewarded for demographic growth and surplus health
            pop_gain = (human.population - pop_pre) * 10.0
            health_bonus = (human.vitals.health - 50.0) * 0.02
            return 0.1 + pop_gain + health_bonus

        elif self.goal_type == "Exploration":
            # Rewarded for discoveries and movement
            discovery_bonus = 2.0 if action_name == "EXPLORE" else 0.05
            return 0.1 + discovery_bonus

        elif self.goal_type == "Culture":
            # Rewarded for knowledge transmission and accumulation
            knowledge_post = human.knowledge.edible_flora_identified + human.knowledge.freshwater_sourcing_skill
            k_gain = (knowledge_post - knowledge_pre) * 20.0
            teach_bonus = 1.0 if action_name == "TEACH_COMMUNITY" else 0.0
            return 0.1 + k_gain + teach_bonus

        else: # Default: "Survive"
            # Homeostatic physiological survival balance
            base_alive = 1.0
            hunger_penalty = (100.0 - human.vitals.energy_hunger) * 0.005
            thirst_penalty = (100.0 - human.vitals.hydration_thirst) * 0.008
            thermal_penalty = (100.0 - human.vitals.thermal_comfort) * 0.004
            return base_alive - (hunger_penalty + thirst_penalty + thermal_penalty)
