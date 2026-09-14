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


class QLearningAgent:
    """
    Active Tabular/State-Discretized Q-Learning agent with Bellman TD updates.
    Verifies the closed loop:
    observation -> policy -> action -> environment transition -> reward -> next observation -> policy update
    """

    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.30,
        epsilon_decay: float = 0.998,
        epsilon_min: float = 0.05,
    ):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state_tuple -> np.ndarray of shape (num_actions,)
        self.q_table: dict[tuple, np.ndarray] = {}

        # Learning telemetry
        self.policy_updates = 0
        self.total_reward = 0.0
        self.latest_td_error = 0.0
        self.td_error_history: list[float] = []

    def discretize_state(self, obs: np.ndarray) -> tuple:
        """
        Discretizes 12-dim continuous observation vector into a compact state tuple:
        (temp_bin, water_avail_bin, biomass_bin, health_bin, energy_bin, thirst_bin, shelter_bin)
        """
        t_bin = 0 if obs[0] < 0.35 else (2 if obs[0] > 0.70 else 1)
        w_bin = 0 if obs[4] < 0.25 else (2 if obs[4] > 0.60 else 1)
        bio_bin = 0 if obs[3] < 0.20 else (2 if obs[3] > 0.60 else 1)
        hp_bin = 0 if obs[6] < 0.35 else (2 if obs[6] > 0.75 else 1)
        eng_bin = 0 if obs[7] < 0.35 else (2 if obs[7] > 0.75 else 1)
        thirst_bin = 0 if obs[8] < 0.35 else (2 if obs[8] > 0.75 else 1)
        sh_bin = 0 if obs[10] < 0.25 else (2 if obs[10] > 0.65 else 1)
        return (t_bin, w_bin, bio_bin, hp_bin, eng_bin, thirst_bin, sh_bin)

    def get_q_values(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: tuple, action_mask: np.ndarray, rng: np.random.RandomState | None = None) -> int:
        """Epsilon-greedy action selection obeying action mask."""
        if rng is None:
            rng = np.random.RandomState()

        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return 0  # STAY

        if rng.rand() < self.epsilon:
            # Exploration among valid actions
            return int(rng.choice(valid_actions))
        else:
            # Exploitation: argmax Q over valid actions
            q_vals = self.get_q_values(state).copy()
            q_vals[action_mask == 0] = -1e9
            best_actions = np.where(q_vals == np.max(q_vals))[0]
            best_valid = [a for a in best_actions if a in valid_actions]
            return int(rng.choice(best_valid if best_valid else valid_actions))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_mask: np.ndarray,
        terminated: bool,
    ):
        """
        Bellman equation TD update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.get_q_values(state)[action]

        if terminated:
            target = reward
        else:
            next_q = self.get_q_values(next_state).copy()
            next_q[next_mask == 0] = -1e9
            max_next_q = np.max(next_q) if np.any(next_mask == 1) else 0.0
            target = reward + self.gamma * max_next_q

        td_error = float(target - current_q)
        self.q_table[state][action] += self.lr * td_error

        # Telemetry updates
        self.policy_updates += 1
        self.latest_td_error = td_error
        self.td_error_history.append(abs(td_error))
        if len(self.td_error_history) > 100:
            self.td_error_history.pop(0)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_reward += reward


class RLLSEnvironmentInterface:
    """
    RLLS 16 Reinforcement Learning Interface.
    Standardized Gym-style environment wrapper around the Earth simulation with
    active closed-loop policy learning and Bellman TD updates.
    """

    def __init__(self, world_instance: "WorldInstance", goal_type: str = "Survive"):
        self.world = world_instance
        self.goal_type = goal_type  # "Survive", "Growth", "Exploration", "Culture"

        self.action_space_size = len(ACTIONS)
        self.observation_dim = 12

        # Step count tracking for episodes
        self.episode_steps = 0
        self.max_episode_steps = 1000

        # Active learning agent
        self.agent = QLearningAgent(num_actions=self.action_space_size)
        self.rng = np.random.RandomState(42)
        self.last_transition: dict = {}

    def get_observation(self) -> np.ndarray:
        """
        Produce a normalized observation vector in [0.0, 1.0] representing
        local environmental conditions and internal physiological vitals.
        """
        obs_env = self.world.get_current_observation()
        human = self.world.human

        temp_norm = np.clip((obs_env.temperature_c + 40.0) / 90.0, 0.0, 1.0)
        humidity_norm = np.clip(obs_env.humidity / 0.025, 0.0, 1.0)
        precip_norm = np.clip(obs_env.precipitation_rate_mm_h / 30.0, 0.0, 1.0)
        biomass_norm = np.clip(obs_env.vegetation_biomass, 0.0, 1.0)
        water_avail = np.clip(obs_env.soil_moisture, 0.0, 1.0)
        wind_norm = np.clip(obs_env.wind_speed / 30.0, 0.0, 1.0)

        health_norm = np.clip(human.vitals.health / 100.0, 0.0, 1.0)
        energy_norm = np.clip(human.vitals.energy_hunger / 100.0, 0.0, 1.0)
        thirst_norm = np.clip(human.vitals.hydration_thirst / 100.0, 0.0, 1.0)
        comfort_norm = np.clip(human.vitals.thermal_comfort / 100.0, 0.0, 1.0)

        shelter_norm = np.clip(human.shelter_level, 0.0, 1.0)
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

        # Cannot gather water if cell is completely arid with no rain and low skill
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

    def step(self, action_id: int, advance_world_clock: bool = True) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action, advance the environment, and compute reward.
        Returns: (observation, reward, terminated, truncated, info)
        """
        human = self.world.human
        obs_env_pre = self.world.get_current_observation()
        pop_pre = human.population
        knowledge_pre = human.knowledge.edible_flora_identified + human.knowledge.freshwater_sourcing_skill

        # 1. Execute chosen intent/action
        action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "STAY"

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

        # 2. Advance world simulation clock by 1 hour (3600s) if requested
        if advance_world_clock:
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

    def step_and_learn(self, advance_world_clock: bool = True) -> dict:
        """
        Executes one full verified RL transition loop:
        observation -> policy -> action -> environment transition -> reward -> next observation -> policy update
        """
        if getattr(self, "_is_updating", False):
            return self.last_transition
        self._is_updating = True
        try:
            obs = self.get_observation()
            mask = self.get_action_mask()
            state = self.agent.discretize_state(obs)

            # 1. Action selection from policy
            action_id = self.agent.select_action(state, mask, self.rng)

            # 2. Environment transition & reward
            next_obs, reward, terminated, truncated, info = self.step(action_id, advance_world_clock=advance_world_clock)
            next_mask = self.get_action_mask()
            next_state = self.agent.discretize_state(next_obs)

            # 3. Policy / Q-value update via Bellman equation
            self.agent.update(state, action_id, reward, next_state, next_mask, terminated)

            self.last_transition = {
                "state": state,
                "action_id": action_id,
                "action_name": ACTIONS[action_id],
                "reward": reward,
                "next_state": next_state,
                "terminated": terminated,
                "td_error": self.agent.latest_td_error,
                "policy_updates": self.agent.policy_updates,
            }
            return self.last_transition
        finally:
            self._is_updating = False

    def get_learning_telemetry(self) -> dict:
        """Returns verified RL learning statistics."""
        avg_td = float(np.mean(self.agent.td_error_history)) if self.agent.td_error_history else 0.0
        return {
            "policy_updates": self.agent.policy_updates,
            "q_table_states": len(self.agent.q_table),
            "avg_td_error": avg_td,
            "latest_td_error": self.agent.latest_td_error,
            "epsilon": self.agent.epsilon,
            "total_reward": self.agent.total_reward,
            "last_action": self.last_transition.get("action_name", "NONE"),
            "last_reward": self.last_transition.get("reward", 0.0),
        }

    def _calculate_reward(self, action_name: str, pop_pre: int, knowledge_pre: float) -> float:
        human = self.world.human

        # Fatal penalty
        if human.population <= 0 or human.vitals.health <= 0.0:
            return -100.0

        if self.goal_type == "Growth":
            pop_gain = (human.population - pop_pre) * 10.0
            health_bonus = (human.vitals.health - 50.0) * 0.02
            return 0.1 + pop_gain + health_bonus

        elif self.goal_type == "Exploration":
            discovery_bonus = 2.0 if action_name == "EXPLORE" else 0.05
            return 0.1 + discovery_bonus

        elif self.goal_type == "Culture":
            knowledge_post = human.knowledge.edible_flora_identified + human.knowledge.freshwater_sourcing_skill
            k_gain = (knowledge_post - knowledge_pre) * 20.0
            teach_bonus = 1.0 if action_name == "TEACH_COMMUNITY" else 0.0
            return 0.1 + k_gain + teach_bonus

        else:  # Default: "Survive"
            base_alive = 1.0
            hunger_penalty = (100.0 - human.vitals.energy_hunger) * 0.005
            thirst_penalty = (100.0 - human.vitals.hydration_thirst) * 0.008
            thermal_penalty = (100.0 - human.vitals.thermal_comfort) * 0.004
            return base_alive - (hunger_penalty + thirst_penalty + thermal_penalty)
