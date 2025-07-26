import numpy as np
import gym
from gym import spaces
from config import ENV_CONFIG

class AquaFlowEnv(gym.Env):
    def __init__(self):
        super(AquaFlowEnv, self).__init__()
        self.num_nodes = ENV_CONFIG["num_nodes"]
        self.time_steps = ENV_CONFIG["time_steps"]
        self.capacity = ENV_CONFIG["capacity_per_node"]

        self.current_step = 0
        self.state = None

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes * 3,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.inflow = np.random.uniform(0.2, 1.0, (self.time_steps, self.num_nodes))
        self.weather = np.random.normal(0.5, ENV_CONFIG["weather_variation"], (self.time_steps,))
        self.levels = np.zeros((self.time_steps, self.num_nodes))
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        inflow = self.inflow[self.current_step]
        level = self.levels[self.current_step]
        weather = np.full(self.num_nodes, self.weather[self.current_step])
        return np.concatenate([inflow, level, weather])

    def step(self, action):
        inflow = self.inflow[self.current_step]
        new_levels = self.levels[self.current_step] + inflow - action * inflow
        overflow = np.maximum(new_levels - self.capacity, 0.0)
        self.levels[self.current_step] = np.minimum(new_levels, self.capacity)

        energy_use = np.sum(action) * ENV_CONFIG["actuator_power_coeff"]

        reward = - np.sum(overflow) - ENV_CONFIG["energy_penalty"] * energy_use

        self.current_step += 1
        done = self.current_step >= self.time_steps

        next_state = self._get_state() if not done else np.zeros_like(self.state)
        info = {
            "overflow": np.sum(overflow),
            "energy": energy_use,
        }
        return next_state, reward, done, info
