import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Tuple
from .rag_safety import RAGSafetyModule

class MicrogridEnv(gym.Env):
    def __init__(self, scenarios=None, rag_safety=None):
        super().__init__()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([5, 5, 5, 1, 24, 0.2]),
            dtype=np.float32
        )
        
        self.scenarios = scenarios if scenarios is not None else np.random.rand(1000, 4)
        self.rag_safety = rag_safety if rag_safety is not None else RAGSafetyModule()
        
        self.max_steps = 96
        self.solar_capacity = 3.0
        self.wind_capacity = 2.0
        self.battery_capacity = 5.0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.battery_soc = 0.5
        self.scenario_idx = np.random.randint(0, len(self.scenarios))
        self.current_scenario = self.scenarios[self.scenario_idx]
        
        self.total_cost = 0
        self.safety_violations = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        solar_gen = self.current_scenario[0] * np.sin(2 * np.pi * (self.step_count % 96) / 96)
        solar_gen = np.clip(solar_gen, 0, self.solar_capacity)
        
        wind_gen = self.current_scenario[1] * (0.5 + 0.5 * np.sin(2 * np.pi * (self.step_count % 96) / 96 + np.pi/4))
        wind_gen = np.clip(wind_gen, 0, self.wind_capacity)
        
        load = self.current_scenario[2] * (0.7 + 0.3 * np.sin(2 * np.pi * (self.step_count % 96) / 96 + np.pi))
        
        time_of_day = (self.step_count % 96) / 96
        electricity_price = self.current_scenario[3]
        
        return np.array([solar_gen, wind_gen, load, self.battery_soc, time_of_day, electricity_price], dtype=np.float32)
    
    def step(self, action):
        action = np.clip(action[0], -1.0, 1.0)
        
        obs = self._get_obs()
        solar_gen, wind_gen, load, _, time_of_day, price = obs
        
        current_state = {
            'battery_soc': self.battery_soc,
            'voltage': 1.0 + np.random.normal(0, 0.01),
            'frequency': 50.0 + np.random.normal(0, 0.1),
            'total_generation': solar_gen + wind_gen + action,
            'load': load
        }
        
        is_safe, safety_penalty = self.rag_safety.check_safety(action, current_state)
        
        if not is_safe:
            self.safety_violations += 1
        
        new_soc = self.battery_soc - (action * 0.25 / self.battery_capacity)
        self.battery_soc = np.clip(new_soc, 0.0, 1.0)
        
        grid_power = load - solar_gen - wind_gen - action
        
        if grid_power > 0:
            cost = grid_power * price * 0.25
        else:
            cost = grid_power * price * 0.1 * 0.25
        
        self.total_cost += cost
        
        reward = -cost - safety_penalty * 0.001
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            'cost': cost,
            'safety_penalty': safety_penalty,
            'is_safe': is_safe,
            'battery_soc': self.battery_soc,
            'grid_power': grid_power
        }
        
        return self._get_obs(), reward, terminated, truncated, info

class SafetyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

class DRLAgent:
    def __init__(self, scenarios=None):
        self.scenarios = scenarios
        self.rag_safety = RAGSafetyModule()
        self.model = None
        
    def create_env(self):
        return MicrogridEnv(scenarios=self.scenarios, rag_safety=self.rag_safety)
    
    def train(self, total_timesteps=100000):
        env = make_vec_env(lambda: self.create_env(), n_envs=4)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        callback = SafetyCallback()
        
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        return self.model
    
    def predict(self, observation):
        if self.model is None:
            return np.array([0.0])
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save_model(self, path):
        if self.model is not None:
            self.model.save(path)
    
    def load_model(self, path):
        env = make_vec_env(lambda: self.create_env(), n_envs=1)
        self.model = PPO.load(path, env=env)
