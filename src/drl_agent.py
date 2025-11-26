import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any
from .rag_safety import OptimizedRAGSafety
from .cmdp_optimizer import CMDPLagrangianOptimizer

class OptimizedMicrogridEnv(gym.Env):
    def __init__(self, scenarios=None, rag_safety=None, cmdp_optimizer=None, 
                 use_graded_penalty=True):
        super().__init__()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([5, 5, 5, 1, 24, 0.2]),
            dtype=np.float32
        )
        
        self.scenarios = scenarios if scenarios is not None else np.random.rand(1000, 4)
        self.rag_safety = rag_safety if rag_safety is not None else OptimizedRAGSafety()
        self.cmdp_optimizer = cmdp_optimizer if cmdp_optimizer is not None else CMDPLagrangianOptimizer()
        self.use_graded_penalty = use_graded_penalty
        
        self.max_steps = 96
        self.solar_capacity = 3.0
        self.wind_capacity = 2.0
        self.battery_capacity = 5.0
        
        self.episode_count = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.battery_soc = 0.5
        self.scenario_idx = np.random.randint(0, len(self.scenarios))
        self.current_scenario = self.scenarios[self.scenario_idx]
        
        self.total_cost = 0
        self.safety_violations = 0
        self.episode_constraint_cost = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        solar_gen = self.current_scenario[0] * np.sin(2 * np.pi * (self.step_count % 96) / 96)
        solar_gen = np.clip(solar_gen, 0, self.solar_capacity)
        
        wind_gen = self.current_scenario[1] * (0.5 + 0.5 * np.sin(2 * np.pi * (self.step_count % 96) / 96 + np.pi/4))
        wind_gen = np.clip(wind_gen, 0, self.wind_capacity)
        
        load = self.current_scenario[2] * (0.7 + 0.3 * np.sin(2 * np.pi * (self.step_count % 96) / 96 + np.pi))
        
        time_of_day = (self.step_count % 96) / 96
        hour = int((self.step_count % 96) // 4)
        if 0 <= hour < 7:
            base_price = 0.12
        elif 17 <= hour < 22:
            base_price = 0.20
        else:
            base_price = 0.15
        price_factor = 0.8 + 0.4 * self.current_scenario[3]
        electricity_price = base_price * price_factor
        
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
        
        if self.use_graded_penalty:
            is_safe, safety_penalty = self.rag_safety.check_safety_graded(action, current_state)
        else:
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
        
        grid_penalty = 0.02 * abs(grid_power)
        
        normalized_safety_penalty = safety_penalty / 3000.0
        self.episode_constraint_cost += normalized_safety_penalty
        
        penalty_weight = self.cmdp_optimizer.get_adaptive_penalty_weight(
            self.episode_count, 1000
        )
        
        base_reward = -cost - grid_penalty
        constraint_penalty = normalized_safety_penalty * penalty_weight
        lagrangian_reward = self.cmdp_optimizer.compute_lagrangian_reward(
            base_reward, constraint_penalty
        )
        
        lagrangian_reward = np.clip(lagrangian_reward, -100, 100)
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        
        if terminated:
            self.cmdp_optimizer.update_lagrange_multiplier(
                self.episode_constraint_cost, self.episode_count
            )
            self.episode_count += 1
        
        truncated = False
        
        info = {
            'cost': cost,
            'safety_penalty': safety_penalty,
            'is_safe': is_safe,
            'battery_soc': self.battery_soc,
            'grid_power': grid_power,
            'lambda': self.cmdp_optimizer.lambda_multiplier
        }
        
        return self._get_obs(), lagrangian_reward, terminated, truncated, info

class OptimizedDRLAgent:
    def __init__(self, scenarios=None, penalty_sharpness=5.5, margin_tightness=0.88):
        self.scenarios = scenarios
        self.rag_safety = OptimizedRAGSafety(
            penalty_sharpness=penalty_sharpness,
            margin_tightness=margin_tightness
        )
        self.cmdp_optimizer = CMDPLagrangianOptimizer(
            initial_lambda=0.05,
            lambda_lr=0.02,
            lambda_decay=0.998,
            aggressive_decay_start=250
        )
        self.model = None
        
    def create_env(self):
        return OptimizedMicrogridEnv(
            scenarios=self.scenarios,
            rag_safety=self.rag_safety,
            cmdp_optimizer=self.cmdp_optimizer,
            use_graded_penalty=True
        )
    
    def train(self, total_timesteps=200000):
        env = make_vec_env(lambda: self.create_env(), n_envs=4)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=6e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=18,
            gamma=0.996,
            gae_lambda=0.98,
            clip_range=0.28,
            ent_coef=0.018,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        self.model.learn(total_timesteps=total_timesteps)
        
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
