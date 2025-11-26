import numpy as np
from typing import Dict, List, Tuple

class CMDPLagrangianOptimizer:
    def __init__(self, 
                 initial_lambda=0.02,
                 lambda_lr=0.025,
                 lambda_decay=0.998,
                 min_lambda_lr=0.0005,
                 adaptive_threshold=0.8,
                 aggressive_decay_start=200):
        self.lambda_multiplier = initial_lambda
        self.lambda_lr = lambda_lr
        self.lambda_decay = lambda_decay
        self.min_lambda_lr = min_lambda_lr
        self.adaptive_threshold = adaptive_threshold
        self.aggressive_decay_start = aggressive_decay_start
        self.episode_count = 0
        self.violation_history = []
        self.cost_history = []
        
    def update_lagrange_multiplier(self, constraint_violation_sum: float, 
                                   episode_count: int) -> float:
        self.episode_count = episode_count
        self.violation_history.append(constraint_violation_sum)
        
        current_lr = max(self.lambda_lr * (self.lambda_decay ** episode_count), 
                        self.min_lambda_lr)
        
        if len(self.violation_history) > 50:
            recent_violations = np.mean(self.violation_history[-50:])
            if recent_violations < 0.05:
                current_lr *= 0.3
        
        gradient = constraint_violation_sum
        self.lambda_multiplier = max(0.0, self.lambda_multiplier + current_lr * gradient)
        
        if episode_count > self.aggressive_decay_start and len(self.violation_history) > 100:
            violation_rate = np.mean([1 if v > 0 else 0 for v in self.violation_history[-100:]])
            if violation_rate < 0.02:
                self.lambda_multiplier *= 0.88
            elif violation_rate < 0.05:
                self.lambda_multiplier *= 0.93
        
        return self.lambda_multiplier
    
    def compute_lagrangian_reward(self, base_reward: float, constraint_cost: float) -> float:
        return base_reward - self.lambda_multiplier * constraint_cost
    
    def get_adaptive_penalty_weight(self, episode: int, total_episodes: int) -> float:
        progress = episode / max(total_episodes, 1)
        
        if progress < 0.25:
            return 1.0
        elif progress < 0.6:
            return 0.6 + 0.4 * (1 - (progress - 0.25) / 0.35)
        else:
            return 0.35 + 0.25 * (1 - (progress - 0.6) / 0.4)
    
    def should_explore_boundary(self, episode: int, total_episodes: int) -> bool:
        progress = episode / max(total_episodes, 1)
        if progress > 0.6 and len(self.violation_history) > 100:
            recent_violation_rate = np.mean([1 if v > 0 else 0 for v in self.violation_history[-100:]])
            return recent_violation_rate < 0.05
        return False
    
    def get_statistics(self) -> Dict:
        return {
            'lambda': self.lambda_multiplier,
            'current_lr': max(self.lambda_lr * (self.lambda_decay ** self.episode_count), 
                            self.min_lambda_lr),
            'total_violations': sum(1 for v in self.violation_history if v > 0),
            'violation_rate': np.mean([1 if v > 0 else 0 for v in self.violation_history]) if self.violation_history else 0,
            'avg_violation_magnitude': np.mean([abs(v) for v in self.violation_history]) if self.violation_history else 0
        }
