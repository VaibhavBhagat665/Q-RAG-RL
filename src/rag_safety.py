import numpy as np
from typing import Dict, Any, Tuple

class OptimizedRAGSafety:
    """RAG-based safety module with graded penalty functions"""
    def __init__(self, penalty_sharpness=5.5, margin_tightness=0.88):
        self.penalty_sharpness = penalty_sharpness
        self.margin_tightness = margin_tightness
    
    def _smooth_penalty(self, value: float, min_limit: float, max_limit: float, 
                       base_penalty: float) -> float:
        if min_limit <= value <= max_limit:
            return 0.0
        
        margin = (max_limit - min_limit) * 0.08
        
        if value < min_limit:
            distance = min_limit - value
            normalized_dist = distance / margin
            return base_penalty * (normalized_dist ** self.penalty_sharpness)
        else:
            distance = value - max_limit
            normalized_dist = distance / margin
            return base_penalty * (normalized_dist ** self.penalty_sharpness)
    
    def check_safety_graded(self, action: float, current_state: Dict[str, Any]) -> Tuple[bool, float]:
        battery_soc = current_state.get('battery_soc', 0.5)
        voltage = current_state.get('voltage', 1.0)
        frequency = current_state.get('frequency', 50.0)
        total_gen = current_state.get('total_generation', 0)
        load = current_state.get('load', 1.0)
        
        total_penalty = 0.0
        violations = []
        
        soc_min = 0.105
        soc_max = 0.895
        soc_penalty = self._smooth_penalty(battery_soc, soc_min, soc_max, 2000)
        total_penalty += soc_penalty
        if battery_soc < 0.1 or battery_soc > 0.9:
            violations.append("Battery SOC violation")
        
        power_limit = 0.98
        power_penalty = self._smooth_penalty(abs(action), 0, power_limit, 1500)
        total_penalty += power_penalty
        if abs(action) > 1.0:
            violations.append("Battery power limit violation")
        
        v_min = 0.952
        v_max = 1.048
        voltage_penalty = self._smooth_penalty(voltage, v_min, v_max, 1000)
        total_penalty += voltage_penalty
        if voltage < 0.95 or voltage > 1.05:
            violations.append("Voltage violation")
        
        f_min = 49.52
        f_max = 50.48
        freq_penalty = self._smooth_penalty(frequency, f_min, f_max, 2000)
        total_penalty += freq_penalty
        if frequency < 49.5 or frequency > 50.5:
            violations.append("Frequency violation")
        
        gen_ratio = total_gen / (load + 1e-6)
        gen_max = 1.095
        gen_penalty = self._smooth_penalty(gen_ratio, 0, gen_max, 1200)
        total_penalty += gen_penalty
        if gen_ratio > 1.1:
            violations.append("Generation-load balance violation")
        
        is_safe = len(violations) == 0
        return is_safe, total_penalty
