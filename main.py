"""
Main simulation script for Q-RAG-RL system.
Integrates quantum scenario generation, RAG safety, DRL agent, and digital twin.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantum_scenarios import QuantumScenarioGenerator
from src.rag_safety import OptimizedRAGSafety
from src.drl_agent import OptimizedDRLAgent, OptimizedMicrogridEnv
from src.digital_twin import MicrogridDigitalTwin
from src.nrel_data import NRELDataLoader
from src.tariff import TariffModel
import numpy as np
import pandas as pd

def run_simulation(lat=39.7391, lon=-104.9847, year=2022, 
                   penalty_sharpness=5.5, margin_tightness=0.88):
    print("Q-RAG-RL System")
    print("=" * 60)
    
    print("1. Quantum Scenario Generator...")
    quantum_gen = QuantumScenarioGenerator(n_qubits=6, n_scenarios=4000)
    quantum_scenarios = quantum_gen.generate_scenarios()
    print(f"Generated {len(quantum_scenarios)} quantum scenarios")
    
    print("\n2. RAG Safety Module...")
    print(f"Penalty Sharpness: {penalty_sharpness}")
    print(f"Margin Tightness: {margin_tightness}")
    
    print("\n3. Digital Twin...")
    digital_twin = MicrogridDigitalTwin(source="case33bw")
    
    print("\n4. NREL Data...")
    nrel_loader = NRELDataLoader()
    solar_df = nrel_loader.get_solar_data(lat=lat, lon=lon, year=year)
    wind_df = nrel_loader.get_wind_data(lat=lat, lon=lon, year=year)
    solar_power, wind_power = nrel_loader.convert_to_power(solar_df, wind_df)
    load_profile = nrel_loader.get_load_profile()
    
    print("\n5. Training DRL Agent...")
    drl_agent = OptimizedDRLAgent(
        scenarios=quantum_scenarios,
        penalty_sharpness=penalty_sharpness,
        margin_tightness=margin_tightness
    )
    model = drl_agent.train(total_timesteps=200000)
    print("Training completed")
    print(f"Final Lambda: {drl_agent.cmdp_optimizer.lambda_multiplier:.4f}")
    
    print("\n6. Running 24-Hour Simulation...")
    results = []
    tariff = TariffModel(lat=lat, lon=lon)
    prices = tariff.get_hourly_prices(hours=24)
    battery_soc = 0.5
    
    for hour in range(24):
        solar_gen_values = [solar_power[hour] / 3] * 3
        wind_gen_values = [wind_power[hour] / 2] * 2
        
        current_state = {
            'battery_soc': battery_soc,
            'voltage': 1.0 + np.random.normal(0, 0.01),
            'frequency': 50.0 + np.random.normal(0, 0.1),
            'total_generation': sum(solar_gen_values) + sum(wind_gen_values),
            'load': load_profile[hour]
        }
        
        obs = np.array([
            sum(solar_gen_values), 
            sum(wind_gen_values), 
            load_profile[hour], 
            battery_soc, 
            hour/24, 
            prices[hour]
        ])
        
        action = drl_agent.predict(obs)[0]
        
        is_safe, safety_penalty = drl_agent.rag_safety.check_safety_graded(action, current_state)
        
        converged, pf_results = digital_twin.simulate_timestep(
            solar_gen_values, wind_gen_values, action
        )
        
        grid_power = load_profile[hour] - sum(solar_gen_values) - sum(wind_gen_values) - action
        if grid_power > 0:
            cost = grid_power * prices[hour]
        else:
            cost = grid_power * prices[hour] * 0.1
        
        battery_soc = float(np.clip(battery_soc - (action * 1.0 / 5.0), 0.0, 1.0))
        
        results.append({
            'hour': hour,
            'solar_gen': sum(solar_gen_values),
            'wind_gen': sum(wind_gen_values),
            'load': load_profile[hour],
            'battery_action': action,
            'battery_soc': battery_soc,
            'price': prices[hour],
            'cost': cost,
            'is_safe': is_safe,
            'safety_penalty': safety_penalty,
            'converged': converged,
            'min_voltage': pf_results.get('min_voltage', 0) if converged else 0,
            'max_voltage': pf_results.get('max_voltage', 0) if converged else 0,
            'total_losses': pf_results.get('total_losses', 0) if converged else 0
        })
        
        print(f"H{hour:2d}: ${cost:6.2f} Safe={is_safe} Act={action:6.3f}MW SOC={battery_soc:.2f}")
    
    print("\n7. Results:")
    df = pd.DataFrame(results)
    
    total_cost = df['cost'].sum()
    violations = len(df[df['is_safe'] == False])
    
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Safety Violations: {violations}/24")
    print(f"Avg Battery Action: {df['battery_action'].mean():.3f} MW")
    print(f"Convergence: {len(df[df['converged'] == True])}/24")
    
    if len(df[df['converged'] == True]) > 0:
        converged_df = df[df['converged'] == True]
        print(f"Min Voltage: {converged_df['min_voltage'].min():.4f} pu")
        print(f"Max Voltage: {converged_df['max_voltage'].max():.4f} pu")
    
    df.to_csv('simulation_results.csv', index=False)
    print("\nSaved to simulation_results.csv")
    
    return df, drl_agent

if __name__ == "__main__":
    run_simulation()
