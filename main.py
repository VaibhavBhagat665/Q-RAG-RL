import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantum_module import QuantumScenarioGenerator, ClassicalGAN
from src.rag_safety import RAGSafetyModule
from src.drl_agent import DRLAgent, MicrogridEnv
from src.digital_twin import MicrogridDigitalTwin
from src.nrel_data import NRELDataLoader
from src.tariff import TariffModel
import numpy as np
import pandas as pd
from datetime import datetime

def run_complete_simulation(lat: float = 39.7391, lon: float = -104.9847, year: int = 2022, twin_source: str = "case33bw"):
    print("🔋 Q-RAG-RL Microgrid Management System")
    print("=" * 50)
    
    print("1️⃣ Initializing Quantum Scenario Generator...")
    quantum_gen = QuantumScenarioGenerator(n_qubits=4, n_scenarios=1000)
    quantum_scenarios = quantum_gen.generate_scenarios()
    print(f"Generated {len(quantum_scenarios)} quantum scenarios")
    
    print("\n2️⃣ Initializing Classical GAN Baseline...")
    classical_gan = ClassicalGAN()
    classical_scenarios = classical_gan.generate_scenarios()
    print(f"Generated {len(classical_scenarios)} classical scenarios")
    
    print("\n3️⃣ Initializing RAG Safety Module...")
    rag_safety = RAGSafetyModule()
    print("Safety constraints loaded into vector database")
    
    print("\n4️⃣ Creating Digital Twin (IEEE 33-Bus System)...")
    digital_twin = MicrogridDigitalTwin(source=twin_source)
    print("IEEE 33-bus microgrid network created")
    
    print("\n5️⃣ Loading NREL Weather Data...")
    nrel_loader = NRELDataLoader()
    solar_df = nrel_loader.get_solar_data(lat=lat, lon=lon, year=year)
    wind_df = nrel_loader.get_wind_data(lat=lat, lon=lon, year=year)
    solar_power, wind_power = nrel_loader.convert_to_power(solar_df, wind_df)
    load_profile = nrel_loader.get_load_profile()
    print(f"Loaded {len(solar_power)} hours of renewable generation data")
    
    print("\n6️⃣ Training DRL Agent with Quantum Scenarios...")
    drl_agent = DRLAgent(scenarios=quantum_scenarios)
    model = drl_agent.train(total_timesteps=50000)
    print("DRL agent training completed")
    
    print("\n7️⃣ Running 24-Hour Simulation...")
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
        
        is_safe, safety_penalty = rag_safety.check_safety(action, current_state)
        
        converged, pf_results = digital_twin.simulate_timestep(
            solar_gen_values, wind_gen_values, action
        )
        
        grid_power = load_profile[hour] - sum(solar_gen_values) - sum(wind_gen_values) - action
        if grid_power > 0:
            cost = grid_power * prices[hour]
        else:
            cost = grid_power * prices[hour] * 0.1
        # update battery SOC (4 x 15-min timesteps approximated as 1h here)
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
        
        print(f"Hour {hour:2d}: Cost=${cost:6.2f}, Safe={is_safe}, Action={action:6.3f}MW")
    
    print("\n8️⃣ Simulation Results Summary:")
    df = pd.DataFrame(results)
    
    print(f"Total Cost: ${df['cost'].sum():.2f}")
    print(f"Average Cost: ${df['cost'].mean():.2f}/hour")
    print(f"Safety Violations: {len(df[df['is_safe'] == False])}/24 hours")
    print(f"Power Flow Convergence: {len(df[df['converged'] == True])}/24 hours")
    print(f"Average Battery Action: {df['battery_action'].mean():.3f} MW")
    print(f"Average Price: ${df['price'].mean():.2f}/kWh")
    
    if len(df[df['converged'] == True]) > 0:
        converged_df = df[df['converged'] == True]
        print(f"Min Voltage: {converged_df['min_voltage'].min():.4f} pu")
        print(f"Max Voltage: {converged_df['max_voltage'].max():.4f} pu")
        print(f"Total Losses: {converged_df['total_losses'].sum():.3f} MW")
    
    df.to_csv('simulation_results.csv', index=False)
    print("\n✅ Results saved to simulation_results.csv")
    
    return df

def test_individual_components():
    print("\n🧪 Testing Individual Components:")
    print("-" * 40)
    
    print("Testing Quantum Module...")
    quantum_gen = QuantumScenarioGenerator(n_qubits=4, n_scenarios=100)
    quantum_scenarios = quantum_gen.generate_scenarios()
    print(f"✅ Quantum scenarios shape: {quantum_scenarios.shape}")
    
    print("\nTesting RAG Safety...")
    rag_safety = RAGSafetyModule()
    test_state = {
        'battery_soc': 0.5,
        'voltage': 1.0,
        'frequency': 50.0,
        'total_generation': 3.0,
        'load': 2.5
    }
    is_safe, penalty = rag_safety.check_safety(0.5, test_state)
    print(f"✅ Safety check: Safe={is_safe}, Penalty={penalty}")
    
    print("\nTesting Digital Twin...")
    digital_twin = MicrogridDigitalTwin()
    converged, results = digital_twin.simulate_timestep([1.0, 1.0, 1.0], [0.8, 0.8], 0.2)
    print(f"✅ Power flow: Converged={converged}")
    
    print("\nTesting DRL Environment...")
    env = MicrogridEnv(scenarios=quantum_scenarios[:10])
    obs, _ = env.reset()
    action = np.array([0.5])
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Environment step: Reward={reward:.3f}, Done={done}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Q-RAG-RL Microgrid Management System")
    parser.add_argument("--mode", choices=["full", "test", "dashboard"], default="full",
                       help="Run mode: full simulation, component tests, or dashboard")
    parser.add_argument("--lat", type=float, default=39.7391, help="Latitude for NREL data and tariff")
    parser.add_argument("--lon", type=float, default=-104.9847, help="Longitude for NREL data and tariff")
    parser.add_argument("--year", type=int, default=2022, help="Year for NREL data")
    parser.add_argument("--twin", type=str, choices=["custom", "case33bw"], default="case33bw", help="Digital twin source")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_complete_simulation(lat=args.lat, lon=args.lon, year=args.year, twin_source=args.twin)
    elif args.mode == "test":
        test_individual_components()
    elif args.mode == "dashboard":
        import subprocess
        subprocess.run(["streamlit", "run", "src/dashboard.py"])
