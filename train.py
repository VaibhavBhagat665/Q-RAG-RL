"""
Train DRL agent on quantum-generated scenarios using PPO algorithm.
Loads baseline scenarios, augments them, and trains for 300k timesteps.
"""
import numpy as np
import pandas as pd
from src.drl_agent import OptimizedDRLAgent

print("="*70)
print("Q-RAG-RL AGENT TRAINING")
print("="*70)

baseline_df = pd.read_csv('simulation_results.csv')

print(f"\n[1/3] Extracting baseline scenario distribution...")
baseline_scenarios = baseline_df[['solar_gen', 'wind_gen', 'load', 'price']].values
print(f"  Extracted {len(baseline_scenarios)} baseline scenarios")
print(f"  Solar: {baseline_scenarios[:,0].min():.3f} - {baseline_scenarios[:,0].max():.3f} MW")
print(f"  Wind: {baseline_scenarios[:,1].min():.3f} - {baseline_scenarios[:,1].max():.3f} MW")
print(f"  Load: {baseline_scenarios[:,2].min():.3f} - {baseline_scenarios[:,2].max():.3f} MW")

print("\n[2/3] Generating augmented scenarios...")
augmented_scenarios = []

for _ in range(500):
    idx = np.random.randint(0, len(baseline_scenarios))
    base = baseline_scenarios[idx]
    
    solar = base[0] + np.random.normal(0, 0.1)
    wind = base[1] + np.random.normal(0, 0.1)
    load = base[2] + np.random.normal(0, 0.1)
    price = base[3]
    
    solar = np.clip(solar, 0, 3.0)
    wind = np.clip(wind, 0, 2.0)
    load = np.clip(load, 1.0, 3.0)
    
    augmented_scenarios.append([solar, wind, load, price])

augmented_scenarios = np.array(augmented_scenarios)
print(f"  Generated {len(augmented_scenarios)} scenarios")

print("\n[3/3] Training agent...")
agent = OptimizedDRLAgent(
    scenarios=augmented_scenarios,
    penalty_sharpness=6.0,
    margin_tightness=0.90
)

print("\n  Training with 300k timesteps...")
agent.train(total_timesteps=300000)

print("\n[4/4] Saving model...")
agent.save_model("models/trained_agent")

print("\n" + "="*70)
print("Training complete!")
print("="*70)
