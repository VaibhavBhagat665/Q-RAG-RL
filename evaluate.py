"""
Evaluate trained agent on real NREL weather data.
Tests model on 24 hours of actual solar/wind measurements.
"""
import pandas as pd
import numpy as np
from src.drl_agent import OptimizedDRLAgent
from src.nrel_data import NRELDataLoader

print("="*70)
print("Q-RAG-RL AGENT EVALUATION ON REAL NREL DATA")
print("="*70)

print("\n1. Loading REAL NREL Weather Data...")
nrel_loader = NRELDataLoader()

solar_df = nrel_loader.get_solar_data(lat=39.7391, lon=-104.9847, year=2020)
wind_df = nrel_loader.get_wind_data(lat=39.7391, lon=-104.9847, year=2020)

print(f"   Solar data: {len(solar_df)} hours")
print(f"   Wind data: {len(wind_df)} hours")

solar_power, wind_power = nrel_loader.convert_to_power(solar_df, wind_df)
load_profile = nrel_loader.get_load_profile()

solar_power_24h = solar_power[:24]
wind_power_24h = wind_power[:24]
load_24h = load_profile[:24]

prices = []
for hour in range(24):
    if 0 <= hour < 7:
        base_price = 0.12
    elif 17 <= hour < 22:
        base_price = 0.20
    else:
        base_price = 0.15
    prices.append(base_price)

print(f"\n2. Evaluating on 24 hours of REAL weather data")
print(f"   Solar range: {solar_power_24h.min():.3f} - {solar_power_24h.max():.3f} MW")
print(f"   Wind range: {wind_power_24h.min():.3f} - {wind_power_24h.max():.3f} MW")
print(f"   Load range: {load_24h.min():.3f} - {load_24h.max():.3f} MW")

print("\n3. Loading trained agent...")
agent = OptimizedDRLAgent()
try:
    agent.load_model("models/trained_agent")
    print("   Model loaded successfully")
except:
    print("   ERROR: Model not found. Run: python train.py")
    exit(1)

print("\n4. Running evaluation...")
print("="*70)

results = []
battery_soc = 0.5

for hour in range(24):
    solar = solar_power_24h[hour]
    wind = wind_power_24h[hour]
    load = load_24h[hour]
    price = prices[hour]
    
    obs = np.array([solar, wind, load, battery_soc, hour/24, price])
    
    action = agent.predict(obs)
    action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
    action_val = np.clip(action_val, -1.0, 1.0)
    
    current_state = {
        'battery_soc': battery_soc,
        'voltage': 1.0 + np.random.normal(0, 0.005),
        'frequency': 50.0 + np.random.normal(0, 0.05),
        'total_generation': solar + wind + action_val,
        'load': load
    }
    
    is_safe, penalty = agent.rag_safety.check_safety_graded(action_val, current_state)
    
    grid_power = load - solar - wind - action_val
    if grid_power > 0:
        cost = grid_power * price
    else:
        cost = grid_power * price * 0.1
    
    battery_soc = np.clip(battery_soc - (action_val * 1.0 / 5.0), 0.0, 1.0)
    
    results.append({
        'hour': hour,
        'solar_gen': solar,
        'wind_gen': wind,
        'load': load,
        'price': price,
        'battery_soc': battery_soc,
        'action': action_val,
        'cost': cost,
        'safe': is_safe,
        'penalty': penalty
    })
    
    status = "SAFE" if is_safe else "UNSAFE"
    print(f"H{hour:2d}: {status:6s} Solar={solar:5.2f} Wind={wind:5.2f} Load={load:5.2f} Cost=${cost:6.3f} SOC={battery_soc:.2f}")

df = pd.DataFrame(results)

baseline_cost = 0
for hour in range(24):
    grid_power = load_24h[hour] - solar_power_24h[hour] - wind_power_24h[hour]
    if grid_power > 0:
        baseline_cost += grid_power * prices[hour]
    else:
        baseline_cost += grid_power * prices[hour] * 0.1

print("\n" + "="*70)
print("RESULTS ON REAL NREL DATA")
print("="*70)

optimized_cost = df['cost'].sum()
violations = len(df[df['safe'] == False])

print(f"\nBaseline (no control): ${baseline_cost:.4f}")
print(f"With Q-RAG-RL:         ${optimized_cost:.4f}")

cost_reduction = ((baseline_cost - optimized_cost) / baseline_cost) * 100
savings = baseline_cost - optimized_cost

print(f"\nCost Reduction: {cost_reduction:.2f}%")
print(f"Absolute Savings: ${savings:.4f}")
print(f"Safety Violations: {violations}/24 hours")

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

if violations == 0:
    print(f"Safety: {violations}/24 violations [PASS]")
else:
    print(f"Safety: {violations}/24 violations [FAIL]")

if cost_reduction > 0:
    print(f"Cost Reduction: {cost_reduction:.2f}% [PASS]")
else:
    print(f"Cost Reduction: {cost_reduction:.2f}% [FAIL]")

print("\n" + "="*70)
if violations == 0 and cost_reduction > 0:
    print("EVALUATION SUCCESSFUL ON REAL DATA")
    print(f"  Zero violations, {cost_reduction:.1f}% cost reduction")
else:
    print("EVALUATION COMPLETE")
print("="*70)

df.to_csv('results/evaluation_real_data.csv', index=False)
print("\nResults saved to results/evaluation_real_data.csv")
print("\nData source: REAL NREL weather station measurements")
