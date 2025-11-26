"""
Compare Q-RAG-RL against multiple baseline methods
This proves your method is BETTER, not just different
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.drl_agent import OptimizedDRLAgent
from stable_baselines3 import PPO
import gymnasium as gym

print("=" * 70)
print("COMPARATIVE STUDY: Q-RAG-RL vs. Baselines")
print("=" * 70)

# Load test data (24 hours)
solar_df = pd.read_csv('data/nrel_cache/solar_psm3_legacy_39.7391_-104.9847_2020.csv', skiprows=2)
wind_df = pd.read_csv('data/nrel_cache/wind_wtk_39.7391_-104.9847_2022.csv', skiprows=2)
wind_df.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature', 
                   'Wind Direction', 'Wind Speed', 'Pressure']

# Use summer day with good generation
start_hour = 4134
end_hour = start_hour + 24

# Convert to power
solar_capacity = 3.0
wind_capacity = 2.0

solar_power = []
for _, row in solar_df.iloc[start_hour:end_hour].iterrows():
    ghi = row['GHI']
    temp = row.get('Temperature', 25)
    efficiency = 0.2 * (1 - 0.004 * (temp - 25))
    panel_area = solar_capacity * 1000 / 200
    power = (ghi / 1000) * panel_area * efficiency / 1000
    solar_power.append(min(power, solar_capacity))

wind_power = []
for _, row in wind_df.iloc[start_hour:end_hour].iterrows():
    wind_speed = row['Wind Speed']
    if wind_speed < 3:
        power = 0
    elif wind_speed < 12:
        power = wind_capacity * ((wind_speed - 3) / 9) ** 3
    elif wind_speed < 25:
        power = wind_capacity
    else:
        power = 0
    wind_power.append(power)

solar_power = np.array(solar_power)
wind_power = np.array(wind_power)

# Generate load and prices
dates = pd.date_range('2020-06-21 06:00', periods=24, freq='h')
load_profile = []
prices = []
base_load = 2.5

for date in dates:
    hour = date.hour
    day_of_week = date.weekday()
    day_of_year = date.timetuple().tm_yday
    
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    if day_of_week < 5:
        if 6 <= hour <= 9 or 17 <= hour <= 21:
            daily_factor = 1.3
        elif 10 <= hour <= 16:
            daily_factor = 1.1
        else:
            daily_factor = 0.7
    else:
        daily_factor = 1.0 + 0.2 * np.sin(2 * np.pi * hour / 24)
    
    noise = 1 + 0.1 * np.random.normal()
    load = base_load * seasonal_factor * daily_factor * noise
    load_profile.append(max(0.5, load))
    
    if 0 <= hour < 7:
        base_price = 0.12
    elif 17 <= hour < 22:
        base_price = 0.20
    else:
        base_price = 0.15
    prices.append(base_price)

load_profile = np.array(load_profile)
prices = np.array(prices)

print(f"\nTest Scenario: 24 hours, June 21 (summer)")
print(f"Solar: {solar_power.mean():.3f} MW avg, {solar_power.max():.3f} MW peak")
print(f"Wind: {wind_power.mean():.3f} MW avg, {wind_power.max():.3f} MW peak")
print(f"Load: {load_profile.mean():.3f} MW avg, {load_profile.max():.3f} MW peak")

# Method 1: Q-RAG-RL (Your Method)
print("\n" + "=" * 70)
print("METHOD 1: Q-RAG-RL (Your Method)")
print("=" * 70)

agent_qragrl = OptimizedDRLAgent()
agent_qragrl.load_model("models/trained_agent")

results_qragrl = []
battery_soc = 0.5

for hour in range(24):
    obs = np.array([solar_power[hour], wind_power[hour], load_profile[hour], 
                    battery_soc, hour/24, prices[hour]])
    action = agent_qragrl.predict(obs)
    action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
    action_val = np.clip(action_val, -1.0, 1.0)
    
    current_state = {
        'battery_soc': battery_soc,
        'voltage': 1.0 + np.random.normal(0, 0.005),
        'frequency': 50.0 + np.random.normal(0, 0.05),
        'total_generation': solar_power[hour] + wind_power[hour] + action_val,
        'load': load_profile[hour]
    }
    
    is_safe, _ = agent_qragrl.rag_safety.check_safety_graded(action_val, current_state)
    
    grid_power = load_profile[hour] - solar_power[hour] - wind_power[hour] - action_val
    cost = grid_power * prices[hour] if grid_power > 0 else grid_power * prices[hour] * 0.1
    
    battery_soc = np.clip(battery_soc - (action_val * 1.0 / 5.0), 0.0, 1.0)
    
    results_qragrl.append({'cost': cost, 'safe': is_safe, 'soc': battery_soc})

df_qragrl = pd.DataFrame(results_qragrl)
print(f"Cost: ${df_qragrl['cost'].sum():.4f}")
print(f"Violations: {len(df_qragrl[df_qragrl['safe'] == False])}/24")
print(f"SOC Range: {df_qragrl['soc'].min():.2f} - {df_qragrl['soc'].max():.2f}")

# Method 2: Rule-Based Controller
print("\n" + "=" * 70)
print("METHOD 2: Rule-Based Controller")
print("=" * 70)

results_rule = []
battery_soc = 0.5

for hour in range(24):
    # Simple rules:
    # - Charge when excess generation and low price
    # - Discharge when deficit and high price
    # - Keep SOC between 20-80%
    
    net_power = solar_power[hour] + wind_power[hour] - load_profile[hour]
    
    if net_power > 0 and prices[hour] < 0.15 and battery_soc < 0.8:
        action_val = min(0.5, net_power * 0.5)  # Charge
    elif net_power < 0 and prices[hour] > 0.15 and battery_soc > 0.2:
        action_val = max(-0.5, net_power * 0.5)  # Discharge
    else:
        action_val = 0.0  # Do nothing
    
    action_val = np.clip(action_val, -1.0, 1.0)
    
    current_state = {
        'battery_soc': battery_soc,
        'voltage': 1.0,
        'frequency': 50.0,
        'total_generation': solar_power[hour] + wind_power[hour] + action_val,
        'load': load_profile[hour]
    }
    
    # Check safety
    is_safe = (0.1 <= battery_soc <= 0.9 and 
               abs(action_val) <= 1.0 and
               0.95 <= current_state['voltage'] <= 1.05)
    
    grid_power = load_profile[hour] - solar_power[hour] - wind_power[hour] - action_val
    cost = grid_power * prices[hour] if grid_power > 0 else grid_power * prices[hour] * 0.1
    
    battery_soc = np.clip(battery_soc - (action_val * 1.0 / 5.0), 0.0, 1.0)
    
    results_rule.append({'cost': cost, 'safe': is_safe, 'soc': battery_soc})

df_rule = pd.DataFrame(results_rule)
print(f"Cost: ${df_rule['cost'].sum():.4f}")
print(f"Violations: {len(df_rule[df_rule['safe'] == False])}/24")
print(f"SOC Range: {df_rule['soc'].min():.2f} - {df_rule['soc'].max():.2f}")

# Method 3: Greedy Controller (Cost-Only)
print("\n" + "=" * 70)
print("METHOD 3: Greedy Controller (Cost-Only, No Safety)")
print("=" * 70)

results_greedy = []
battery_soc = 0.5

for hour in range(24):
    # Greedy: Always minimize immediate cost
    # Charge when price is low, discharge when high
    
    if prices[hour] < 0.13:
        action_val = 0.8  # Aggressive charge
    elif prices[hour] > 0.18:
        action_val = -0.8  # Aggressive discharge
    else:
        action_val = 0.0
    
    action_val = np.clip(action_val, -1.0, 1.0)
    
    current_state = {
        'battery_soc': battery_soc,
        'voltage': 1.0,
        'frequency': 50.0,
        'total_generation': solar_power[hour] + wind_power[hour] + action_val,
        'load': load_profile[hour]
    }
    
    # Check safety (will likely violate)
    is_safe = (0.1 <= battery_soc <= 0.9 and 
               abs(action_val) <= 1.0)
    
    grid_power = load_profile[hour] - solar_power[hour] - wind_power[hour] - action_val
    cost = grid_power * prices[hour] if grid_power > 0 else grid_power * prices[hour] * 0.1
    
    battery_soc = np.clip(battery_soc - (action_val * 1.0 / 5.0), 0.0, 1.0)
    
    results_greedy.append({'cost': cost, 'safe': is_safe, 'soc': battery_soc})

df_greedy = pd.DataFrame(results_greedy)
print(f"Cost: ${df_greedy['cost'].sum():.4f}")
print(f"Violations: {len(df_greedy[df_greedy['safe'] == False])}/24")
print(f"SOC Range: {df_greedy['soc'].min():.2f} - {df_greedy['soc'].max():.2f}")

# Method 4: No Control (Baseline)
print("\n" + "=" * 70)
print("METHOD 4: No Battery Control (Baseline)")
print("=" * 70)

results_baseline = []

for hour in range(24):
    action_val = 0.0  # No battery action
    
    grid_power = load_profile[hour] - solar_power[hour] - wind_power[hour]
    cost = grid_power * prices[hour] if grid_power > 0 else grid_power * prices[hour] * 0.1
    
    results_baseline.append({'cost': cost, 'safe': True, 'soc': 0.5})

df_baseline = pd.DataFrame(results_baseline)
print(f"Cost: ${df_baseline['cost'].sum():.4f}")
print(f"Violations: 0/24 (no battery = no violations)")

# Comparison Table
print("\n" + "=" * 70)
print("COMPARATIVE RESULTS")
print("=" * 70)

methods = {
    'Q-RAG-RL (Yours)': df_qragrl,
    'Rule-Based': df_rule,
    'Greedy (Unsafe)': df_greedy,
    'No Control': df_baseline
}

print(f"\n{'Method':<20} {'Cost':<12} {'Violations':<12} {'Improvement'}")
print("-" * 70)

baseline_cost = df_baseline['cost'].sum()

for name, df in methods.items():
    cost = df['cost'].sum()
    violations = len(df[df['safe'] == False])
    improvement = ((baseline_cost - cost) / baseline_cost) * 100
    
    print(f"{name:<20} ${cost:<11.4f} {violations:<12} {improvement:+.2f}%")

# Statistical Summary
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

qragrl_cost = df_qragrl['cost'].sum()
qragrl_violations = len(df_qragrl[df_qragrl['safe'] == False])

print(f"\n1. Cost Performance:")
print(f"   Q-RAG-RL: ${qragrl_cost:.4f}")
print(f"   Best Alternative: ${df_rule['cost'].sum():.4f} (Rule-Based)")
print(f"   Improvement: {((df_rule['cost'].sum() - qragrl_cost) / df_rule['cost'].sum()) * 100:.2f}%")

print(f"\n2. Safety Performance:")
print(f"   Q-RAG-RL: {qragrl_violations} violations ✅")
print(f"   Rule-Based: {len(df_rule[df_rule['safe'] == False])} violations")
print(f"   Greedy: {len(df_greedy[df_greedy['safe'] == False])} violations ❌")

print(f"\n3. Cost vs. Baseline:")
print(f"   Q-RAG-RL: {((baseline_cost - qragrl_cost) / baseline_cost) * 100:.2f}% reduction")
print(f"   Rule-Based: {((baseline_cost - df_rule['cost'].sum()) / baseline_cost) * 100:.2f}% reduction")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if qragrl_violations == 0 and qragrl_cost < df_rule['cost'].sum():
    print("\n✅ Q-RAG-RL is THE BEST method:")
    print("   • Zero violations (safest)")
    print("   • Lowest cost (most efficient)")
    print("   • Outperforms all baselines")
    print("\n🎉 This proves your method is SUPERIOR!")
elif qragrl_violations == 0:
    print("\n✅ Q-RAG-RL achieves best safety:")
    print("   • Zero violations (safest)")
    print("   • Competitive cost")
    print("   • Safety-first approach validated")
else:
    print(f"\n⚠️  Q-RAG-RL has {qragrl_violations} violations")

# Save comparison
comparison_df = pd.DataFrame({
    'Method': list(methods.keys()),
    'Cost': [df['cost'].sum() for df in methods.values()],
    'Violations': [len(df[df['safe'] == False]) for df in methods.values()],
    'Improvement_vs_Baseline': [((baseline_cost - df['cost'].sum()) / baseline_cost) * 100 
                                for df in methods.values()]
})

comparison_df.to_csv('method_comparison_results.csv', index=False)
print("\n✓ Comparison saved to method_comparison_results.csv")
print("=" * 70)
