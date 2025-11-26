"""
Test model performance on industry datasets (NREL, OPSD, CAISO, EIA, London).
Evaluates cost and safety metrics across different data sources.
"""
import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.drl_agent import OptimizedMicrogridEnv

print("=" * 80)
print("TESTING ON INDUSTRY DATASETS")
print("=" * 80)

model_path = 'models/aggressive_optimized_agent.zip'
if not os.path.exists(model_path):
    model_path = 'models/trained_agent.zip'

print(f"\nLoading model: {model_path}")
model = PPO.load(model_path)

datasets = {
    'NREL_Solar': 'data/nrel_cache/solar_psm3_legacy_39.7391_-104.9847_2020.csv',
    'Industry_Standard': 'data/industry/industry_standard_dataset.csv',
    'OPSD_Germany': 'data/industry/opsd_germany_2020.csv',
    'London_Smart_Meter': 'data/industry/london_smart_meter.csv',
    'EIA_Grid': 'data/industry/eia_grid_data.csv',
    'CAISO': 'data/industry/caiso_grid_data.csv'
}

results_summary = []

def test_dataset(name, filepath):
    print(f"\n{'=' * 80}")
    print(f"Dataset: {name}")
    print(f"{'=' * 80}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        if name == 'NREL_Solar':
            df = pd.read_csv(filepath, skiprows=2)
        else:
            df = pd.read_csv(filepath)
        
        print(f"Loaded {len(df)} hours")
        
        test_hours = min(24, len(df))
        
        if name == 'NREL_Solar':
            solar_data = df['GHI'].values[:test_hours] / 300
            wind_data = df['Wind Speed'].values[:test_hours] / 5
            load_data = np.ones(test_hours) * 2.5
        elif name == 'Industry_Standard':
            solar_data = df['ghi'].values[:test_hours] / 300
            wind_data = df['wind_speed'].values[:test_hours] / 5
            load_data = df['load'].values[:test_hours]
        elif name == 'OPSD_Germany':
            cols = df.columns.tolist()
            solar_cols = [c for c in cols if 'solar' in c.lower() or 'DE_solar' in c]
            wind_cols = [c for c in cols if 'wind' in c.lower() or 'DE_wind' in c]
            load_cols = [c for c in cols if 'load' in c.lower() or 'DE_load' in c]
            
            solar_data = df[solar_cols[0]].values[:test_hours] / 10000 if solar_cols else np.zeros(test_hours)
            wind_data = df[wind_cols[0]].values[:test_hours] / 10000 if wind_cols else np.zeros(test_hours)
            load_data = df[load_cols[0]].values[:test_hours] / 10000 if load_cols else np.ones(test_hours) * 2.5
            
            solar_data = np.nan_to_num(solar_data, 0)
            wind_data = np.nan_to_num(wind_data, 0)
            load_data = np.nan_to_num(load_data, 2.5)
        elif name == 'London_Smart_Meter':
            solar_data = np.zeros(test_hours)
            wind_data = np.zeros(test_hours)
            load_data = df['load_kwh'].values[:test_hours]
        elif name == 'EIA_Grid':
            solar_data = df['solar_capacity_factor'].values[:test_hours] * 3.0
            wind_data = df['wind_capacity_factor'].values[:test_hours] * 2.0
            load_data = df['demand_mw'].values[:test_hours] / 20000
        elif name == 'CAISO':
            solar_data = df['solar_mw'].values[:test_hours] / 5000
            wind_data = df['wind_mw'].values[:test_hours] / 2000
            load_data = df['demand_mw'].values[:test_hours] / 10000
        else:
            print(f"Unknown dataset format")
            return None
        
        scenarios = []
        for i in range(test_hours):
            scenarios.append([solar_data[i], wind_data[i], load_data[i], 0.5])
        scenarios = np.array(scenarios)
        
        env = OptimizedMicrogridEnv(scenarios=scenarios)
        obs, _ = env.reset()
        
        total_cost = 0
        total_reward = 0
        violations = 0
        
        for hour in range(test_hours):
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            
            hour_cost = -reward if reward < 0 else 0
            total_cost += hour_cost
            total_reward += reward
            
            if env.safety_violations > violations:
                violations = env.safety_violations
            
            if done or truncated:
                obs, _ = env.reset()
                env.scenario_idx = min(hour + 1, len(scenarios) - 1)
        
        avg_cost = total_cost / test_hours
        safety_rate = (test_hours - violations) / test_hours * 100
        
        print(f"\nResults:")
        print(f"   Hours tested: {test_hours}")
        print(f"   Average cost: ${avg_cost:.4f}/hour")
        print(f"   Safety rate: {safety_rate:.1f}%")
        print(f"   Violations: {violations}/{test_hours}")
        
        if safety_rate >= 95 and avg_cost < 0.15:
            rating = "EXCELLENT"
        elif safety_rate >= 90 and avg_cost < 0.20:
            rating = "GOOD"
        elif safety_rate >= 85:
            rating = "ACCEPTABLE"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        print(f"   Performance: {rating}")
        
        return {
            'dataset': name,
            'hours': test_hours,
            'avg_cost': avg_cost,
            'safety_rate': safety_rate,
            'violations': violations,
            'rating': rating
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

for name, filepath in datasets.items():
    result = test_dataset(name, filepath)
    if result:
        results_summary.append(result)

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

if results_summary:
    summary_df = pd.DataFrame(results_summary)
    print(f"\n{summary_df.to_string(index=False)}")
    
    summary_df.to_csv('results/industry_test_results.csv', index=False)
    print(f"\nResults saved to: results/industry_test_results.csv")
    
    print(f"\nOverall Performance:")
    print(f"   Datasets tested: {len(results_summary)}")
    print(f"   Average cost: ${summary_df['avg_cost'].mean():.4f}/hour")
    print(f"   Average safety rate: {summary_df['safety_rate'].mean():.1f}%")
    
    print(f"\nTesting complete!")
else:
    print("\nNo datasets could be tested")

print(f"\n{'=' * 80}")
