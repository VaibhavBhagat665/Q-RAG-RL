import numpy as np
from src.drl_agent import OptimizedDRLAgent, OptimizedMicrogridEnv
from src.quantum_scenarios import QuantumScenarioGenerator

def create_stress_scenario_1():
    scenarios = []
    for _ in range(100):
        solar = np.random.uniform(2.2, 2.6)
        wind = np.random.uniform(0.3, 0.6)
        load = np.random.uniform(1.8, 2.2)
        price = np.random.uniform(0.18, 0.20)
        scenarios.append([solar, wind, load, price])
    return np.array(scenarios)

def create_stress_scenario_2():
    scenarios = []
    for _ in range(100):
        solar = np.random.uniform(0.2, 0.5)
        wind = np.random.uniform(0.2, 0.5)
        load = np.random.uniform(2.2, 2.6)
        price = np.random.uniform(0.18, 0.20)
        scenarios.append([solar, wind, load, price])
    return np.array(scenarios)

def create_stress_scenario_3():
    scenarios = []
    for i in range(100):
        if i < 20:
            solar = np.random.uniform(2.0, 2.5)
            wind = np.random.uniform(0.8, 1.2)
            load = np.random.uniform(1.5, 1.9)
        elif i < 40:
            solar = np.random.uniform(0.3, 0.6)
            wind = np.random.uniform(0.3, 0.6)
            load = np.random.uniform(2.2, 2.6)
        else:
            solar = np.random.uniform(0.5, 2.5)
            wind = np.random.uniform(0.5, 1.5)
            load = np.random.uniform(1.5, 2.5)
        price = np.random.uniform(0.15, 0.20)
        scenarios.append([solar, wind, load, price])
    return np.array(scenarios)

def test_scenario(agent, scenarios, scenario_name):
    env = OptimizedMicrogridEnv(scenarios=scenarios, rag_safety=agent.rag_safety, 
                                cmdp_optimizer=agent.cmdp_optimizer)
    
    total_cost = 0
    total_violations = 0
    episodes = len(scenarios)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_cost = 0
        ep_violations = 0
        
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_cost += info['cost']
            if not info['is_safe']:
                ep_violations += 1
        
        total_cost += ep_cost
        total_violations += ep_violations
    
    avg_cost = total_cost / episodes
    avg_violations = total_violations / episodes
    
    print(f"\n{scenario_name}:")
    print(f"  Average Cost: ${avg_cost:.4f}")
    print(f"  Average Violations: {avg_violations:.2f}")
    print(f"  Total Violations: {total_violations}")
    
    return avg_cost, total_violations

if __name__ == "__main__":
    print("Loading trained agent...")
    agent = OptimizedDRLAgent()
    
    try:
        agent.load_model("models/trained_agent")
        print("Model loaded successfully")
    except Exception as e:
        print(f"No trained model found: {e}")
        print("Train first using: python train_agent.py")
        exit(1)
    
    print("\n" + "="*60)
    print("SCENARIO TESTING SUITE")
    print("="*60)
    
    print("\nScenario 1: High Generation + Moderate Load")
    print("  Condition: PV 2.2-2.6MW, Wind 0.3-0.6MW, Load 1.8-2.2MW")
    print("  Challenge: Surplus ~1MW, battery absorption + grid export")
    s1_scenarios = create_stress_scenario_1()
    s1_cost, s1_viol = test_scenario(agent, s1_scenarios, "Scenario 1 Results")
    
    print("\n" + "-"*60)
    print("\nScenario 2: Low Generation + High Load")
    print("  Condition: PV 0.2-0.5MW, Wind 0.2-0.5MW, Load 2.2-2.6MW")
    print("  Challenge: Deficit ~1.8MW, battery discharge + grid import")
    s2_scenarios = create_stress_scenario_2()
    s2_cost, s2_viol = test_scenario(agent, s2_scenarios, "Scenario 2 Results")
    
    print("\n" + "-"*60)
    print("\nScenario 3: Dynamic Transitions")
    print("  Condition: 20 steps surplus, 20 steps deficit, 60 mixed")
    print("  Challenge: Battery cycling + SOC management")
    s3_scenarios = create_stress_scenario_3()
    s3_cost, s3_viol = test_scenario(agent, s3_scenarios, "Scenario 3 Results")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_violations = s1_viol + s2_viol + s3_viol
    avg_cost = (s1_cost + s2_cost + s3_cost) / 3
    
    print(f"Total Violations: {total_violations}")
    print(f"Average Cost: ${avg_cost:.4f}")
    
    if total_violations == 0:
        print("\n✓ All scenarios passed")
    else:
        print(f"\n⚠ {total_violations} violations detected")
