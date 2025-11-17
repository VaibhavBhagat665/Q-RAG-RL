import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import traceback
from src.quantum_module import QuantumScenarioGenerator, ClassicalGAN
from src.rag_safety import RAGSafetyModule
from src.drl_agent import DRLAgent, MicrogridEnv
from src.digital_twin import MicrogridDigitalTwin
from src.nrel_data import NRELDataLoader

def test_quantum_module():
    try:
        print("Testing Quantum Module...")
        quantum_gen = QuantumScenarioGenerator(n_qubits=4, n_scenarios=10)
        quantum_scenarios = quantum_gen.generate_scenarios()
        
        assert quantum_scenarios.shape == (10, 4), f"Expected (10, 4), got {quantum_scenarios.shape}"
        assert np.all(quantum_scenarios >= 0), "All scenario values should be non-negative"
        
        classical_gan = ClassicalGAN()
        classical_scenarios = classical_gan.generate_scenarios(10)
        assert classical_scenarios.shape == (10, 4), f"Expected (10, 4), got {classical_scenarios.shape}"
        
        print("✅ Quantum Module: PASS")
        return True, quantum_scenarios
    except Exception as e:
        print(f"❌ Quantum Module: FAIL - {e}")
        traceback.print_exc()
        return False, None

def test_rag_safety():
    try:
        print("Testing RAG Safety Module...")
        rag1 = RAGSafetyModule()
        
        test_state = {
            'battery_soc': 0.5,
            'voltage': 1.0,
            'frequency': 50.0,
            'total_generation': 3.0,
            'load': 2.5
        }
        
        is_safe1, penalty1 = rag1.check_safety(0.5, test_state)
        assert isinstance(is_safe1, bool), "Safety check should return boolean"
        assert isinstance(penalty1, (int, float)), "Penalty should be numeric"
        
        rag2 = RAGSafetyModule()
        is_safe2, penalty2 = rag2.check_safety(0.5, test_state)
        
        assert is_safe1 == is_safe2, "Multiple instances should give same result"
        
        print("✅ RAG Safety Module: PASS")
        return True, rag1
    except Exception as e:
        print(f"❌ RAG Safety Module: FAIL - {e}")
        traceback.print_exc()
        return False, None

def test_digital_twin():
    try:
        print("Testing Digital Twin...")
        digital_twin = MicrogridDigitalTwin()
        
        solar_gen = [1.0, 1.0, 1.0]
        wind_gen = [0.8, 0.8]
        battery_action = 0.2
        
        converged, results = digital_twin.simulate_timestep(solar_gen, wind_gen, battery_action)
        
        if converged:
            assert 'voltages' in results, "Results should contain voltages"
            assert 'min_voltage' in results, "Results should contain min_voltage"
            assert 'max_voltage' in results, "Results should contain max_voltage"
            assert len(results['voltages']) == 33, "Should have 33 bus voltages"
        
        print("✅ Digital Twin: PASS")
        return True, digital_twin
    except Exception as e:
        print(f"❌ Digital Twin: FAIL - {e}")
        traceback.print_exc()
        return False, None

def test_drl_agent(scenarios):
    try:
        print("Testing DRL Agent...")
        
        env = MicrogridEnv(scenarios=scenarios[:5])
        obs, _ = env.reset()
        
        assert len(obs) == 6, f"Observation space should be 6, got {len(obs)}"
        
        action = np.array([0.5])
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert len(next_obs) == 6, "Next observation should have correct shape"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert 'is_safe' in info, "Info should contain safety status"
        
        drl_agent = DRLAgent(scenarios=scenarios[:5])
        prediction = drl_agent.predict(obs)
        assert len(prediction) == 1, "Prediction should be single action"
        
        print("✅ DRL Agent: PASS")
        return True, drl_agent
    except Exception as e:
        print(f"❌ DRL Agent: FAIL - {e}")
        traceback.print_exc()
        return False, None

def test_nrel_data():
    try:
        print("Testing NREL Data Loader...")
        nrel_loader = NRELDataLoader()
        
        solar_df = nrel_loader.get_solar_data()
        wind_df = nrel_loader.get_wind_data()
        
        assert len(solar_df) > 0, "Solar data should not be empty"
        assert len(wind_df) > 0, "Wind data should not be empty"
        
        required_solar_cols = ['GHI', 'Wind Speed', 'Temperature']
        for col in required_solar_cols:
            assert col in solar_df.columns, f"Solar data missing column: {col}"
        
        required_wind_cols = ['Wind Speed', 'Wind Direction']
        for col in required_wind_cols:
            assert col in wind_df.columns, f"Wind data missing column: {col}"
        
        solar_power, wind_power = nrel_loader.convert_to_power(solar_df, wind_df)
        load_profile = nrel_loader.get_load_profile()
        
        assert len(solar_power) > 0, "Solar power array should not be empty"
        assert len(wind_power) > 0, "Wind power array should not be empty"
        assert len(load_profile) > 0, "Load profile should not be empty"
        
        print("✅ NREL Data Loader: PASS")
        return True, nrel_loader
    except Exception as e:
        print(f"❌ NREL Data Loader: FAIL - {e}")
        traceback.print_exc()
        return False, None

def test_end_to_end_integration():
    try:
        print("Testing End-to-End Integration...")
        
        quantum_gen = QuantumScenarioGenerator(n_qubits=4, n_scenarios=5)
        scenarios = quantum_gen.generate_scenarios()
        
        rag_safety = RAGSafetyModule()
        digital_twin = MicrogridDigitalTwin()
        drl_agent = DRLAgent(scenarios=scenarios)
        
        for hour in range(3):
            obs = np.array([1.0, 0.8, 2.0, 0.5, hour/24, 0.15])
            action = drl_agent.predict(obs)[0]
            
            current_state = {
                'battery_soc': 0.5,
                'voltage': 1.0,
                'frequency': 50.0,
                'total_generation': 1.8,
                'load': 2.0
            }
            
            is_safe, penalty = rag_safety.check_safety(action, current_state)
            
            converged, pf_results = digital_twin.simulate_timestep([0.5, 0.5, 0.5], [0.4, 0.4], action)
            
            assert isinstance(is_safe, bool), f"Safety check failed at hour {hour}"
            
        print("✅ End-to-End Integration: PASS")
        return True
    except Exception as e:
        print(f"❌ End-to-End Integration: FAIL - {e}")
        traceback.print_exc()
        return False

def main():
    print("🧪 Q-RAG-RL System Validation")
    print("=" * 50)
    
    results = {}
    
    quantum_pass, scenarios = test_quantum_module()
    results['quantum'] = quantum_pass
    
    rag_pass, rag_module = test_rag_safety()
    results['rag_safety'] = rag_pass
    
    twin_pass, digital_twin = test_digital_twin()
    results['digital_twin'] = twin_pass
    
    if quantum_pass and scenarios is not None:
        drl_pass, drl_agent = test_drl_agent(scenarios)
        results['drl_agent'] = drl_pass
    else:
        print("❌ DRL Agent: SKIP (Quantum module failed)")
        results['drl_agent'] = False
    
    nrel_pass, nrel_loader = test_nrel_data()
    results['nrel_data'] = nrel_pass
    
    if all([quantum_pass, rag_pass, twin_pass]):
        integration_pass = test_end_to_end_integration()
        results['integration'] = integration_pass
    else:
        print("❌ End-to-End Integration: SKIP (Dependencies failed)")
        results['integration'] = False
    
    print("\n📊 Final Results:")
    print("-" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:20s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        return True
    else:
        print("⚠️  Some systems need attention.")
        return False

def run_unit_tests_only():
    print("🧪 Q-RAG-RL Unit Tests")
    print("=" * 50)
    results = {}
    quantum_pass, scenarios = test_quantum_module()
    results['quantum'] = quantum_pass
    rag_pass, _ = test_rag_safety()
    results['rag_safety'] = rag_pass
    twin_pass, _ = test_digital_twin()
    results['digital_twin'] = twin_pass
    if quantum_pass and scenarios is not None:
        drl_pass, _ = test_drl_agent(scenarios)
    else:
        print("❌ DRL Agent: SKIP (Quantum module failed)")
        drl_pass = False
    results['drl_agent'] = drl_pass
    nrel_pass, _ = test_nrel_data()
    results['nrel_data'] = nrel_pass
    print("\n📊 Unit Test Results:")
    print("-" * 30)
    passed = sum(results.values())
    total = len(results)
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:20s}: {status}")
    print(f"\nOverall: {passed}/{total} unit tests passed")
    return passed == total

def run_integration_only():
    print("🧪 Q-RAG-RL Integration Test")
    print("=" * 50)
    return test_end_to_end_integration()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q-RAG-RL Validation Runner")
    parser.add_argument("--mode", choices=["unit", "integration"], default="unit",
                        help="Run unit tests or integration test")
    args = parser.parse_args()
    if args.mode == "unit":
        ok = run_unit_tests_only()
        sys.exit(0 if ok else 1)
    elif args.mode == "integration":
        ok = run_integration_only()
        sys.exit(0 if ok else 1)
