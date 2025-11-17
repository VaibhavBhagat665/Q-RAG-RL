# Q-RAG-RL Microgrid Management System

Advanced microgrid optimization combining Quantum Computing, Retrieval-Augmented Generation (RAG), and Deep Reinforcement Learning.

## System Architecture

### 1. Quantum Module (PennyLane)
- **Quantum Scenario Generator**: Uses 4-qubit circuits to generate correlated solar/wind/load scenarios
- **Classical GAN Baseline**: TensorFlow-based GAN for comparison
- **Output**: 1000 24-hour ahead scenarios for training

### 2. RAG Safety Module (LangChain + ChromaDB)
- **Vector Database**: Embedded safety constraints using sentence-transformers
- **Real-time Safety Check**: `check_safety(action, state) → (bool, penalty)`
- **Knowledge Base**: Voltage limits, battery SOC, frequency, power constraints

### 3. DRL Agent (Stable-Baselines3 PPO)
- **Environment**: 3MW solar, 2MW wind, 5MWh battery microgrid
- **State**: [solar_gen, wind_gen, load, battery_soc, time, price]
- **Action**: Battery charge/discharge rate (-1 to +1 MW)
- **Reward**: -operational_cost - safety_penalty

### 4. Digital Twin (Pandapower)
- **IEEE 33-Bus System**: Complete distribution network model
- **Power Flow**: Real-time voltage stability analysis
- **Metrics**: Voltage violations, losses, line loadings

### 5. NREL Data Integration
- **Solar Data**: NSRDB API with synthetic fallback
- **Wind Data**: Wind Toolkit with Weibull distribution
- **Load Profiles**: Realistic residential/commercial patterns

### 6. Streamlit Dashboard
- Real-time monitoring and control
- Quantum vs classical comparison
- Safety constraint visualization
- Power flow analysis

## Quick Start

### Installation
```bash
git clone <repository>
cd qragrl
pip install -r requirements.txt
```

### Run Options

#### Full Simulation
```bash
python main.py --mode full
```

#### Component Testing
```bash
python main.py --mode test
```

#### Interactive Dashboard
```bash
python main.py --mode dashboard
```

## Project Structure

```
qragrl/
├── src/
│   ├── quantum_module.py     # PennyLane quantum circuits
│   ├── rag_safety.py        # ChromaDB safety constraints
│   ├── drl_agent.py         # PPO reinforcement learning
│   ├── digital_twin.py      # Pandapower IEEE 33-bus
│   ├── nrel_data.py         # Weather data integration
│   └── dashboard.py         # Streamlit interface
├── data/                    # Generated datasets
├── main.py                  # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Key Features

### Quantum Enhancement
- **Correlated Scenarios**: Quantum entanglement models renewable correlations
- **Variational Circuits**: Parameterized quantum circuits for scenario diversity
- **Performance Comparison**: Direct quantum vs classical benchmarking

### Safety-First Design
- **Vector-based Constraints**: Semantic search for relevant safety rules
- **Real-time Validation**: Every action checked against safety database
- **Penalty System**: Graduated penalties for constraint violations

### Realistic Simulation
- **IEEE Standard**: Industry-standard 33-bus test system
- **Real Weather Data**: NREL database integration with fallbacks
- **Power Flow Analysis**: Full AC power flow with voltage stability

### Interactive Monitoring
- **Live Dashboard**: Real-time system state visualization
- **Performance Metrics**: Cost, safety, technical performance tracking
- **Control Interface**: Manual override and parameter adjustment

## Technical Specifications

### Quantum Circuit
- 4 qubits with RY-CNOT-RZ layers
- Measurement in Pauli-Z basis
- Parameter optimization via gradient descent

### Safety Constraints
- Battery SOC: 10-90%
- Grid voltage: 0.95-1.05 pu
- Frequency: 49.5-50.5 Hz
- Power limits: ±1 MW battery

### System Capacities
- Solar: 3 MW distributed across buses 5, 10, 15
- Wind: 2 MW distributed across buses 8, 12
- Battery: 5 MWh at bus 1 with ±1 MW power rating

### Performance Targets
- Cost minimization through optimal battery scheduling
- Voltage stability maintenance across all buses
- 99%+ safety constraint compliance
- Real-time operation with <1s response time

## Results and Metrics

The system generates comprehensive performance reports including:

- **Economic Performance**: $/hour operational cost, peak shaving effectiveness
- **Technical Performance**: Voltage stability, power quality, system losses
- **Safety Performance**: Constraint violation frequency and severity
- **Quantum Advantage**: Scenario quality comparison metrics

## Dependencies

Core libraries:
- `pandapower==2.13.1` - Power system analysis
- `pennylane==0.32.0` - Quantum computing
- `stable-baselines3==2.1.0` - Reinforcement learning
- `chromadb==0.4.18` - Vector database
- `streamlit==1.29.0` - Web dashboard

## License

MIT License - Academic and commercial use permitted

## Citation

```
@software{qragrl2025,
  title={Q-RAG-RL: Quantum-Enhanced Microgrid Management},
  author={Vaibhav Bhagat},
  year={2025},
  url={https://github.com/VaibhavBhagat665/Q-RAG-RL}
}
```
