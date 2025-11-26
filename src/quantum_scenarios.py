import numpy as np
import torch
import torch.nn as nn
try:
    import pennylane as qml
except Exception:
    qml = None

class QuantumScenarioGenerator:
    def __init__(self, n_qubits=6, n_scenarios=1000, stress_factor=1.5):
        self.n_qubits = n_qubits
        self.n_scenarios = n_scenarios
        self.stress_factor = stress_factor
        if qml is not None:
            try:
                self.dev = qml.device("lightning.qubit", wires=n_qubits)
                self.qnode = qml.QNode(self.quantum_circuit, self.dev)
            except Exception:
                self.dev = None
                self.qnode = None
        else:
            self.dev = None
            self.qnode = None
        
    def quantum_circuit(self, params):
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        
        for i in range(self.n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[self.n_qubits-1, 0])
        
        for i in range(self.n_qubits):
            qml.RZ(params[i+self.n_qubits], wires=i)
        
        for i in range(0, self.n_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
            
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def generate_boundary_scenarios(self):
        scenarios = []
        rng = np.random.default_rng(42)
        
        cov = np.array([
            [1.0, 0.75, -0.65, 0.45, -0.3, 0.2],
            [0.75, 1.0, -0.55, 0.35, -0.25, 0.15],
            [-0.65, -0.55, 1.0, -0.7, 0.4, -0.3],
            [0.45, 0.35, -0.7, 1.0, -0.5, 0.4],
            [-0.3, -0.25, 0.4, -0.5, 1.0, -0.6],
            [0.2, 0.15, -0.3, 0.4, -0.6, 1.0]
        ])
        mean = np.zeros(6)
        raw = rng.multivariate_normal(mean, cov, size=self.n_scenarios)
        raw = np.tanh(raw * self.stress_factor)
        
        solar = (raw[:, 0] + 1) * 1.5
        wind = (raw[:, 1] + 1) * 1.0
        load = 2 + (raw[:, 2] + 1) * 0.6
        price = 0.1 + (raw[:, 3] + 1) * 0.06
        
        solar_volatility = np.abs(raw[:, 4])
        load_spike = np.abs(raw[:, 5])
        
        for i in range(len(solar)):
            if solar_volatility[i] > 0.7:
                solar[i] *= 0.3
            if load_spike[i] > 0.7:
                load[i] *= 1.4
        
        return np.stack([solar, wind, load, price], axis=1)
    
    def generate_stress_scenarios(self):
        scenarios = []
        
        for _ in range(self.n_scenarios // 3):
            scenarios.append([0.15, 0.2, 2.8, 0.19])
        
        for _ in range(self.n_scenarios // 3):
            scenarios.append([2.9, 1.9, 0.8, 0.11])
        
        for _ in range(self.n_scenarios - 2 * (self.n_scenarios // 3)):
            solar = np.random.uniform(0.1, 3.0)
            wind = np.random.uniform(0.1, 2.0)
            load = np.random.uniform(1.5, 3.0)
            price = np.random.uniform(0.1, 0.2)
            scenarios.append([solar, wind, load, price])
        
        return np.array(scenarios)

class EnhancedGAN:
    def __init__(self, latent_dim=16, device=None):
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = self._build_generator().to(self.device).eval()

    def _build_generator(self):
        model = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        return model

    def generate_scenarios(self, n_scenarios=1000, boundary_bias=0.3):
        with torch.no_grad():
            noise = torch.randn(n_scenarios, self.latent_dim, device=self.device)
            
            boundary_mask = torch.rand(n_scenarios, device=self.device) < boundary_bias
            noise[boundary_mask] *= 2.5
            
            scenarios = self.generator(noise).cpu().numpy()

        scenarios[:, 0] *= 3.0
        scenarios[:, 1] *= 2.0
        scenarios[:, 2] = 2 + scenarios[:, 2] * 1.2
        scenarios[:, 3] = 0.1 + scenarios[:, 3] * 0.1
        
        return scenarios
