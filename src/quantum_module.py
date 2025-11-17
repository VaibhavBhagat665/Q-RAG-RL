import numpy as np
import torch
import torch.nn as nn
try:
    import pennylane as qml
except Exception:
    qml = None

class QuantumScenarioGenerator:
    def __init__(self, n_qubits=4, n_scenarios=1000):
        self.n_qubits = n_qubits
        self.n_scenarios = n_scenarios
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
        
        for i in range(self.n_qubits):
            qml.RZ(params[i+self.n_qubits], wires=i)
            
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def generate_scenarios(self):
        scenarios = []
        if self.qnode is not None:
            for _ in range(self.n_scenarios):
                params = np.random.uniform(0, 2*np.pi, 2*self.n_qubits)
                measurements = self.qnode(params)
                solar = (measurements[0] + 1) * 1.5
                wind = (measurements[1] + 1) * 1.0
                load = 2 + (measurements[2] + 1) * 0.5
                price = 0.1 + (measurements[3] + 1) * 0.05
                scenarios.append([solar, wind, load, price])
            return np.array(scenarios)
        rng = np.random.default_rng(42)
        cov = np.array([[1.0, 0.6, -0.3, 0.2],
                        [0.6, 1.0, -0.2, 0.1],
                        [-0.3, -0.2, 1.0, -0.4],
                        [0.2, 0.1, -0.4, 1.0]])
        mean = np.array([0.0, 0.0, 0.0, 0.0])
        raw = rng.multivariate_normal(mean, cov, size=self.n_scenarios)
        raw = np.tanh(raw)
        solar = (raw[:, 0] + 1) * 1.5
        wind = (raw[:, 1] + 1) * 1.0
        load = 2 + (raw[:, 2] + 1) * 0.5
        price = 0.1 + (raw[:, 3] + 1) * 0.05
        return np.stack([solar, wind, load, price], axis=1)

class ClassicalGAN:
    def __init__(self, latent_dim=10, device=None):
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = self._build_generator().to(self.device).eval()

    def _build_generator(self):
        model = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        return model

    def generate_scenarios(self, n_scenarios=1000):
        with torch.no_grad():
            noise = torch.randn(n_scenarios, self.latent_dim, device=self.device)
            scenarios = self.generator(noise).cpu().numpy()

        scenarios[:, 0] *= 3.0
        scenarios[:, 1] *= 2.0
        scenarios[:, 2] = 2 + scenarios[:, 2] * 1.0
        scenarios[:, 3] = 0.1 + scenarios[:, 3] * 0.1
        
        return scenarios
