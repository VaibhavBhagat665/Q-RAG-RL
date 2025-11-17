import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class MicrogridDigitalTwin:
    def __init__(self, source: str = "custom"):
        if source == "case33bw":
            try:
                self.net = pn.case33bw()
            except Exception:
                self.net = self._create_ieee33_microgrid()
        else:
            self.net = self._create_ieee33_microgrid()
        self.battery_bus = 1
        self.solar_buses = [5, 10, 15]
        self.wind_buses = [8, 12]
        self.storage_elements = []
        self.renewable_elements = []
        self._add_renewable_and_storage()
        
    def _create_ieee33_microgrid(self):
        net = pp.create_empty_network()
        
        buses_data = [
            (0, 12.66, "slack"),
            (1, 12.66, "PQ"), (2, 12.66, "PQ"), (3, 12.66, "PQ"), (4, 12.66, "PQ"), (5, 12.66, "PQ"),
            (6, 12.66, "PQ"), (7, 12.66, "PQ"), (8, 12.66, "PQ"), (9, 12.66, "PQ"), (10, 12.66, "PQ"),
            (11, 12.66, "PQ"), (12, 12.66, "PQ"), (13, 12.66, "PQ"), (14, 12.66, "PQ"), (15, 12.66, "PQ"),
            (16, 12.66, "PQ"), (17, 12.66, "PQ"), (18, 12.66, "PQ"), (19, 12.66, "PQ"), (20, 12.66, "PQ"),
            (21, 12.66, "PQ"), (22, 12.66, "PQ"), (23, 12.66, "PQ"), (24, 12.66, "PQ"), (25, 12.66, "PQ"),
            (26, 12.66, "PQ"), (27, 12.66, "PQ"), (28, 12.66, "PQ"), (29, 12.66, "PQ"), (30, 12.66, "PQ"),
            (31, 12.66, "PQ"), (32, 12.66, "PQ")
        ]
        
        for bus_id, vn_kv, bus_type in buses_data:
            pp.create_bus(net, vn_kv=vn_kv, name=f"Bus_{bus_id}")
        
        lines_data = [
            (0, 1, 0.0922, 0.0477, 0.5), (1, 2, 0.493, 0.2511, 0.366),
            (2, 3, 0.366, 0.1864, 0.38), (3, 4, 0.3811, 0.1941, 0.819),
            (4, 5, 0.819, 0.707, 0.1872), (5, 6, 0.1872, 0.6188, 0.7114),
            (6, 7, 0.7114, 0.2351, 1.03), (7, 8, 1.03, 0.74, 1.044),
            (8, 9, 1.044, 0.74, 0.1966), (9, 10, 0.1966, 0.065, 0.3744),
            (10, 11, 0.3744, 0.1238, 1.468), (11, 12, 1.468, 1.155, 0.5416),
            (12, 13, 0.5416, 0.7129, 0.591), (13, 14, 0.591, 0.526, 0.7463),
            (14, 15, 0.7463, 0.545, 1.289), (15, 16, 1.289, 1.721, 0.732),
            (16, 17, 0.732, 0.574, 0.164), (17, 18, 0.164, 0.1565, 1.5042),
            (1, 19, 1.5042, 1.3554, 0.4095), (19, 20, 0.4095, 0.4784, 0.7089),
            (20, 21, 0.7089, 0.9373, 0.4512), (21, 22, 0.4512, 0.3083, 0.898),
            (3, 23, 0.898, 0.7091, 0.896), (23, 24, 0.896, 0.7011, 0.203),
            (5, 25, 0.203, 0.1034, 0.2842), (25, 26, 0.2842, 0.1447, 1.059),
            (26, 27, 1.059, 0.9337, 0.8042), (27, 28, 0.8042, 0.7006, 0.5075),
            (28, 29, 0.5075, 0.2585, 0.9744), (29, 30, 0.9744, 0.963, 0.3105),
            (30, 31, 0.3105, 0.3619, 0.341), (31, 32, 0.341, 0.5302, 0)
        ]
        
        for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in lines_data:
            pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, 
                                         length_km=length_km, r_ohm_per_km=r_ohm_per_km, 
                                         x_ohm_per_km=x_ohm_per_km, c_nf_per_km=10, 
                                         max_i_ka=0.4)
        
        load_data = [
            (1, 100, 60), (2, 90, 40), (3, 120, 80), (4, 60, 30), (5, 60, 20),
            (6, 200, 100), (7, 200, 100), (8, 60, 20), (9, 60, 20), (10, 45, 30),
            (11, 60, 35), (12, 60, 35), (13, 120, 80), (14, 60, 10), (15, 60, 20),
            (16, 60, 20), (17, 90, 40), (18, 90, 40), (19, 90, 40), (20, 90, 40),
            (21, 90, 40), (22, 90, 50), (23, 420, 200), (24, 420, 200), (25, 60, 25),
            (26, 60, 25), (27, 60, 20), (28, 120, 70), (29, 200, 600), (30, 150, 70),
            (31, 210, 100), (32, 60, 40)
        ]
        
        for bus, p_mw, q_mvar in load_data:
            pp.create_load(net, bus=bus, p_mw=p_mw/1000, q_mvar=q_mvar/1000)
        
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)
        
        return net
    
    def _add_renewable_and_storage(self):
        for bus in self.solar_buses:
            sgen = pp.create_sgen(net=self.net, bus=bus, p_mw=1.0, q_mvar=0.0, 
                                name=f"Solar_Bus_{bus}", type="PV")
            self.renewable_elements.append(('solar', sgen, bus))
        
        for bus in self.wind_buses:
            sgen = pp.create_sgen(net=self.net, bus=bus, p_mw=1.0, q_mvar=0.0, 
                                name=f"Wind_Bus_{bus}", type="WP")
            self.renewable_elements.append(('wind', sgen, bus))
        
        storage = pp.create_storage(net=self.net, bus=self.battery_bus, p_mw=0.0, 
                                  max_e_mwh=5.0, soc_percent=50.0, 
                                  min_e_mwh=0.5, max_p_mw=1.0, min_p_mw=-1.0)
        self.storage_elements.append(storage)
    
    def update_renewable_generation(self, solar_gen: List[float], wind_gen: List[float]):
        solar_idx = 0
        wind_idx = 0
        
        for gen_type, sgen_idx, bus in self.renewable_elements:
            if gen_type == 'solar' and solar_idx < len(solar_gen):
                self.net.sgen.at[sgen_idx, 'p_mw'] = solar_gen[solar_idx]
                solar_idx += 1
            elif gen_type == 'wind' and wind_idx < len(wind_gen):
                self.net.sgen.at[sgen_idx, 'p_mw'] = wind_gen[wind_idx]
                wind_idx += 1
    
    def update_battery_action(self, action: float):
        if len(self.storage_elements) > 0:
            storage_idx = self.storage_elements[0]
            self.net.storage.at[storage_idx, 'p_mw'] = action
    
    def run_powerflow(self) -> Tuple[bool, Dict]:
        try:
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            
            results = {
                'converged': True,
                'voltages': self.net.res_bus.vm_pu.values.tolist(),
                'voltage_angles': self.net.res_bus.va_degree.values.tolist(),
                'line_loadings': self.net.res_line.loading_percent.values.tolist(),
                'total_losses': self.net.res_line.pl_mw.sum(),
                'min_voltage': self.net.res_bus.vm_pu.min(),
                'max_voltage': self.net.res_bus.vm_pu.max(),
                'voltage_violations': len(self.net.res_bus[(self.net.res_bus.vm_pu < 0.95) | 
                                                         (self.net.res_bus.vm_pu > 1.05)]),
                'battery_soc': self.net.res_storage.soc_percent.iloc[0] if len(self.storage_elements) > 0 else 50.0
            }
            
            return True, results
            
        except Exception as e:
            return False, {'error': str(e), 'converged': False}
    
    def get_system_metrics(self) -> Dict:
        if hasattr(self.net, 'res_bus'):
            return {
                'avg_voltage': self.net.res_bus.vm_pu.mean(),
                'voltage_std': self.net.res_bus.vm_pu.std(),
                'total_generation': self.net.res_sgen.p_mw.sum() if hasattr(self.net, 'res_sgen') else 0,
                'total_load': self.net.res_load.p_mw.sum() if hasattr(self.net, 'res_load') else 0,
                'system_losses': self.net.res_line.pl_mw.sum() if hasattr(self.net, 'res_line') else 0,
                'max_line_loading': self.net.res_line.loading_percent.max() if hasattr(self.net, 'res_line') else 0
            }
        else:
            return {}
    
    def simulate_timestep(self, solar_gen: List[float], wind_gen: List[float], 
                         battery_action: float) -> Tuple[bool, Dict]:
        self.update_renewable_generation(solar_gen, wind_gen)
        self.update_battery_action(battery_action)
        
        converged, results = self.run_powerflow()
        
        if converged:
            metrics = self.get_system_metrics()
            results.update(metrics)
        
        return converged, results
