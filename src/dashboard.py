import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum_module import QuantumScenarioGenerator, ClassicalGAN
from src.rag_safety import RAGSafetyModule
from src.drl_agent import DRLAgent
from src.digital_twin import MicrogridDigitalTwin
from src.nrel_data import NRELDataLoader

class MicrogridDashboard:
    def __init__(self):
        self.quantum_gen = QuantumScenarioGenerator()
        self.classical_gan = ClassicalGAN()
        self.rag_safety = RAGSafetyModule()
        self.digital_twin = MicrogridDigitalTwin()
        self.nrel_loader = NRELDataLoader()
        
        if 'drl_agent' not in st.session_state:
            st.session_state.drl_agent = None
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = []
        if 'real_time_running' not in st.session_state:
            st.session_state.real_time_running = False
    
    def run(self):
        st.set_page_config(
            page_title="Q-RAG-RL Microgrid Management",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔋 Q-RAG-RL Microgrid Management System")
        st.markdown("Real-time quantum-enhanced microgrid optimization with safety constraints")
        
        self.sidebar_controls()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Real-time Monitoring", 
            "🔮 Quantum vs Classical", 
            "🛡️ Safety Analysis", 
            "⚡ Power Flow", 
            "📈 Performance Metrics"
        ])
        
        with tab1:
            self.real_time_monitoring()
        
        with tab2:
            self.quantum_classical_comparison()
        
        with tab3:
            self.safety_analysis()
        
        with tab4:
            self.power_flow_analysis()
        
        with tab5:
            self.performance_metrics()
    
    def sidebar_controls(self):
        st.sidebar.header("🎛️ Control Panel")
        
        if st.sidebar.button("🚀 Train DRL Agent", type="primary"):
            with st.spinner("Training DRL Agent..."):
                quantum_scenarios = self.quantum_gen.generate_scenarios()
                agent = DRLAgent(scenarios=quantum_scenarios)
                st.session_state.drl_agent = agent.train(total_timesteps=50000)
                st.sidebar.success("✅ Training completed!")
        
        st.sidebar.subheader("🎯 Simulation Parameters")
        solar_capacity = st.sidebar.slider("Solar Capacity (MW)", 1.0, 5.0, 3.0)
        wind_capacity = st.sidebar.slider("Wind Capacity (MW)", 0.5, 3.0, 2.0)
        battery_capacity = st.sidebar.slider("Battery Capacity (MWh)", 2.0, 10.0, 5.0)
        
        st.sidebar.subheader("📡 Real-time Control")
        if st.sidebar.button("▶️ Start Real-time Simulation"):
            st.session_state.real_time_running = True
        
        if st.sidebar.button("⏸️ Stop Simulation"):
            st.session_state.real_time_running = False
        
        if st.sidebar.button("🔄 Reset Simulation"):
            st.session_state.simulation_data = []
            st.session_state.real_time_running = False
    
    def real_time_monitoring(self):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔋 Battery SOC", 
                f"{np.random.uniform(40, 90):.1f}%",
                delta=f"{np.random.uniform(-2, 2):.1f}%"
            )
        
        with col2:
            st.metric(
                "⚡ Grid Voltage", 
                f"{np.random.uniform(0.98, 1.02):.3f} pu",
                delta=f"{np.random.uniform(-0.01, 0.01):.3f}"
            )
        
        with col3:
            st.metric(
                "💰 Cost ($/h)", 
                f"${np.random.uniform(50, 200):.0f}",
                delta=f"${np.random.uniform(-20, 20):.0f}"
            )
        
        with col4:
            st.metric(
                "🛡️ Safety Score", 
                f"{np.random.uniform(85, 99):.1f}%",
                delta=f"{np.random.uniform(-1, 1):.1f}%"
            )
        
        if st.session_state.real_time_running:
            self.update_real_time_data()
            st.rerun()
        
        if len(st.session_state.simulation_data) > 0:
            self.plot_real_time_charts()
    
    def update_real_time_data(self):
        current_time = datetime.now()
        hour = current_time.hour
        
        solar_gen = max(0, 3 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        wind_gen = 2 * (0.5 + 0.5 * np.random.random())
        load = 2.5 * (0.8 + 0.4 * np.sin(2 * np.pi * hour / 24))
        
        if st.session_state.drl_agent:
            obs = np.array([solar_gen, wind_gen, load, 0.5, hour/24, 0.15])
            action = st.session_state.drl_agent.predict(obs)[0]
        else:
            action = np.random.uniform(-0.5, 0.5)
        
        data_point = {
            'timestamp': current_time,
            'solar_gen': solar_gen,
            'wind_gen': wind_gen,
            'load': load,
            'battery_action': action,
            'cost': abs(load - solar_gen - wind_gen) * 0.15,
            'voltage': 1.0 + np.random.normal(0, 0.01),
            'safety_score': np.random.uniform(90, 99)
        }
        
        st.session_state.simulation_data.append(data_point)
        
        if len(st.session_state.simulation_data) > 100:
            st.session_state.simulation_data.pop(0)
    
    def plot_real_time_charts(self):
        df = pd.DataFrame(st.session_state.simulation_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Power Generation & Load", "Battery Action", "Grid Voltage", "Operational Cost"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['solar_gen'], name="Solar", line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['wind_gen'], name="Wind", line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['load'], name="Load", line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['battery_action'], name="Battery Action", 
                      line=dict(color='green'), fill='tonexty'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['voltage'], name="Voltage", line=dict(color='purple')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cost'], name="Cost", line=dict(color='darkred')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Real-time System Monitoring")
        st.plotly_chart(fig, use_container_width=True)
    
    def quantum_classical_comparison(self):
        st.subheader("🔮 Quantum vs Classical Scenario Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Quantum Scenarios"):
                with st.spinner("Generating quantum scenarios..."):
                    quantum_scenarios = self.quantum_gen.generate_scenarios()
                    st.session_state.quantum_scenarios = quantum_scenarios
        
        with col2:
            if st.button("Generate Classical Scenarios"):
                with st.spinner("Generating classical scenarios..."):
                    classical_scenarios = self.classical_gan.generate_scenarios()
                    st.session_state.classical_scenarios = classical_scenarios
        
        if hasattr(st.session_state, 'quantum_scenarios') and hasattr(st.session_state, 'classical_scenarios'):
            self.plot_scenario_comparison()
    
    def plot_scenario_comparison(self):
        quantum_df = pd.DataFrame(st.session_state.quantum_scenarios, 
                                columns=['Solar', 'Wind', 'Load', 'Price'])
        classical_df = pd.DataFrame(st.session_state.classical_scenarios, 
                                  columns=['Solar', 'Wind', 'Load', 'Price'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Solar Generation", "Wind Generation", "Load Demand", "Electricity Price")
        )
        
        variables = ['Solar', 'Wind', 'Load', 'Price']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for var, (row, col) in zip(variables, positions):
            fig.add_trace(
                go.Histogram(x=quantum_df[var], name=f"Quantum {var}", 
                           opacity=0.7, nbinsx=30, histnorm='probability'),
                row=row, col=col
            )
            fig.add_trace(
                go.Histogram(x=classical_df[var], name=f"Classical {var}", 
                           opacity=0.7, nbinsx=30, histnorm='probability'),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Quantum vs Classical Scenario Distributions")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Statistical Comparison")
        
        comparison_data = []
        for var in variables:
            q_mean = quantum_df[var].mean()
            c_mean = classical_df[var].mean()
            q_std = quantum_df[var].std()
            c_std = classical_df[var].std()
            
            comparison_data.append({
                'Variable': var,
                'Quantum Mean': f"{q_mean:.3f}",
                'Classical Mean': f"{c_mean:.3f}",
                'Quantum Std': f"{q_std:.3f}",
                'Classical Std': f"{c_std:.3f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    def safety_analysis(self):
        st.subheader("🛡️ Safety Constraint Analysis")
        
        sample_state = {
            'battery_soc': st.slider("Battery SOC", 0.0, 1.0, 0.5),
            'voltage': st.slider("Grid Voltage (pu)", 0.9, 1.1, 1.0),
            'frequency': st.slider("Frequency (Hz)", 49.0, 51.0, 50.0),
            'total_generation': st.slider("Total Generation (MW)", 0.0, 10.0, 3.0),
            'load': st.slider("Load (MW)", 1.0, 5.0, 2.5)
        }
        
        action = st.slider("Battery Action (MW)", -2.0, 2.0, 0.0)
        
        is_safe, penalty = self.rag_safety.check_safety(action, sample_state)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if is_safe:
                st.success("✅ All constraints satisfied")
            else:
                st.error(f"❌ Safety violations detected (Penalty: {penalty})")
        
        with col2:
            st.info(f"Safety Score: {max(0, 100 - penalty/10):.1f}%")
        
        st.subheader("📋 Active Constraints")
        constraints_query = "battery voltage frequency power balance"
        results = self.rag_safety.get_relevant_constraints(constraints_query)
        
        if results and results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                with st.expander(f"Constraint {i+1}: {metadata['category'].upper()}"):
                    st.write(doc)
                    if 'threshold_min' in metadata:
                        st.write(f"Min: {metadata['threshold_min']}")
                    if 'threshold_max' in metadata:
                        st.write(f"Max: {metadata['threshold_max']}")
                    st.write(f"Penalty: {metadata['penalty']}")
    
    def power_flow_analysis(self):
        st.subheader("⚡ IEEE 33-Bus System Power Flow")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**System Parameters**")
            solar_gen = [st.slider(f"Solar Bus {bus} (MW)", 0.0, 2.0, 1.0) 
                        for bus in [5, 10, 15]]
            wind_gen = [st.slider(f"Wind Bus {bus} (MW)", 0.0, 1.5, 0.8) 
                       for bus in [8, 12]]
            battery_action = st.slider("Battery Action (MW)", -1.0, 1.0, 0.0)
            
            if st.button("Run Power Flow"):
                converged, results = self.digital_twin.simulate_timestep(
                    solar_gen, wind_gen, battery_action
                )
                
                if converged:
                    st.success("✅ Power flow converged")
                    st.session_state.pf_results = results
                else:
                    st.error("❌ Power flow failed to converge")
        
        with col2:
            if hasattr(st.session_state, 'pf_results'):
                results = st.session_state.pf_results
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.metric("Min Voltage", f"{results['min_voltage']:.4f} pu")
                    st.metric("Max Voltage", f"{results['max_voltage']:.4f} pu")
                    st.metric("Voltage Violations", results['voltage_violations'])
                
                with col2b:
                    st.metric("System Losses", f"{results['total_losses']:.3f} MW")
                    st.metric("Battery SOC", f"{results['battery_soc']:.1f}%")
                
                if 'voltages' in results:
                    voltage_df = pd.DataFrame({
                        'Bus': range(len(results['voltages'])),
                        'Voltage (pu)': results['voltages']
                    })
                    
                    fig = px.bar(voltage_df, x='Bus', y='Voltage (pu)', 
                               title="Bus Voltages")
                    fig.add_hline(y=0.95, line_dash="dash", line_color="red")
                    fig.add_hline(y=1.05, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
    
    def performance_metrics(self):
        st.subheader("📈 System Performance Analysis")
        
        if len(st.session_state.simulation_data) > 10:
            df = pd.DataFrame(st.session_state.simulation_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Economic Performance**")
                avg_cost = df['cost'].mean()
                total_cost = df['cost'].sum()
                st.metric("Average Cost ($/h)", f"${avg_cost:.2f}")
                st.metric("Total Cost ($)", f"${total_cost:.2f}")
                
                cost_fig = px.line(df, x='timestamp', y='cost', 
                                 title="Operational Cost Over Time")
                st.plotly_chart(cost_fig, use_container_width=True)
            
            with col2:
                st.write("**Technical Performance**")
                avg_voltage = df['voltage'].mean()
                voltage_std = df['voltage'].std()
                st.metric("Average Voltage (pu)", f"{avg_voltage:.4f}")
                st.metric("Voltage Std Dev", f"{voltage_std:.4f}")
                
                voltage_fig = px.line(df, x='timestamp', y='voltage',
                                    title="Grid Voltage Over Time")
                voltage_fig.add_hline(y=0.95, line_dash="dash", line_color="red")
                voltage_fig.add_hline(y=1.05, line_dash="dash", line_color="red")
                st.plotly_chart(voltage_fig, use_container_width=True)
            
            st.subheader("🎯 Control Actions Analysis")
            action_fig = px.histogram(df, x='battery_action', nbins=20,
                                    title="Distribution of Battery Actions")
            st.plotly_chart(action_fig, use_container_width=True)
        
        else:
            st.info("📊 Start real-time simulation to see performance metrics")

def main():
    dashboard = MicrogridDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
