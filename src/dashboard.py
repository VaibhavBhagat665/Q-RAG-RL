"""
Streamlit Dashboard for Q-RAG-RL Microgrid Management
Simplified version with correct imports and no emojis
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum_scenarios import QuantumScenarioGenerator
from src.rag_safety import OptimizedRAGSafety
from src.drl_agent import OptimizedDRLAgent
from src.nrel_data import NRELDataLoader

st.set_page_config(
    page_title="Q-RAG-RL Microgrid Management",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Q-RAG-RL Microgrid Management System")
st.markdown("Quantum-enhanced reinforcement learning for safe microgrid control")

# Initialize components
@st.cache_resource
def load_components():
    quantum_gen = QuantumScenarioGenerator(n_qubits=6, n_scenarios=1000)
    rag_safety = OptimizedRAGSafety()
    nrel_loader = NRELDataLoader()
    return quantum_gen, rag_safety, nrel_loader

quantum_gen, rag_safety, nrel_loader = load_components()

# Load trained agent
@st.cache_resource
def load_agent():
    agent = OptimizedDRLAgent()
    try:
        agent.load_model("models/trained_agent")
        return agent, True
    except:
        return agent, False

agent, model_loaded = load_agent()

# Sidebar
st.sidebar.header("Control Panel")

if model_loaded:
    st.sidebar.success("Model loaded successfully")
else:
    st.sidebar.warning("No trained model found. Run: python train.py")

st.sidebar.subheader("Simulation Parameters")
solar_capacity = st.sidebar.slider("Solar Capacity (MW)", 1.0, 5.0, 3.0)
wind_capacity = st.sidebar.slider("Wind Capacity (MW)", 0.5, 3.0, 2.0)
battery_capacity = st.sidebar.slider("Battery Capacity (MWh)", 2.0, 10.0, 5.0)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Real-time Monitoring", "Safety Analysis", "Performance Metrics"])

with tab1:
    st.subheader("Real-time System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Battery SOC", f"{np.random.uniform(40, 90):.1f}%")
    
    with col2:
        st.metric("Grid Voltage", f"{np.random.uniform(0.98, 1.02):.3f} pu")
    
    with col3:
        st.metric("Frequency", f"{np.random.uniform(49.8, 50.2):.2f} Hz")
    
    with col4:
        st.metric("Grid Power", f"{np.random.uniform(-1, 2):.2f} MW")
    
    # Load real data
    if st.button("Load Real NREL Data"):
        with st.spinner("Loading real weather data..."):
            solar_df = nrel_loader.get_solar_data()
            wind_df = nrel_loader.get_wind_data()
            solar_power, wind_power = nrel_loader.convert_to_power(solar_df, wind_df)
            load_profile = nrel_loader.get_load_profile()
            
            # Plot first 24 hours
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(24)), y=solar_power[:24], name="Solar", mode='lines'))
            fig.add_trace(go.Scatter(x=list(range(24)), y=wind_power[:24], name="Wind", mode='lines'))
            fig.add_trace(go.Scatter(x=list(range(24)), y=load_profile[:24], name="Load", mode='lines'))
            fig.update_layout(title="24-Hour Generation and Load Profile", xaxis_title="Hour", yaxis_title="Power (MW)")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Safety Constraint Analysis")
    
    st.write("Test safety constraints with different battery actions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        battery_soc = st.slider("Battery SOC", 0.0, 1.0, 0.5)
        voltage = st.slider("Voltage (pu)", 0.90, 1.10, 1.0)
        frequency = st.slider("Frequency (Hz)", 49.0, 51.0, 50.0)
    
    with col2:
        action = st.slider("Battery Action (MW)", -2.0, 2.0, 0.0)
        total_gen = st.slider("Total Generation (MW)", 0.0, 5.0, 2.0)
        load = st.slider("Load (MW)", 0.0, 5.0, 2.5)
    
    # Check safety
    sample_state = {
        'battery_soc': battery_soc,
        'voltage': voltage,
        'frequency': frequency,
        'total_generation': total_gen,
        'load': load
    }
    
    is_safe, penalty = rag_safety.check_safety_graded(action, sample_state)
    
    if is_safe:
        st.success(f"SAFE: All constraints satisfied (Penalty: {penalty:.2f})")
    else:
        st.error(f"UNSAFE: Safety violations detected (Penalty: {penalty:.2f})")
    
    # Show constraint limits
    st.subheader("Constraint Limits")
    constraints_df = pd.DataFrame({
        'Constraint': ['Battery SOC', 'Voltage', 'Frequency', 'Battery Power', 'Gen/Load Ratio'],
        'Min Limit': ['10%', '0.95 pu', '49.5 Hz', '-1.0 MW', 'N/A'],
        'Max Limit': ['90%', '1.05 pu', '50.5 Hz', '+1.0 MW', '110%'],
        'Current Value': [
            f'{battery_soc*100:.1f}%',
            f'{voltage:.3f} pu',
            f'{frequency:.2f} Hz',
            f'{action:.2f} MW',
            f'{(total_gen/load)*100:.1f}%' if load > 0 else 'N/A'
        ]
    })
    st.dataframe(constraints_df, use_container_width=True)

with tab3:
    st.subheader("Performance Metrics")
    
    # Load evaluation results if available
    try:
        eval_df = pd.read_csv('results/evaluation_real_data.csv')
        
        st.write(f"Evaluation on {len(eval_df)} hours of real data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cost = eval_df['cost'].sum()
            st.metric("Total Cost", f"${total_cost:.2f}")
        
        with col2:
            violations = len(eval_df[eval_df['safe'] == False])
            st.metric("Safety Violations", f"{violations}/{len(eval_df)}")
        
        with col3:
            avg_soc = eval_df['battery_soc'].mean()
            st.metric("Avg Battery SOC", f"{avg_soc*100:.1f}%")
        
        # Plot cost over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eval_df['hour'], y=eval_df['cost'], mode='lines+markers', name='Cost'))
        fig.update_layout(title="Cost per Hour", xaxis_title="Hour", yaxis_title="Cost ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot battery SOC
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=eval_df['hour'], y=eval_df['battery_soc']*100, mode='lines+markers', name='SOC'))
        fig2.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Min SOC")
        fig2.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Max SOC")
        fig2.update_layout(title="Battery State of Charge", xaxis_title="Hour", yaxis_title="SOC (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
    except FileNotFoundError:
        st.info("No evaluation results found. Run: python evaluate.py")

st.sidebar.markdown("---")
st.sidebar.info("Q-RAG-RL: Quantum-Enhanced Reinforcement Learning for Microgrid Control")
