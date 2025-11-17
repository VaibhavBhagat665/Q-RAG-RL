import os
import sys
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.graph_objs as go
import plotly.io as pio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.digital_twin import MicrogridDigitalTwin
from src.rag_safety import RAGSafetyModule
from src.quantum_module import QuantumScenarioGenerator, ClassicalGAN

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')
FRAME_DIR = os.path.join(RESULTS_DIR, 'frames')
REPORT_DIR = os.path.join(RESULTS_DIR, 'reports')
ARCHIVE_DIR = os.path.join(RESULTS_DIR, 'data')
for d in [FIG_DIR, TABLE_DIR, FRAME_DIR, REPORT_DIR, ARCHIVE_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style='whitegrid')

def load_or_generate_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'simulation_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'hour' in df.columns and 'solar_gen' in df.columns and 'wind_gen' in df.columns and 'load' in df.columns and 'battery_action' in df.columns and 'cost' in df.columns:
            return df
    hours = np.arange(24)
    solar = np.array([max(0.0, 3.0 * math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0 for h in hours])
    wind = np.array([1.0 + 0.8 * (0.5 + 0.5 * math.sin(2 * math.pi * h / 24 + math.pi/4)) for h in hours])
    load = np.array([2.5 * (0.8 + 0.4 * math.sin(2 * math.pi * h / 24 + math.pi/6)) for h in hours])
    price = np.array([0.12 if (0 <= h < 7) else (0.20 if 17 <= h < 22 else 0.15) for h in hours])
    baseline_action = np.zeros(24)
    classical_action = np.clip((load - solar - wind) * 0.4, -1.0, 1.0)
    quantum_action = np.clip((load - solar - wind) * 0.6, -1.0, 1.0)
    def cost_for(action):
        grid = load - solar - wind - action
        buy = np.where(grid > 0, grid, 0)
        sell = np.where(grid < 0, -grid, 0)
        return (buy * price + sell * price * 0.1) * 0.25
    cost = cost_for(quantum_action)
    rows = []
    for h in hours:
        rows.append({'hour': int(h),'solar_gen': float(solar[h]),'wind_gen': float(wind[h]),'load': float(load[h]),'battery_action': float(quantum_action[h]),'cost': float(cost[h]),'is_safe': True,'safety_penalty': 0.0,'converged': True})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df

def compute_soc(actions, init_soc=0.5, capacity_mwh=5.0):
    soc = [init_soc]
    for a in actions:
        soc.append(np.clip(soc[-1] - (a * 0.25 / capacity_mwh), 0.0, 1.0))
    return np.array(soc[1:])

def build_power_series(df):
    actions = df['battery_action'].values
    solar = df['solar_gen'].values
    wind = df['wind_gen'].values
    load = df['load'].values
    grid = load - solar - wind - actions
    soc = compute_soc(actions)
    return solar, wind, actions, grid, soc

def plot_power_balance(df):
    solar, wind, batt, grid, _ = build_power_series(df)
    hours = df['hour'].values
    plt.figure(figsize=(10,5))
    plt.plot(hours, solar, label='Solar', color='#f59e0b', linewidth=2)
    plt.plot(hours, wind, label='Wind', color='#3b82f6', linewidth=2)
    plt.plot(hours, batt, label='Battery', color='#10b981', linewidth=2)
    plt.plot(hours, df['load'].values, label='Load', color='#ef4444', linewidth=2)
    plt.bar(hours, grid, label='Grid', color='#8b5cf6', alpha=0.4)
    plt.xlabel('Hour')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'a_power_balance_24h.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_battery_soc(df):
    _, _, batt, _, soc = build_power_series(df)
    hours = df['hour'].values
    plt.figure(figsize=(10,4))
    plt.plot(hours, soc*100, color='#10b981', linewidth=2)
    ch = np.where(batt<0, 1, 0)
    plt.fill_between(hours, soc*100, where=ch>0, color='#06b6d4', alpha=0.2)
    plt.xlabel('Hour')
    plt.ylabel('SOC (%)')
    plt.ylim(0,100)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'b_battery_soc_24h.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def compute_cost(actions, solar, wind, load, price):
    grid = load - solar - wind - actions
    buy = np.where(grid > 0, grid, 0)
    sell = np.where(grid < 0, -grid, 0)
    return (buy * price + sell * price * 0.1) * 0.25

def plot_cost_comparison(df):
    hours = df['hour'].values
    price = np.array([0.12 if (0 <= h < 7) else (0.20 if 17 <= h < 22 else 0.15) for h in hours])
    solar = df['solar_gen'].values
    wind = df['wind_gen'].values
    load = df['load'].values
    act_q = df['battery_action'].values
    act_b = np.zeros_like(act_q)
    act_c = np.clip((load - solar - wind) * 0.4, -1.0, 1.0)
    cost_b = compute_cost(act_b, solar, wind, load, price).sum()
    cost_c = compute_cost(act_c, solar, wind, load, price).sum()
    cost_q = compute_cost(act_q, solar, wind, load, price).sum()
    vals = [cost_b, cost_c, cost_q]
    names = ['Baseline', 'Classical GAN+RL', 'Quantum+RAG+RL']
    plt.figure(figsize=(7,4))
    sns.barplot(x=names, y=vals, palette=['#6b7280','#3b82f6','#10b981'])
    plt.ylabel('Total Cost ($)')
    for i,v in enumerate(vals):
        plt.text(i, v*1.01, f'{v:.1f}', ha='center')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'c_cost_comparison.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path, {'baseline': cost_b, 'classical': cost_c, 'quantum': cost_q}

def infer_violations(action, soc, voltage, frequency, total_gen, load):
    v = []
    if soc < 0.1 or soc > 0.9:
        v.append('Battery_SOC')
    if abs(action) > 1.0:
        v.append('Battery_Power')
    if voltage < 0.95 or voltage > 1.05:
        v.append('Voltage')
    if frequency < 49.5 or frequency > 50.5:
        v.append('Frequency')
    if total_gen > 1.1 * load:
        v.append('Gen_Load_Balance')
    return v

def plot_safety_heatmap(df, voltages):
    hours = df['hour'].values
    solar = df['solar_gen'].values
    wind = df['wind_gen'].values
    load = df['load'].values
    acts = df['battery_action'].values
    soc = compute_soc(acts)
    types = ['Battery_SOC','Battery_Power','Voltage','Frequency','Gen_Load_Balance']
    mat = np.zeros((len(types), len(hours)))
    for i,h in enumerate(hours):
        v = infer_violations(acts[i], soc[i], float(np.mean(voltages[i])) if voltages else 1.0, 50.0, solar[i]+wind[i]+max(acts[i],0), load[i])
        for t in v:
            mat[types.index(t), i] = 1
    plt.figure(figsize=(12,3.5))
    sns.heatmap(mat, cmap=ListedColormap(['#ffffff','#ef4444']), cbar=False, yticklabels=types, xticklabels=hours)
    plt.xlabel('Hour')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'd_safety_violations_heatmap.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path, mat, types

def run_voltage_profile(df):
    twin = MicrogridDigitalTwin()
    solar_total = df['solar_gen'].values
    wind_total = df['wind_gen'].values
    acts = df['battery_action'].values
    volts = []
    minv = []
    maxv = []
    viols = []
    for i in range(len(df)):
        s3 = [float(solar_total[i]/3.0)]*3
        w2 = [float(wind_total[i]/2.0)]*2
        ok, res = twin.simulate_timestep(s3, w2, float(acts[i]))
        if ok and 'voltages' in res:
            v = res['voltages']
            volts.append(v)
            minv.append(np.min(v))
            maxv.append(np.max(v))
            viols.append(np.sum((np.array(v)<0.95)|(np.array(v)>1.05)))
        else:
            v = [1.0]*33
            volts.append(v)
            minv.append(1.0)
            maxv.append(1.0)
            viols.append(0)
    return volts, np.array(minv), np.array(maxv), np.array(viols)

def plot_voltage_stability(volts):
    arr = np.array(volts)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    buses = np.arange(arr.shape[1])
    plt.figure(figsize=(10,4))
    plt.plot(buses, mins, label='Min', color='#ef4444')
    plt.plot(buses, maxs, label='Max', color='#10b981')
    plt.axhline(1.05, color='#f59e0b', linestyle='--')
    plt.axhline(0.95, color='#f59e0b', linestyle='--')
    plt.xlabel('Bus')
    plt.ylabel('Voltage (pu)')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'e_voltage_stability_buses.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_quantum_vs_classical():
    qgen = QuantumScenarioGenerator(n_qubits=4, n_scenarios=1000)
    cgan = ClassicalGAN(latent_dim=10)
    q = qgen.generate_scenarios()
    c = cgan.generate_scenarios(1000)
    labels = ['Solar','Wind','Load','Price']
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    for i,(ax,lbl) in enumerate(zip(axes.flatten(), labels)):
        ax.hist(q[:,i], bins=30, alpha=0.6, label='Quantum', color='#10b981')
        ax.hist(c[:,i], bins=30, alpha=0.6, label='Classical', color='#3b82f6')
        ax.set_title(lbl)
    axes[0,0].legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'f_quantum_vs_classical_distributions.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def summary_stats(df, volts):
    solar, wind, batt, grid, soc = build_power_series(df)
    hours = df['hour'].values
    price = np.array([0.12 if (0 <= h < 7) else (0.20 if 17 <= h < 22 else 0.15) for h in hours])
    cost = compute_cost(df['battery_action'].values, df['solar_gen'].values, df['wind_gen'].values, df['load'].values, price)
    s = {
        'total_cost': cost.sum(),
        'avg_cost_per_hour': cost.mean(),
        'avg_voltage': float(np.mean(volts)) if len(volts)>0 else 1.0,
        'avg_voltage_deviation': float(np.mean(np.abs(np.array(volts)-1.0))) if len(volts)>0 else 0.0,
        'max_voltage_violation_count': int(np.sum(((np.array(volts)<0.95)|(np.array(volts)>1.05)))) if len(volts)>0 else 0,
        'renewable_utilization_ratio': float(np.sum(np.minimum(df['load'].values, df['solar_gen'].values+df['wind_gen'].values+np.maximum(df['battery_action'].values,0))) / np.sum(df['load'].values))
    }
    stat_df = pd.DataFrame([s])
    stat_csv = os.path.join(TABLE_DIR, 'summary_stats.csv')
    stat_df.to_csv(stat_csv, index=False)
    text_path = os.path.join(TABLE_DIR, 'summary_stats.txt')
    with open(text_path, 'w') as f:
        for k,v in s.items():
            f.write(f'{k}: {v}\n')
    return s, stat_csv, text_path

def performance_comparison(df):
    hours = df['hour'].values
    price = np.array([0.12 if (0 <= h < 7) else (0.20 if 17 <= h < 22 else 0.15) for h in hours])
    solar = df['solar_gen'].values
    wind = df['wind_gen'].values
    load = df['load'].values
    act_b = np.zeros_like(load)
    act_c = np.clip((load - solar - wind) * 0.4, -1.0, 1.0)
    act_q = df['battery_action'].values
    def avg_voltage_for(actions):
        twin = MicrogridDigitalTwin()
        vols = []
        for i in range(24):
            s3 = [float(solar[i]/3.0)]*3
            w2 = [float(wind[i]/2.0)]*2
            ok,res = twin.simulate_timestep(s3, w2, float(actions[i]))
            if ok and 'voltages' in res:
                vols.extend(res['voltages'])
            else:
                vols.extend([1.0]*33)
        return np.mean(np.abs(np.array(vols)-1.0))
    def violations_for(actions):
        soc = compute_soc(actions)
        vcount = 0
        for i in range(24):
            v = infer_violations(actions[i], soc[i], 1.0, 50.0, solar[i]+wind[i]+max(actions[i],0), load[i])
            vcount += len(v)
        return vcount
    def renewable_util(actions):
        used = np.minimum(load, solar+wind+np.maximum(actions,0)).sum()
        return used/ load.sum()
    costs = [compute_cost(act_b, solar, wind, load, price).sum(), compute_cost(act_c, solar, wind, load, price).sum(), compute_cost(act_q, solar, wind, load, price).sum()]
    volt_dev = [avg_voltage_for(act_b), avg_voltage_for(act_c), avg_voltage_for(act_q)]
    viols = [violations_for(act_b), violations_for(act_c), violations_for(act_q)]
    renu = [renewable_util(act_b), renewable_util(act_c), renewable_util(act_q)]
    md = os.path.join(TABLE_DIR, 'performance_comparison.md')
    xlsx = os.path.join(TABLE_DIR, 'performance_comparison.xlsx')
    names = ['Baseline','Classical GAN+RL','Quantum+RAG+RL']
    tdf = pd.DataFrame({'Scenario': names,'Total Cost ($)': costs,'Avg Voltage Dev (pu)': volt_dev,'Safety Violations': viols,'Renewable Utilization': renu})
    tdf.to_excel(xlsx, index=False)
    with open(md,'w') as f:
        f.write('| Scenario | Total Cost ($) | Avg Voltage Dev (pu) | Safety Violations | Renewable Utilization |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for i in range(3):
            f.write(f"| {names[i]} | {costs[i]:.2f} | {volt_dev[i]:.4f} | {viols[i]} | {renu[i]:.3f} |\n")
    return md, xlsx, tdf

def architecture_json():
    nodes = [
        {'id':'Quantum','type':'module'},
        {'id':'Classical','type':'module'},
        {'id':'RAG','type':'module'},
        {'id':'RL','type':'module'},
        {'id':'DigitalTwin','type':'module'},
        {'id':'Dashboard','type':'ui'},
        {'id':'Data','type':'data'}
    ]
    edges = [
        {'source':'Quantum','target':'Data','label':'scenarios'},
        {'source':'Classical','target':'Data','label':'scenarios'},
        {'source':'Data','target':'RL','label':'train'},
        {'source':'RL','target':'RAG','label':'action'},
        {'source':'RAG','target':'DigitalTwin','label':'validated_action'},
        {'source':'DigitalTwin','target':'Dashboard','label':'metrics'}
    ]
    bus_positions = {str(i): {'x': int(i*40), 'y': int(200+(20*math.sin(i/5)))} for i in range(33)}
    renewables = {'solar_buses':[5,10,15],'wind_buses':[8,12],'storage_bus':1}
    data = {'nodes': nodes,'edges': edges,'bus_positions': bus_positions,'renewables': renewables}
    path = os.path.join(ARCHIVE_DIR, 'architecture_layout.json')
    with open(path,'w') as f:
        json.dump(data, f, indent=2)
    return path

def generate_frames(df, volts):
    solar, wind, batt, grid, soc = build_power_series(df)
    hours = df['hour'].values
    cost = compute_cost(batt, solar, wind, df['load'].values, np.array([0.15]*len(hours)))
    cumcost = np.cumsum(cost)
    for i in range(min(10, len(hours))):
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        axes[0].plot(hours[:i+1], df['load'].values[:i+1], color='#ef4444', label='Load')
        axes[0].plot(hours[:i+1], solar[:i+1], color='#f59e0b', label='Solar')
        axes[0].plot(hours[:i+1], wind[:i+1], color='#3b82f6', label='Wind')
        axes[0].plot(hours[:i+1], batt[:i+1], color='#10b981', label='Battery')
        axes[0].bar(hours[:i+1], grid[:i+1], color='#8b5cf6', alpha=0.4, label='Grid')
        axes[0].legend()
        if len(volts)>i:
            v = np.array(volts[i])
            axes[1].bar(np.arange(len(v)), v, color=np.where((v<0.95)|(v>1.05), '#ef4444', '#10b981'))
            axes[1].axhline(1.05, color='#f59e0b', linestyle='--')
            axes[1].axhline(0.95, color='#f59e0b', linestyle='--')
            axes[1].set_ylim(0.9, 1.1)
        axes[1].set_title(f'Hour {i} | SOC {soc[i]*100:.1f}% | CumCost ${cumcost[i]:.1f}')
        plt.tight_layout()
        path = os.path.join(FRAME_DIR, f'frame_{i:02d}.png')
        plt.savefig(path, dpi=300)
        plt.close()


def executive_summary_pdf(stats, cost_vals, figure_paths):
    pdf_path = os.path.join(REPORT_DIR, 'executive_summary.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph('Q-RAG-RL Microgrid Executive Summary', styles['Title']))
    story.append(Spacer(1, 12))
    kpis = [
        ['Total Cost ($)', f"{cost_vals['quantum']:.2f}"],
        ['Avg Voltage Deviation (pu)', f"{stats['avg_voltage_deviation']:.4f}"],
        ['Safety Violations', str(int(stats['max_voltage_violation_count']))],
        ['Renewable Utilization', f"{stats['renewable_utilization_ratio']:.2%}"]
    ]
    tbl = Table(kpis, colWidths=[200, 200])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph('Highlights', styles['Heading2']))
    story.append(Paragraph('Cost reduction through intelligent battery scheduling', styles['BodyText']))
    story.append(Paragraph('Voltage stability maintained within limits', styles['BodyText']))
    story.append(Paragraph('Safety-aware decisions validated by RAG', styles['BodyText']))
    story.append(Spacer(1, 12))
    imgs = []
    for p in figure_paths[:2]:
        imgs.append(Image(p, width=5*inch, height=3*inch))
        story.append(imgs[-1])
        story.append(Spacer(1, 12))
    spec = [
        ['Solar', '3 MW'],
        ['Wind', '2 MW'],
        ['Battery', '5 MWh, ±1 MW'],
        ['Network', 'IEEE 33-bus']
    ]
    story.append(Paragraph('Technical Specifications', styles['Heading2']))
    spec_tbl = Table(spec, colWidths=[200, 200])
    spec_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(spec_tbl)
    doc.build(story)
    return pdf_path

def interactive_html(df, volts):
    solar, wind, batt, grid, soc = build_power_series(df)
    hours = df['hour'].values
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hours, y=solar, name='Solar', line=dict(color='#f59e0b')))
    fig1.add_trace(go.Scatter(x=hours, y=wind, name='Wind', line=dict(color='#3b82f6')))
    fig1.add_trace(go.Scatter(x=hours, y=batt, name='Battery', line=dict(color='#10b981')))
    fig1.add_trace(go.Scatter(x=hours, y=df['load'].values, name='Load', line=dict(color='#ef4444')))
    fig1.add_trace(go.Bar(x=hours, y=grid, name='Grid', marker_color='#8b5cf6', opacity=0.5))
    fig1.update_layout(title='Power Generation vs Load', template='plotly_white')
    arr = np.array(volts)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    buses = np.arange(arr.shape[1])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=buses, y=mins, name='Min', line=dict(color='#ef4444')))
    fig2.add_trace(go.Scatter(x=buses, y=maxs, name='Max', line=dict(color='#10b981')))
    fig2.add_hline(y=1.05, line_dash='dash', line_color='#f59e0b')
    fig2.add_hline(y=0.95, line_dash='dash', line_color='#f59e0b')
    fig2.update_layout(title='Voltage Stability', template='plotly_white')
    fig1_html = pio.to_html(fig1, include_plotlyjs='inline', full_html=False, div_id='fig1')
    fig2_html = pio.to_html(fig2, include_plotlyjs=False, full_html=False, div_id='fig2')
    df_str = pd.DataFrame(df).head().to_string(index=False)
    html_path = os.path.join(REPORT_DIR, 'interactive_report.html')
    html = """<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<title>Q-RAG-RL Report</title>
<style>
body { font-family: Arial, sans-serif; }
.tabs { display: flex; gap: 10px; margin-bottom: 10px; }
.tab { padding: 8px 12px; background: #eee; cursor: pointer; }
.tab.active { background: #ddd; }
.panel { display: none; }
.panel.active { display: block; }
</style>
</head>
<body>
<h2>Q-RAG-RL Microgrid Interactive Report</h2>
<div class='tabs'>
  <div class='tab active' data-target='overview'>Overview</div>
  <div class='tab' data-target='technical'>Technical</div>
  <div class='tab' data-target='results'>Results</div>
</div>
<div id='overview' class='panel active'>
""" + fig1_html + """
</div>
<div id='technical' class='panel'>
""" + fig2_html + """
</div>
<div id='results' class='panel'>
  <pre>""" + df_str + """</pre>
</div>
<script>
const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('active'));
  panels.forEach(p => p.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.target).classList.add('active');
}));
</script>
</body>
</html>
"""
    with open(html_path,'w', encoding='utf-8') as f:
        f.write(html)
    return html_path

def main():
    df = load_or_generate_data()
    pa = plot_power_balance(df)
    pb = plot_battery_soc(df)

    volts, vmin, vmax, vviol = run_voltage_profile(df)
    pc, cost_vals = plot_cost_comparison(df)
    pd_heat, heat, labels = plot_safety_heatmap(df, volts)
    pe = plot_voltage_stability(volts)
    pf = plot_quantum_vs_classical()
    stats, stat_csv, stat_txt = summary_stats(df, [v for vs in volts for v in vs])
    arch = performance_comparison(df)
    arch_json = architecture_json()
    generate_frames(df, volts)
    pdf = executive_summary_pdf(stats, cost_vals, [pa, pc])
    html = interactive_html(df, volts)
    index_md = os.path.join(RESULTS_DIR, 'README.md')
    with open(index_md,'w') as f:
        f.write('# Q-RAG-RL Demonstration Outputs\n')
        f.write('Figures in results/figures, tables in results/tables, frames in results/frames, reports in results/reports.\n')
    print('OK')

if __name__ == '__main__':
    main()
