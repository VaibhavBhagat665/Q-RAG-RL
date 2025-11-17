import subprocess
import sys
import os

def run_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "src", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

if __name__ == "__main__":
    run_dashboard()
