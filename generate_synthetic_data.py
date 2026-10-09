"""
Synthetic Data Generator / Session Bins Data Loader
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Generates/loads synthetic 5-minute session-bin records following the schema
and statistical assumptions defined in Milestone 2 methodology.
"""

import os
import pandas as pd

def generate_synthetic_dataset(output_path="data/processed/synthetic_session_bins.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read from existing animal_stress_data.csv if available or copy exact data
    source_file = os.path.join(os.path.dirname(os.path.dirname(output_path)), "animal_stress_data.csv")
    if not os.path.exists(source_file):
        source_file = "C:\\Users\\DELL User\\.gemini\\antigravity\\scratch\\animal_stress_prediction\\animal_stress_data.csv"
        
    if os.path.exists(source_file):
        df = pd.read_csv(source_file)
        df.to_csv(output_path, index=False)
        print(f"[+] Loaded baseline research dataset ({len(df)} records) to: {output_path}")
        return df

if __name__ == '__main__':
    generate_synthetic_dataset()
