"""
Preprocessing Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Handles data cleaning, time parsing, feature encoding, and split-key preparation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_time_to_minutes(time_str):
    """Converts time format (HH:MM or H:MM) into minutes past midnight."""
    if pd.isna(time_str):
        return np.nan
    h, m = map(int, str(time_str).strip().split(':'))
    return h * 60 + m

def preprocess_dataset(input_csv="data/processed/synthetic_session_bins.csv"):
    if not os.path.exists(input_csv):
        print(f"[*] Data file missing at {input_csv}. Generating synthetic data...")
        try:
            from src.generate_synthetic_data import generate_synthetic_dataset
        except ImportError:
            from generate_synthetic_data import generate_synthetic_dataset
        generate_synthetic_dataset(output_path=input_csv)

    df = pd.read_csv(input_csv)
    
    # Target Encoding
    df['Target'] = df['Stress_Label'].map({'Stress_Absent': 0, 'Stress_Present': 1})

    # Temporal Parsing
    df['Date_Parsed'] = pd.to_datetime(df['Observation_Date'], format='%m/%d/%Y')
    df['Month'] = df['Date_Parsed'].dt.month
    df['Day'] = df['Date_Parsed'].dt.day
    df['DayOfWeek'] = df['Date_Parsed'].dt.dayofweek

    # Convert Time String to Minutes
    time_cols = ['Keeper_Arrival_Time', 'Keeper_Departure_Time', 'Meal_Start_Time', 'Meal_End_Time']
    for col in time_cols:
        df[f'{col}_Min'] = df[col].apply(parse_time_to_minutes)

    # Calculate Derived Durations
    df['Keeper_Presence_Duration_Calc'] = df['Keeper_Departure_Time_Min'] - df['Keeper_Arrival_Time_Min']
    df['Meal_Duration_Calc'] = df['Meal_End_Time_Min'] - df['Meal_Start_Time_Min']
    df['Keeper_Arrival_to_Meal_Start'] = df['Meal_Start_Time_Min'] - df['Keeper_Arrival_Time_Min']

    print(f"[+] Dataset preprocessed cleanly: {len(df)} rows, {df['Target'].sum()} stress-present instances.")
    return df

if __name__ == '__main__':
    df_clean = preprocess_dataset()
    print(df_clean.head())
