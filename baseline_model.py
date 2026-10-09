"""
Baseline Model Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Implements the baseline classifier (Random Forest) trained ONLY on general movement/trajectory features.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

try:
    from src.features import build_transformers
except ImportError:
    from features import build_transformers

def get_baseline_pipeline(n_estimators=100, random_state=42):
    baseline_transformer, _ = build_transformers()
    
    pipeline = Pipeline([
        ('prep', baseline_transformer),
        ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])
    return pipeline
