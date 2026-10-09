"""
Branch A Tabular Models Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Implements Branch A classifiers (Random Forest & SVM) trained on feeding schedule + keeper routine features.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from src.features import build_transformers
except ImportError:
    from features import build_transformers

def get_tabular_rf_pipeline(n_estimators=100, random_state=42):
    _, full_transformer = build_transformers()
    
    return Pipeline([
        ('prep', full_transformer),
        ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])

def get_tabular_svm_pipeline(kernel='rbf', C=1.0, random_state=42):
    _, full_transformer = build_transformers()
    
    return Pipeline([
        ('prep', full_transformer),
        ('clf', SVC(kernel=kernel, C=C, probability=True, random_state=random_state))
    ])
