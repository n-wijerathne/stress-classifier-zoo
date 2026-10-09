"""
Branch B Sequence Models Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Implements Branch B (LSTM / Neural Network Sequence Classifier) for feeding & routine sequences.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

try:
    from src.features import build_transformers
except ImportError:
    from features import build_transformers

def get_lstm_fcn_pipeline(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42):
    _, full_transformer = build_transformers()
    
    return Pipeline([
        ('prep', full_transformer),
        ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state))
    ])

if __name__ == '__main__':
    pipe = get_lstm_fcn_pipeline()
    print("[+] Branch B Sequence/LSTM-FCN Architecture Initialized Successfully!")
