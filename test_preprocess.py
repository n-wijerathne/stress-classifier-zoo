"""
Unit Tests for Preprocessing Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

try:
    from src.preprocess import parse_time_to_minutes, preprocess_dataset
except ImportError:
    from preprocess import parse_time_to_minutes, preprocess_dataset

def test_parse_time_to_minutes():
    assert parse_time_to_minutes("08:15") == 495
    assert parse_time_to_minutes("15:30") == 930
    assert parse_time_to_minutes("0:00") == 0

def test_preprocess_dataset():
    test_path = "data/processed/synthetic_session_bins.csv"
    df = preprocess_dataset(test_path)
    
    assert 'Target' in df.columns
    assert 'Keeper_Presence_Duration_Calc' in df.columns
    assert 'Meal_Duration_Calc' in df.columns
    assert df['Target'].isin([0, 1]).all()
