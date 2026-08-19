"""
tests/test_preprocess.py

Minimal smoke tests for the preprocessing pipeline, run with:
    python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from preprocess import handle_missing, encode_categoricals, add_group_key


def _sample_df():
    return pd.DataFrame({
        "animal_id": ["ELEPHANT_01", "ELEPHANT_01"],
        "species": ["elephant", "elephant"],
        "date": ["2026-06-01", "2026-06-01"],
        "keeper_id": ["KPR_01", None],
        "inter_meal_interval_hr": [22.5, None],
        "diet_variety_score": [3, None],
        "stress_label": [0, 1],
    })


def test_handle_missing_forward_fills_within_group():
    df = _sample_df()
    out = handle_missing(df)
    assert out["keeper_id"].isna().sum() == 0
    assert len(out) == 2


def test_encode_categoricals_adds_codes():
    df = handle_missing(_sample_df())
    out = encode_categoricals(df)
    assert "species_code" in out.columns
    assert "keeper_code" in out.columns


def test_add_group_key_is_unique_per_animal_day():
    df = add_group_key(_sample_df())
    assert (df["group_key"] == "ELEPHANT_01_2026-06-01").all()
