"""Run with:  python -m pytest tests/"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
import preprocess


def _raw():
    return pd.read_csv(config.PUBLIC_INPUT)


def test_parse_time():
    assert preprocess.parse_time("8:25") == 505
    assert preprocess.parse_time(" 15:05 ") == 905
    assert pd.isna(preprocess.parse_time("25:00"))
    assert pd.isna(preprocess.parse_time(None))


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        preprocess.standardise_columns(_raw().drop(columns=["Keeper_Arrival_Time"]))


def test_new_data_without_label_can_be_prepared_but_not_cleaned():
    df = preprocess.prepare(_raw().drop(columns=["Stress_Label"]))
    assert "keeper_presence_min" in df.columns
    with pytest.raises(ValueError):
        preprocess.clean(df)


def test_engineered_durations_are_consistent():
    df = preprocess.prepare(_raw())
    assert (df["keeper_presence_min"] > 0).all()
    assert (df["meal_duration_min"] > 0).all()
    expected = df["meal_start_time_min"] - df["keeper_arrival_time_min"]
    assert (df["keeper_arrival_to_meal_start_min"] == expected).all()


def test_clean_encodes_target_and_keeps_all_rows():
    df = preprocess.clean(preprocess.prepare(_raw()))
    assert set(df["target"]) == {0, 1}
    assert len(df) == len(_raw())
    assert df["group_key"].nunique() == 28


def test_invalid_label_raises():
    df = preprocess.prepare(_raw())
    df.loc[0, "stress_label"] = "Maybe"
    with pytest.raises(ValueError):
        preprocess.clean(df)


def test_public_data_has_no_individual_ids():
    assert "Individual_ID" not in _raw().columns
