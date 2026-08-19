"""
features.py

Builds the engineered feature set described in Section 2.2 of the
methodology: separate feature groups for Branch A (tabular), Branch B
(LSTM sequence), and the baseline model.
"""

import pandas as pd

TABULAR_FEATURES = [
    "inter_meal_interval_hr",
    "diet_variety_score",
    "keeper_proximity_minutes",
    "minutes_since_keeper_arrival",
    "minutes_to_meal",
    "species_code",
    "keeper_code",
]

BASELINE_FEATURES = [
    "activity_level",
    "boundary_time_pct",
    "location_changes",
]

LABEL_COL = "stress_label"
GROUP_COL = "group_key"


def get_tabular_xy(df: pd.DataFrame):
    X = df[TABULAR_FEATURES].copy()
    y = df[LABEL_COL].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups


def get_baseline_xy(df: pd.DataFrame):
    X = df[BASELINE_FEATURES].copy()
    y = df[LABEL_COL].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups


def build_sequences(df: pd.DataFrame, window: int = 6):
    """
    Builds windowed sequences per animal/day for the LSTM branch (Branch B).
    Each sequence is `window` consecutive 5-minute bins; the label is the
    stress_label of the final bin in the window (predicting current state
    from recent history), consistent with the anticipatory-pattern
    rationale in Section 2.2.
    """
    import numpy as np

    seq_features = ["minutes_to_meal", "minutes_since_keeper_arrival",
                     "keeper_proximity_minutes", "inter_meal_interval_hr"]

    sequences, labels, groups = [], [], []
    for (_, day_df) in df.sort_values("bin_start").groupby(["animal_id", "date"]):
        vals = day_df[seq_features].to_numpy()
        y = day_df[LABEL_COL].to_numpy()
        g = day_df[GROUP_COL].to_numpy()
        for i in range(window, len(day_df) + 1):
            sequences.append(vals[i - window:i])
            labels.append(y[i - 1])
            groups.append(g[i - 1])

    return np.array(sequences), np.array(labels), np.array(groups)
