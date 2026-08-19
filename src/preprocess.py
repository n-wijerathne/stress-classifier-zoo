"""
preprocess.py

Cleans and prepares the merged session-bin table for modelling, per
Section 2.1 ("Preprocessing") of the Milestone 2 methodology.

Input:  data/processed/synthetic_session_bins.csv
        (or, once real data exists, the equivalent merged real file
        produced by ingest.py from raw keeper/feeding + observation logs)
Output: data/processed/model_ready.csv

Steps implemented, matching the methodology document:
  1. Missing-value handling
  2. Categorical encoding (species, keeper_id)
  3. Continuous feature scaling (fit on train fold only — see evaluate.py;
     this script keeps raw values and defers scaling to the CV loop so
     no leakage occurs)
  4. Day-grouped split key preparation (so evaluate.py can group by day)
"""

import pandas as pd


def load_raw(path: str = "data/processed/synthetic_session_bins.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Drop rows with a missing label — cannot train/evaluate on unlabeled rows.
    df = df.dropna(subset=["stress_label"])
    # Forward-fill small gaps in feeding-derived fields within the same animal/day,
    # per the methodology's "missing-log handling" rule.
    fill_cols = ["inter_meal_interval_hr", "diet_variety_score", "keeper_id"]
    df[fill_cols] = df.groupby(["animal_id", "date"])[fill_cols].ffill()
    df = df.dropna(subset=fill_cols)
    print(f"handle_missing: kept {len(df)}/{before} rows")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["species_code"] = df["species"].astype("category").cat.codes
    df["animal_code"] = df["animal_id"].astype("category").cat.codes
    df["keeper_code"] = df["keeper_id"].astype("category").cat.codes
    return df


def add_group_key(df: pd.DataFrame) -> pd.DataFrame:
    # Group key used later for day-grouped CV splitting (Section 2.4).
    df = df.copy()
    df["group_key"] = df["animal_id"] + "_" + df["date"]
    return df


def run(in_path: str = "data/processed/synthetic_session_bins.csv",
        out_path: str = "data/processed/model_ready.csv") -> pd.DataFrame:
    df = load_raw(in_path)
    df = handle_missing(df)
    df = encode_categoricals(df)
    df = add_group_key(df)
    df.to_csv(out_path, index=False)
    print(f"preprocess.run: wrote {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run()
