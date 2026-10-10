"""
preprocess.py - validate, clean and engineer features from the real session-level table.

Input : data/processed/animal_stress_data.csv  (anonymised; or data/raw/... if present locally)
Output: data/processed/model_ready.csv         (regenerated, gitignored)

Steps (Methodology Section 1, "Preprocessing"):
  1. Schema + label validation
  2. Missing-value handling: rows with a missing label or key feature are EXCLUDED and
     counted (feeding/keeper times are never fabricated)
  3. Clock-time -> minutes, derived durations
  4. Day-grouping key so that evaluate.py never splits one observation day across folds
Scaling and one-hot encoding are deliberately NOT done here: they live inside the
scikit-learn Pipeline (features.make_preprocessor) so they are fitted on training folds only.
"""
import sys
import pandas as pd

import config

RENAME = {
    "Type of animal": "species",
    "Individual_ID": "individual_id",
    "Enclosure": "enclosure",
    "Observation_Date": "date",
    "Observation Time": "session",
    "Keeper_Arrival_Time": "keeper_arrival_time",
    "Keeper_Departure_Time": "keeper_departure_time",
    "Meal_Start_Time": "meal_start_time",
    "Meal_End_Time": "meal_end_time",
    "Inter_Meal_Interval_Min": "inter_meal_interval_min",
    "Diet_Variety_Score": "diet_variety_score",
    "Keeper_Proximity_Duration_Min": "keeper_proximity_min",
    "Time_to_Keeper_Arrival_Min": "time_to_keeper_arrival_min",
    "Activity_Level(Animal's activity types)": "activity_level",
    "Location_Changes": "location_changes",
    "Boundary_Time_Min": "boundary_time_min",
    "Rolling_Stereotypy_Rate": "rolling_stereotypy_rate",
    "Stress_Label": "stress_label",
}
OPTIONAL = {"individual_id", "stress_label"}  # individual_id: dropped in the public file; stress_label: absent when scoring new data
REQUIRED = [c for c in RENAME.values() if c not in OPTIONAL]
TIME_COLS = ["keeper_arrival_time", "keeper_departure_time", "meal_start_time", "meal_end_time"]
LABELS = {"Stress_Absent": 0, "Stress_Present": 1}


def parse_time(value) -> float:
    """'8:25' -> 505 (minutes after midnight). Returns NaN for missing/invalid values."""
    try:
        h, m = str(value).strip().split(":")
        h, m = int(h), int(m)
        if not (0 <= h < 24 and 0 <= m < 60):
            return float("nan")
        return float(h * 60 + m)
    except (ValueError, AttributeError):
        return float("nan")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip()).rename(columns=RENAME)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Time parsing + derived durations. Works with or without a label column."""
    df = df.copy()
    for c in TIME_COLS:
        df[c + "_min"] = df[c].map(parse_time)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["keeper_presence_min"] = df["keeper_departure_time_min"] - df["keeper_arrival_time_min"]
    df["meal_duration_min"] = df["meal_end_time_min"] - df["meal_start_time_min"]
    df["keeper_arrival_to_meal_start_min"] = df["meal_start_time_min"] - df["keeper_arrival_time_min"]
    # data-quality flag only (NOT a model feature)
    df["meal_ended_after_keeper_left"] = (df["meal_end_time_min"] > df["keeper_departure_time_min"]).astype(int)
    df["group_key"] = df["date_parsed"].dt.strftime("%Y-%m-%d")
    return df


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise + engineer a raw-schema table (used by predict.py on new data)."""
    return engineer(standardise_columns(df_raw))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate labels, drop unusable rows (counted, never imputed) and add the 0/1 target."""
    if "stress_label" not in df.columns:
        raise ValueError("stress_label column is required for training/evaluation")
    before = len(df)
    bad = set(df["stress_label"].dropna().unique()) - set(LABELS)
    if bad:
        raise ValueError(f"Unexpected stress_label values: {bad}")
    df = df.dropna(subset=["stress_label"]).copy()
    df["target"] = df["stress_label"].map(LABELS).astype(int)
    key_cols = ["group_key", "keeper_presence_min", "meal_duration_min", "keeper_arrival_to_meal_start_min"]
    df = df.dropna(subset=key_cols)
    dups = int(df.duplicated().sum())
    if dups:
        print(f"  warning: {dups} duplicate rows found (kept - verify against field sheets)")
    print(f"clean: kept {len(df)}/{before} rows")
    return df.reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    print(f"rows={len(df)}  observation days={df['group_key'].nunique()}  "
          f"{df['group_key'].min()} -> {df['group_key'].max()}")
    print("class distribution:\n", df.groupby("species")["stress_label"].value_counts().to_string())
    n_flag = int(df["meal_ended_after_keeper_left"].sum())
    print(f"data-quality: {n_flag} sessions where the meal ended after the keeper left "
          f"(plausible, but check against field sheets)")
    print(f"data-quality: {(df['keeper_proximity_min'] > df['keeper_presence_min']).sum()} sessions with "
          f"proximity > keeper presence")


def run(in_path=None, out_path=config.MODEL_READY) -> pd.DataFrame:
    in_path = in_path or config.default_input()
    print(f"reading {in_path}")
    df = prepare(pd.read_csv(in_path))
    df = clean(df)
    report(df)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
