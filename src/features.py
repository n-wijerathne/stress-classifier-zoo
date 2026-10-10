"""
features.py - feature groups and the scikit-learn preprocessor.

Feature groups
  MOVEMENT        general movement/trajectory proxies (what existing zoo-monitoring systems use)
  CONTEXT         species, enclosure, observation session (morning/afternoon)
  FEEDING_KEEPER  the study's feature domain: feeding schedule + keeper routine
  STEREOTYPY      rolling stereotypy rate (kept separate: it is behaviour-derived and may be
                  close to the label, so it is reported as an extended set, not the main model)

BASELINE = MOVEMENT + CONTEXT. It gets the same context columns as the proposed model so that
the ONLY difference between the two is the feeding/keeper features (no confounding by species,
enclosure or time of day).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MOVEMENT = ["activity_level", "location_changes", "boundary_time_min"]
CONTEXT = ["species", "enclosure", "session"]
FEEDING_KEEPER = [
    "inter_meal_interval_min",
    "diet_variety_score",
    "keeper_proximity_min",
    "time_to_keeper_arrival_min",
    "keeper_presence_min",
    "meal_duration_min",
    "keeper_arrival_to_meal_start_min",
]
STEREOTYPY = ["rolling_stereotypy_rate"]
LABEL_COL = "target"
GROUP_COL = "group_key"

# name -> (numeric columns, categorical columns)
FEATURE_SETS = {
    "movement_only":              (MOVEMENT, []),
    "baseline":                   (MOVEMENT, CONTEXT),
    "baseline+keeper_arrival":    (MOVEMENT + ["time_to_keeper_arrival_min"], CONTEXT),   # ablation
    "proposed":                   (MOVEMENT + FEEDING_KEEPER, CONTEXT),
    "proposed+stereotypy":        (MOVEMENT + FEEDING_KEEPER + STEREOTYPY, CONTEXT),
}


def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """z-score numeric columns, one-hot categorical ones. Always used inside a Pipeline so it is
    fitted on the training fold only (no leakage)."""
    steps = [("num", StandardScaler(), list(num_cols))]
    if cat_cols:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)))
    return ColumnTransformer(steps)


def columns_for(set_name: str):
    num, cat = FEATURE_SETS[set_name]
    return list(num), list(cat)


def get_xy(df: pd.DataFrame, set_name: str):
    num, cat = columns_for(set_name)
    return df[num + cat].copy(), df[LABEL_COL].copy(), df[GROUP_COL].copy()
