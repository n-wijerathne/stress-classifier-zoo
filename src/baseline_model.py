"""
baseline_model.py - the baseline from Methodology Section 3.

A Random Forest trained only on general movement/trajectory features (+ the same context columns
as the proposed model), with NO feeding-schedule or keeper-routine variables. It re-uses exactly
the same Random Forest settings as Branch A, so any difference in performance comes from the
feature set alone.
"""
from features import columns_for
from models_tabular import build_pipeline


def build_baseline_pipeline(seed: int = 42):
    num, cat = columns_for("baseline")
    return build_pipeline("RF", num, cat, seed)
