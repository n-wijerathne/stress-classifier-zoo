"""
evaluate.py

Implements Section 2.4 of the methodology:
  - Stratified, day-grouped 5-fold cross-validation
  - F1, precision, recall, AUC-ROC per fold
  - Paired significance test (baseline vs. proposed model) across folds

Run this after preprocess.py has produced data/processed/model_ready.csv.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from features import get_tabular_xy, get_baseline_xy
from baseline_model import build_baseline_model
from models_tabular import build_random_forest


def cross_validate(model_builder, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    n_splits: int = 5, random_state: int = 42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale continuous features using train-fold statistics only (no leakage).
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model = clone(model_builder()) if not hasattr(model_builder(), "fit") else model_builder()
        model = model_builder()
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds

        fold_scores.append({
            "fold": fold_i,
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "auc_roc": roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan"),
        })

    return pd.DataFrame(fold_scores)


def compare_models(df: pd.DataFrame):
    X_base, y_base, g_base = get_baseline_xy(df)
    X_prop, y_prop, g_prop = get_tabular_xy(df)

    print("Running baseline (movement/trajectory-only) cross-validation...")
    base_scores = cross_validate(build_baseline_model, X_base, y_base, g_base)

    print("Running proposed model (feeding-schedule + keeper-routine) cross-validation...")
    prop_scores = cross_validate(build_random_forest, X_prop, y_prop, g_prop)

    print("\nBaseline F1 per fold:", base_scores["f1"].tolist())
    print("Proposed F1 per fold:", prop_scores["f1"].tolist())

    # Normality check to choose paired t-test vs Wilcoxon signed-rank test,
    # per Section 2.4 of the methodology.
    diffs = prop_scores["f1"].to_numpy() - base_scores["f1"].to_numpy()
    _, p_normal = stats.shapiro(diffs)

    if p_normal > 0.05:
        stat, p_value = stats.ttest_rel(prop_scores["f1"], base_scores["f1"])
        test_used = "paired t-test"
    else:
        stat, p_value = stats.wilcoxon(prop_scores["f1"], base_scores["f1"])
        test_used = "Wilcoxon signed-rank test"

    print(f"\nSignificance test used: {test_used}")
    print(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Result: statistically significant difference (alpha = 0.05).")
    else:
        print("Result: NOT statistically significant at alpha = 0.05 "
              "(a valid, informative outcome per the falsifiability statement).")

    return base_scores, prop_scores, test_used, p_value


if __name__ == "__main__":
    df = pd.read_csv("data/processed/model_ready.csv")
    compare_models(df)
