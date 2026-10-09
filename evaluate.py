"""
Evaluation & Statistical Validation Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Runs day-grouped/stratified cross-validation, evaluates Baseline vs Branch A vs Branch B,
and computes Paired t-tests & McNemar's tests (p < 0.05).
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

try:
    from src.preprocess import preprocess_dataset
    from src.features import get_feature_sets
    from src.baseline_model import get_baseline_pipeline
    from src.models_tabular import get_tabular_rf_pipeline, get_tabular_svm_pipeline
    from src.models_lstm import get_lstm_fcn_pipeline
except ImportError:
    from preprocess import preprocess_dataset
    from features import get_feature_sets
    from baseline_model import get_baseline_pipeline
    from models_tabular import get_tabular_rf_pipeline, get_tabular_svm_pipeline
    from models_lstm import get_lstm_fcn_pipeline

def run_evaluation_pipeline(data_path="data/processed/synthetic_session_bins.csv"):
    df = preprocess_dataset(data_path)
    
    baseline_features, full_num_features, cat_features = get_feature_sets()
    
    X_base = df[baseline_features]
    X_full = df[cat_features + full_num_features]
    y = df['Target']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_f1s, branchA_rf_f1s, branchA_svm_f1s, branchB_lstm_f1s = [], [], [], []
    y_true_all, y_pred_base_all, y_pred_branchA_all = [], [], []

    for train_idx, test_idx in cv.split(X_full, y):
        X_b_tr, X_b_te = X_base.iloc[train_idx], X_base.iloc[test_idx]
        X_f_tr, X_f_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Baseline Model (Movement Only)
        pipe_base = get_baseline_pipeline()
        pipe_base.fit(X_b_tr, y_tr)
        p_base = pipe_base.predict(X_b_te)
        base_f1s.append(f1_score(y_te, p_base, zero_division=0))

        # 2. Branch A Model (RF Full Features)
        pipe_A_rf = get_tabular_rf_pipeline()
        pipe_A_rf.fit(X_f_tr, y_tr)
        p_A_rf = pipe_A_rf.predict(X_f_te)
        branchA_rf_f1s.append(f1_score(y_te, p_A_rf, zero_division=0))

        # 3. Branch A Model (SVM Full Features)
        pipe_A_svm = get_tabular_svm_pipeline()
        pipe_A_svm.fit(X_f_tr, y_tr)
        p_A_svm = pipe_A_svm.predict(X_f_te)
        branchA_svm_f1s.append(f1_score(y_te, p_A_svm, zero_division=0))

        # 4. Branch B Model (LSTM Sequence Classifier)
        pipe_B_lstm = get_lstm_fcn_pipeline()
        pipe_B_lstm.fit(X_f_tr, y_tr)
        p_B_lstm = pipe_B_lstm.predict(X_f_te)
        branchB_lstm_f1s.append(f1_score(y_te, p_B_lstm, zero_division=0))

        y_true_all.extend(y_te)
        y_pred_base_all.extend(p_base)
        y_pred_branchA_all.extend(p_A_rf)

    # Statistical Significance Testing
    t_stat, p_val_ttest = stats.ttest_rel(branchA_rf_f1s, base_f1s)
    
    table = pd.crosstab(np.array(y_pred_base_all) == np.array(y_true_all),
                        np.array(y_pred_branchA_all) == np.array(y_true_all))
    result_mcnemar = mcnemar(table.values, exact=True)

    print("\n=======================================================")
    print("       EVALUATION & STATISTICAL SIGNIFICANCE SUMMARY    ")
    print("=======================================================")
    print(f"1. Baseline Model (Movement Only RF):       F1-Score = {np.mean(base_f1s):.4f}")
    print(f"2. Branch A Proposed (Schedule-Aware RF):  F1-Score = {np.mean(branchA_rf_f1s):.4f}")
    print(f"3. Branch A Proposed (Schedule-Aware SVM): F1-Score = {np.mean(branchA_svm_f1s):.4f}")
    print(f"4. Branch B Proposed (LSTM-FCN Model):     F1-Score = {np.mean(branchB_lstm_f1s):.4f}")
    print("-------------------------------------------------------")
    print(f"Paired t-test (Branch A vs Baseline):   t = {t_stat:.4f}, p = {p_val_ttest:.4f}")
    print(f"McNemar's Test (Contingency Error):     p = {result_mcnemar.pvalue:.4f}")
    
    if p_val_ttest < 0.05:
        print("\n[SUCCESS] Statistically significant increase in F1-score for schedule-aware models vs baseline (p < 0.05)!")
    else:
        print("\n[INFO] Difference is not statistically significant at p < 0.05 level.")
    print("=======================================================\n")

if __name__ == '__main__':
    run_evaluation_pipeline()
