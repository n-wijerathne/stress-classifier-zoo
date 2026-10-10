"""
evaluate.py - Methodology Section 4 (Evaluation Plan), run on the real data.

  * day-grouped test hold-out (~14%), never used for model selection
  * stratified, day-grouped 5-fold CV on the remaining development days
  * F1 (primary), precision, recall, AUC-ROC - mean +/- std across folds
  * baseline vs proposed: Shapiro-Wilk -> paired t-test or Wilcoxon; plus 10x repeated CV with the
    Nadeau-Bengio corrected resampled t-test (more reliable than 5 folds alone)
  * McNemar's test on the held-out test days
  * ablation over feature sets and out-of-fold permutation importance
Outputs go to results/.   Run:  python src/evaluate.py
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.contingency_tables import mcnemar

import config
import preprocess
from features import FEATURE_SETS, columns_for, get_xy
from models_tabular import MODEL_NAMES, build_pipeline

warnings.filterwarnings("ignore")


def split_dev_test(df: pd.DataFrame, seed: int = config.SEED):
    """Hold out whole observation days (stratified by label) as the final test set."""
    sgkf = StratifiedGroupKFold(n_splits=config.TEST_SPLITS, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, df["target"], df["group_key"]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _metrics(y_true, pred, prob):
    return dict(f1=f1_score(y_true, pred, zero_division=0),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0),
                auc_roc=roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan"))


def cv_scores(model_name: str, set_name: str, data: pd.DataFrame, seed: int = config.SEED,
              importance: bool = False):
    """Stratified day-grouped k-fold CV. Scaling/encoding are fitted inside each training fold."""
    num, cat = columns_for(set_name)
    X, y, g = get_xy(data, set_name)
    cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    rows, imps = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, g)):
        assert not set(g.iloc[tr]) & set(g.iloc[te]), "a day leaked across train/test"
        pipe = build_pipeline(model_name, num, cat, seed).fit(X.iloc[tr], y.iloc[tr])
        pred, prob = pipe.predict(X.iloc[te]), pipe.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": fold, "n_test": len(te), "n_train": len(tr), **_metrics(y.iloc[te], pred, prob)})
        if importance:
            r = permutation_importance(pipe, X.iloc[te], y.iloc[te], scoring="f1", n_repeats=20,
                                       random_state=seed)
            imps.append(r.importances_mean)
    out = pd.DataFrame(rows)
    if importance:
        return out, pd.DataFrame(imps, columns=num + cat)
    return out


def oof_predictions(model_name: str, set_name: str, data: pd.DataFrame, seed: int = config.SEED):
    """Out-of-fold predictions (every dev row predicted by a model that never saw its day)."""
    num, cat = columns_for(set_name)
    X, y, g = get_xy(data, set_name)
    cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    out = data.copy()
    out["prob"] = np.nan
    for tr, te in cv.split(X, y, g):
        pipe = build_pipeline(model_name, num, cat, seed).fit(X.iloc[tr], y.iloc[tr])
        out.loc[out.index[te], "prob"] = pipe.predict_proba(X.iloc[te])[:, 1]
    out["pred"] = (out["prob"] >= 0.5).astype(int)
    return out


def error_analysis(oof: pd.DataFrame) -> pd.DataFrame:
    """Where does the proposed model miss stress? (feeds the Failure Analysis later)"""
    pos = oof[oof.target == 1]
    rows = []
    for col in ["species", "session", "enclosure"]:
        for val, grp in pos.groupby(col):
            rows.append({"group": f"{col}={val}", "stress_sessions": len(grp),
                         "missed (false negatives)": int((grp.pred == 0).sum()),
                         "miss_rate": round(float((grp.pred == 0).mean()), 3),
                         "mean_time_to_keeper_arrival_min": round(float(grp.time_to_keeper_arrival_min.mean()), 1)})
    fn, tp = pos[pos.pred == 0], pos[pos.pred == 1]
    rows.append({"group": "ALL missed stress sessions", "stress_sessions": len(pos),
                 "missed (false negatives)": len(fn), "miss_rate": round(len(fn) / len(pos), 3),
                 "mean_time_to_keeper_arrival_min": round(float(fn.time_to_keeper_arrival_min.mean()), 1)})
    rows.append({"group": "ALL correctly found stress sessions", "stress_sessions": len(tp),
                 "missed (false negatives)": 0, "miss_rate": 0.0,
                 "mean_time_to_keeper_arrival_min": round(float(tp.time_to_keeper_arrival_min.mean()), 1)})
    fp = oof[(oof.target == 0) & (oof.pred == 1)]
    rows.append({"group": "false positives (non-stress sessions flagged; count in stress_sessions column)",
                 "stress_sessions": len(fp), "missed (false negatives)": 0, "miss_rate": float("nan"),
                 "mean_time_to_keeper_arrival_min": round(float(fp.time_to_keeper_arrival_min.mean()), 1) if len(fp) else float("nan")})
    return pd.DataFrame(rows)



def mean_std(df: pd.DataFrame, cols=("f1", "precision", "recall", "auc_roc")) -> dict:
    return {c: f"{df[c].mean():.3f} ± {df[c].std():.3f}" for c in cols}


def nadeau_bengio(diffs, n_test: int, n_train: int):
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1.0 / n + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=n - 1)


def paired_test(prop_f1, base_f1):
    diffs = np.asarray(prop_f1) - np.asarray(base_f1)
    p_norm = stats.shapiro(diffs).pvalue if len(set(diffs)) > 1 else 1.0
    if p_norm > 0.05:
        p = stats.ttest_rel(prop_f1, base_f1).pvalue
        name = "paired t-test"
    else:
        p = stats.wilcoxon(prop_f1, base_f1).pvalue
        name = "Wilcoxon signed-rank"
    return name, p_norm, p


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df = preprocess.run()
    dev, test = split_dev_test(df)
    print(f"\ndev days={dev.group_key.nunique()} rows={len(dev)} stress={dev.target.mean():.1%} | "
          f"test days={test.group_key.nunique()} rows={len(test)} stress={test.target.mean():.1%}")
    assert not set(dev.group_key) & set(test.group_key)
    summary = {"dev_rows": len(dev), "test_rows": len(test)}

    # 1. all models on the proposed feature set -------------------------------------------
    rows = []
    for m in MODEL_NAMES:
        rows.append({"model": m, **mean_std(cv_scores(m, "proposed", dev))})
    t1 = pd.DataFrame(rows)
    t1.to_csv(config.RESULTS_DIR / "cv_model_comparison.csv", index=False)
    print("\n[1] 5-fold day-grouped CV, proposed features (mean ± std over folds)\n", t1.to_string(index=False))

    # 2. baseline vs proposed, same model (RF), same folds --------------------------------
    base = cv_scores("RF", "baseline", dev)
    prop, imps = cv_scores("RF", "proposed", dev, importance=True)
    folds = pd.DataFrame({"fold": base["fold"], "baseline_f1": base["f1"], "proposed_f1": prop["f1"]})
    folds.to_csv(config.RESULTS_DIR / "baseline_vs_proposed_folds.csv", index=False)
    test_name, p_norm, p = paired_test(prop["f1"], base["f1"])
    print(f"\n[2] Baseline vs proposed (RF)\n    baseline F1 per fold {base['f1'].round(3).tolist()}"
          f"\n    proposed F1 per fold {prop['f1'].round(3).tolist()}"
          f"\n    Shapiro p={p_norm:.3f} -> {test_name}: p={p:.4f}"
          f"  ({'significant' if p < config.ALPHA else 'NOT significant'} at alpha={config.ALPHA})"
          f"\n    note: with 5 folds the smallest possible two-sided Wilcoxon p is 0.0625")

    # 3. repeated CV + Nadeau-Bengio ------------------------------------------------------
    all_diffs, wins, base_means, prop_means = [], 0, [], []
    n_tr = n_te = 0
    for s in range(config.N_REPEATS):
        b, pr = cv_scores("RF", "baseline", dev, seed=s), cv_scores("RF", "proposed", dev, seed=s)
        all_diffs.extend((pr["f1"] - b["f1"]).tolist())
        base_means.append(b["f1"].mean()); prop_means.append(pr["f1"].mean())
        wins += int(pr["f1"].mean() > b["f1"].mean())
        n_tr, n_te = int(b["n_train"].mean()), int(b["n_test"].mean())
    t_nb, p_nb = nadeau_bengio(all_diffs, n_te, n_tr)
    print(f"\n[3] {config.N_REPEATS}x repeated 5-fold CV: baseline F1 {np.mean(base_means):.3f}, "
          f"proposed F1 {np.mean(prop_means):.3f}, proposed better in {wins}/{config.N_REPEATS} repeats; "
          f"Nadeau-Bengio corrected t={t_nb:.2f}, p={p_nb:.2g}")
    summary.update(baseline_f1_repeated=float(np.mean(base_means)), proposed_f1_repeated=float(np.mean(prop_means)),
                   repeats_proposed_better=f"{wins}/{config.N_REPEATS}", nadeau_bengio_p=float(p_nb),
                   fold_test=test_name, fold_test_p=float(p))

    # 4. ablation over feature sets (repeated CV) ----------------------------------------
    abl = []
    for name in FEATURE_SETS:
        r = [cv_scores("RF", name, dev, seed=s) for s in range(config.N_REPEATS)]
        r = pd.concat(r)
        abl.append({"feature_set": name, **mean_std(r)})
    abl = pd.DataFrame(abl)
    abl.to_csv(config.RESULTS_DIR / "ablation_feature_sets.csv", index=False)
    print(f"\n[4] Ablation (RF, {config.N_REPEATS}x repeated 5-fold CV, mean ± std over all folds)\n",
          abl.to_string(index=False))

    # 5. held-out test days ---------------------------------------------------------------
    res, preds = [], {}
    for label, set_name in [("baseline", "baseline"), ("proposed", "proposed")]:
        num, cat = columns_for(set_name)
        pipe = build_pipeline("RF", num, cat, config.SEED).fit(dev[num + cat], dev["target"])
        pred, prob = pipe.predict(test[num + cat]), pipe.predict_proba(test[num + cat])[:, 1]
        preds[label] = pred
        res.append({"model": label, **{k: round(v, 3) for k, v in _metrics(test["target"], pred, prob).items()}})
    t5 = pd.DataFrame(res)
    t5.to_csv(config.RESULTS_DIR / "test_set_results.csv", index=False)
    cb, cp = preds["baseline"] == test["target"].to_numpy(), preds["proposed"] == test["target"].to_numpy()
    table = [[int((cb & cp).sum()), int((cb & ~cp).sum())], [int((~cb & cp).sum()), int((~cb & ~cp).sum())]]
    p_mc = mcnemar(table, exact=True).pvalue
    print(f"\n[5] Held-out test days (n={len(test)}, {int(test.target.sum())} stress-positive)\n",
          t5.to_string(index=False), f"\n    McNemar exact p={p_mc:.4f}  (very small test set: low power)")
    summary["mcnemar_p"] = float(p_mc)

    # 6. permutation importance (out-of-fold) --------------------------------------------
    imp = imps.mean().sort_values()
    err = imps.std().reindex(imp.index)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, xerr=err.values, color="#2c5f8a", ecolor="#999")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean drop in F1 when feature is shuffled (out-of-fold, ± std over folds)")
    ax.set_title("Permutation importance - Random Forest, proposed features")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "permutation_importance.png", dpi=200); plt.close(fig)
    imp.sort_values(ascending=False).round(4).to_csv(config.RESULTS_DIR / "permutation_importance.csv", header=["mean_f1_drop"])
    print("\n[6] Permutation importance (top 5)\n", imp.sort_values(ascending=False).head(5).round(3).to_string())

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.boxplot([folds["baseline_f1"], folds["proposed_f1"]], labels=["Baseline", "Proposed"])
    for _, r in folds.iterrows():
        ax.plot([1, 2], [r["baseline_f1"], r["proposed_f1"]], color="#999", lw=0.8, marker="o", ms=3)
    ax.set_ylabel("F1 (stress-present)"); ax.set_title("Per-fold F1, day-grouped 5-fold CV")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "f1_baseline_vs_proposed.png", dpi=200); plt.close(fig)

    # 7. error analysis on out-of-fold predictions -----------------------------------------
    ea = error_analysis(oof_predictions("RF", "proposed", dev))
    ea.to_csv(config.RESULTS_DIR / "error_analysis.csv", index=False)
    print("\n[7] Error analysis (out-of-fold, proposed RF)\n", ea.to_string(index=False))

    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nresults written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
