"""
inter_annotator.py - Cohen's kappa for the double-coded observation sessions (Methodology Section 1,
"Annotation Process": two researchers independently code ~10% of sessions; target kappa >= 0.70).

Input : data/processed/inter_annotator_labels.csv   (columns: session_id, coder_a, coder_b)
        - one row per double-coded session
        - coder_a / coder_b hold each researcher's independent label: Stress_Present / Stress_Absent
        (use inter_annotator_template.csv as the header; add REAL double-coded labels only)
Output: results/inter_annotator_agreement.json and results/inter_annotator_confusion.csv

Run:  python src/inter_annotator.py [path/to/labels.csv]
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import config

LABELS = ["Stress_Absent", "Stress_Present"]
DEFAULT_INPUT = config.DATA_DIR / "processed" / "inter_annotator_labels.csv"
TARGET_KAPPA = 0.70


def landis_koch(k: float) -> str:
    """Conventional verbal interpretation of kappa (Landis & Koch, 1977)."""
    if np.isnan(k):
        return "undefined"
    for upper, name in [(0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                        (0.60, "moderate"), (0.80, "substantial"), (1.01, "almost perfect")]:
        if k <= upper:
            return name
    return "almost perfect"


def agreement(df: pd.DataFrame, n_boot: int = 2000, seed: int = config.SEED) -> dict:
    for c in ("coder_a", "coder_b"):
        bad = set(df[c].dropna().unique()) - set(LABELS)
        if bad:
            raise ValueError(f"Unexpected labels in {c}: {bad}")
    df = df.dropna(subset=["coder_a", "coder_b"])
    if len(df) < 2:
        raise ValueError("Need at least 2 double-coded sessions")
    a, b = df["coder_a"].to_numpy(), df["coder_b"].to_numpy()
    kappa = cohen_kappa_score(a, b, labels=LABELS)

    rng = np.random.default_rng(seed)
    boots = []
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # degenerate bootstrap resamples (one label only)
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        k = cohen_kappa_score(a[idx], b[idx], labels=LABELS)
        if not np.isnan(k):
            boots.append(k)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    cm = confusion_matrix(a, b, labels=LABELS)
    return {
        "n_sessions": int(len(df)),
        "percent_agreement": float((a == b).mean()),
        "cohens_kappa": float(kappa),
        "kappa_95ci_bootstrap": [float(lo), float(hi)],
        "interpretation": landis_koch(kappa),
        "target_kappa": TARGET_KAPPA,
        "meets_target": bool(kappa >= TARGET_KAPPA),
        "confusion_matrix": pd.DataFrame(cm, index=[f"A:{l}" for l in LABELS],
                                         columns=[f"B:{l}" for l in LABELS]),
    }


def main(path=DEFAULT_INPUT):
    df = pd.read_csv(path)
    res = agreement(df)
    cm = res.pop("confusion_matrix")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "inter_annotator_agreement.json").write_text(json.dumps(res, indent=2))
    cm.to_csv(config.RESULTS_DIR / "inter_annotator_confusion.csv")
    print(json.dumps(res, indent=2))
    print(cm.to_string())
    print("\nTarget kappa >= 0.70:", "MET" if res["meets_target"] else "NOT MET - report honestly, discuss and re-train coders if needed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT)
