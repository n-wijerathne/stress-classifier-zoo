"""
predict.py - score new sessions with the saved model.

Usage:  python src/predict.py --input new_sessions.csv [--output predictions.csv] [--threshold 0.5]
`new_sessions.csv` must use the same columns as data/processed/animal_stress_data.csv
(the Stress_Label column is not needed).
"""
import argparse

import joblib
import pandas as pd

import config
import preprocess
from features import columns_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    pipe = joblib.load(config.MODELS_DIR / "stress_rf_proposed.joblib")
    df = preprocess.prepare(pd.read_csv(a.input))
    num, cat = columns_for("proposed")
    prob = pipe.predict_proba(df[num + cat])[:, 1]
    out = df[["date", "species", "enclosure", "session"]].copy()
    out["stress_probability"] = prob.round(3)
    out["prediction"] = ["Stress_Present" if p >= a.threshold else "Stress_Absent" for p in prob]
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved {a.output}")


if __name__ == "__main__":
    main()
