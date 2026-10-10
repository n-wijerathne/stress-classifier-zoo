"""
train_final.py - fit the final proposed model on ALL labelled sessions and save it.

Run only AFTER evaluate.py: the reported performance comes from the day-grouped CV / held-out
days, NOT from this model's predictions on its own training data.
Output: models/stress_rf_proposed.joblib + models/model_card.json
"""
import json
import platform

import joblib
import sklearn

import config
import preprocess
from features import columns_for
from models_tabular import build_pipeline


def main():
    df = preprocess.run()
    num, cat = columns_for("proposed")
    pipe = build_pipeline("RF", num, cat, config.SEED).fit(df[num + cat], df["target"])
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODELS_DIR / "stress_rf_proposed.joblib")
    card = {"model": "RandomForest (class_weight=balanced, 300 trees, min_samples_leaf=3)",
            "feature_set": "proposed", "numeric_features": num, "categorical_features": cat,
            "training_rows": len(df), "label": "target (1 = Stress_Present)",
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
            "note": "Pickles are sklearn-version specific; re-run `python src/train_final.py` if loading fails."}
    (config.MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print("saved", config.MODELS_DIR / "stress_rf_proposed.joblib")


if __name__ == "__main__":
    main()
