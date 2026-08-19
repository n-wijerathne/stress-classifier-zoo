"""
baseline_model.py

The baseline defined in Section 2.3 of the methodology: a Random Forest
trained ONLY on general movement/trajectory features (no feeding-schedule
or keeper-routine variables). Used as the comparison point for Branch A
and Branch B.
"""

from sklearn.ensemble import RandomForestClassifier


def build_baseline_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
