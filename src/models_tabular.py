"""
models_tabular.py

Branch A from Section 2.2 of the methodology: Random Forest and SVM
(RBF kernel), trained on the feeding-schedule / keeper-routine feature
set (see features.TABULAR_FEATURES).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(random_state: int = 42) -> Pipeline:
    # SVM is sensitive to feature scale, so it is wrapped in a pipeline
    # with a StandardScaler fitted only on the training fold (see evaluate.py).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                     class_weight="balanced", probability=True,
                     random_state=random_state)),
    ])
