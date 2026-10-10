"""
models_tabular.py - Branch A classifiers (tabular, session level).

Random Forest is the primary model (interpretable for keeper staff, robust on small noisy data);
SVM (RBF) is the second tabular classifier from the methodology. Gradient Boosting, Logistic
Regression and an MLP are added as comparison models. The MLP stands in for the "Branch B"
sequence model: the real dataset has one row per animal-session (190 rows), not 5-minute
sequences, so an LSTM-FCN cannot be trained on it.
Class imbalance is handled with class weights (SMOTE is not used - see README).
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from features import make_preprocessor

MODEL_NAMES = ["RF", "SVM", "GB", "LR", "MLP"]


def build_model(name: str, seed: int = 42):
    models = {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                   probability=True, random_state=seed),
        "GB": GradientBoostingClassifier(random_state=seed),
        "LR": LogisticRegression(class_weight="balanced", max_iter=1000),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000, random_state=seed),
    }
    return models[name]


def build_pipeline(name: str, num_cols, cat_cols, seed: int = 42) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)),
                     ("clf", build_model(name, seed))])
