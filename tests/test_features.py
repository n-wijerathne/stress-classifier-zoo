import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
import features
import pandas as pd
import preprocess


def _df():
    return preprocess.clean(preprocess.prepare(pd.read_csv(config.PUBLIC_INPUT)))


def test_baseline_contains_no_feeding_or_keeper_features():
    num, cat = features.columns_for("baseline")
    assert not set(num + cat) & set(features.FEEDING_KEEPER + features.STEREOTYPY)


def test_baseline_and_proposed_share_context_columns():
    assert features.columns_for("baseline")[1] == features.columns_for("proposed")[1]


def test_proposed_is_baseline_plus_feeding_keeper():
    b = set(sum(features.columns_for("baseline"), []))
    p = set(sum(features.columns_for("proposed"), []))
    assert p - b == set(features.FEEDING_KEEPER)


def test_get_xy_shapes_and_no_label_leak():
    X, y, g = features.get_xy(_df(), "proposed")
    assert len(X) == len(y) == len(g) == 190
    assert "target" not in X.columns and "stress_label" not in X.columns
