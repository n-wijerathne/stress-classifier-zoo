import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
import evaluate
import preprocess


def _df():
    return preprocess.clean(preprocess.prepare(pd.read_csv(config.PUBLIC_INPUT)))


def test_dev_and_test_share_no_observation_day():
    dev, test = evaluate.split_dev_test(_df())
    assert not set(dev.group_key) & set(test.group_key)
    assert len(dev) + len(test) == 190


def test_cv_runs_and_returns_valid_scores():
    dev, _ = evaluate.split_dev_test(_df())
    s = evaluate.cv_scores("RF", "baseline", dev)   # asserts internally that no day leaks between folds
    assert len(s) == config.N_SPLITS
    assert s["f1"].between(0, 1).all()


def test_nadeau_bengio_is_more_conservative_than_naive_t():
    from scipy import stats
    d = np.array([0.3, 0.35, 0.25, 0.4, 0.3, 0.28, 0.33, 0.31, 0.29, 0.36])
    _, p_nb = evaluate.nadeau_bengio(d, n_test=30, n_train=130)
    p_naive = stats.ttest_1samp(d, 0).pvalue
    assert p_nb >= p_naive
