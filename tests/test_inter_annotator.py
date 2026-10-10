import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import inter_annotator as ia

P, A = "Stress_Present", "Stress_Absent"


def test_perfect_agreement_gives_kappa_one():
    df = pd.DataFrame({"coder_a": [P, A, P, A, A, P], "coder_b": [P, A, P, A, A, P]})
    r = ia.agreement(df, n_boot=50)
    assert r["cohens_kappa"] == pytest.approx(1.0)
    assert r["percent_agreement"] == 1.0 and r["meets_target"]


def test_known_kappa_value():
    # 10 sessions, 8 agree: p_o=0.8, p_e=0.5 -> kappa = 0.6
    a = [P] * 5 + [A] * 5
    b = [P] * 4 + [A] + [A] * 4 + [P]
    r = ia.agreement(pd.DataFrame({"coder_a": a, "coder_b": b}), n_boot=50)
    assert r["cohens_kappa"] == pytest.approx(0.6)
    assert not r["meets_target"] and r["interpretation"] == "moderate"


def test_invalid_label_raises():
    with pytest.raises(ValueError):
        ia.agreement(pd.DataFrame({"coder_a": [P, "Maybe"], "coder_b": [P, A]}))
