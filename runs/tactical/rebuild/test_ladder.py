"""Tests for the rebuild ladder. Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider runs/tactical/rebuild/test_ladder.py"""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def test_l1_matches_sklearn_on_synthetic_data():
    rs = np.random.RandomState(0); n, f = 3000, 6
    X = rs.randn(n, f).astype(np.float32); X[:, 1] += 0.7 * X[:, 0]          # a correlated pair
    w = np.array([0.8, -0.5, 0.3, 0.0, 0.0, 0.2]); y = (rs.rand(n) < 1 / (1 + np.exp(-(X @ w + 0.1)))).astype(np.float32)
    m = np.ones((n, 1), np.float32); m[rs.rand(n) < 0.1] = 0                   # some rows masked out
    keep = m[:, 0] > 0
    ref = LogisticRegression(C=0.1, max_iter=1000).fit(X[keep], y[keep].astype(int))
    model, lin = ladder.make_model(f, k=1)
    ladder.fit(model, lin, [X], y[:, None], m, ladder.lam_vec(0.1, m), lr=ladder.LIN_LR, epochs=ladder.LIN_STEPS)
    assert np.allclose(lin.kernel.numpy()[:, 0], ref.coef_[0], atol=2e-3)
    assert abs(float(lin.bias.numpy()[0]) - ref.intercept_[0]) < 2e-3
    p = ladder.predict(model, [X])[:, 0]
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, ref.predict_proba(X)[:, 1])) < 1e-4


def test_residual_branches_start_at_the_linear_model():
    for kw in ({"mlp": True}, {"seq": True}):
        model, lin = ladder.make_model(4, **kw)
        lin.set_weights([np.ones((4, 3), np.float32), np.zeros(3, np.float32)])
        x = [np.random.randn(5, 4).astype(np.float32)] + ([np.random.randn(5, 60, 5).astype(np.float32)] if kw.get("seq") else [])
        assert np.allclose(model(x).numpy(), x[0].sum(1, keepdims=True).repeat(3, 1), atol=1e-6)


CACHE = sorted(glob.glob(ladder.lab.CACHE))
# first cached slice, tb7 / logreg C=0.1 (the lab before the refactor: results.jsonl of 2026-10-09, identical after it)
EXPECTED_FIRST_SLICE_AUC3 = 0.5609


@pytest.mark.skipif(not CACHE, reason="lab cache not available")
def test_lab_evaluation_is_the_same_function_on_one_cached_slice():
    S = ladder.load_slice(CACHE[0], "tb7"); P = ladder.step_L0(S)
    r = ladder.lab.evaluate_slice(S["D"], P, np.random.default_rng(0))
    assert round(r["auc3"], 4) == EXPECTED_FIRST_SLICE_AUC3
    assert r["ll_h1"] < r["ll_const_h1"] + 0.01 and set(r) >= {"auc_h0", "brier_h1", "hon10_hit", "hon10_null95", "hon5_bps"}
