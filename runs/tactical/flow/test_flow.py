"""Seconds-long tests for the flow study (SPEC.md). run: python -m pytest -q -p no:cacheprovider runs/tactical/flow/test_flow.py"""
import os, sys
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import flowlib as FL      # noqa: E402
import flowrun as FR      # noqa: E402


def synth(n, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, 2e-4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    v = rng.gamma(2.0, 5.0, n)
    v[100:110] = 0.0; l[200] = h[200] = c[200]                      # zero-volume and flat bars
    return o, h, l, c, v


@pytest.mark.parametrize("bar_min,n,cut", [(1, 7000, 4500), (5, 3000, 1800)])
def test_features_at_bar_t_do_not_see_later_bars(bar_min, n, cut):
    o, h, l, c, v = synth(n)
    F0 = FL.flow_features(o, h, l, c, v, bar_min)
    assert np.isfinite(F0[-1]).all() and F0.shape == (n, 9)
    rng = np.random.default_rng(9)
    o2, h2, l2, c2, v2 = (x.copy() for x in (o, h, l, c, v))
    for x in (o2, h2, l2, c2):
        x[cut + 1:] *= rng.uniform(0.5, 2.0, n - cut - 1)
    v2[cut + 1:] = rng.gamma(2.0, 50.0, n - cut - 1)
    F1 = FL.flow_features(o2, h2, l2, c2, v2, bar_min)
    assert not np.allclose(np.nan_to_num(F0[cut + 1:]), np.nan_to_num(F1[cut + 1:]))          # the change does matter later
    np.testing.assert_array_equal(F0[:cut + 1], F1[:cut + 1])                                  # and never earlier (exact)


def test_next_open_label():
    o, h, l, c, v = synth(400)
    A = np.stack([o, h, l, c, v], 1)
    a = FL.anchors(len(A), 15, 10)
    assert a.min() == FL.L and (a + 14 < len(A)).all()
    r = FL.next_open_return(A, a, 15)
    np.testing.assert_allclose(r, c[a + 14] / o[a] - 1)
    assert not np.allclose(r, c[a + 14] / c[a - 1] - 1)                    # not the close-entry label
    r5 = FL.next_open_return(A, FL.anchors(len(A), 3, 2), 3)
    a5 = FL.anchors(len(A), 3, 2)
    np.testing.assert_allclose(r5, c[a5 + 2] / o[a5] - 1)


def make_d(n_rows=60000, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-06-01", periods=n_rows, freq="10min")
    X = rng.normal(size=(n_rows, 3)).astype(np.float32)
    r = 1e-3 * (0.3 * X[:, 0] + rng.normal(size=n_rows))
    return {"tb7": X[:, :2], "flow": X[:, 2:], "t": t.values.astype("datetime64[ns]").astype(np.int64), "r": r}, t


def test_selection_refuses_late_months_and_ignores_them():
    d, t = make_d()
    assert t[-1] > pd.Timestamp("2024-07-01")
    with pytest.raises(AssertionError):
        FR.run_cell(d, "1m", "tb7+flow", "logreg", 60, "2024-05", "2024-07", FR.LAST_SEL_MONTH)
    a = FR.run_cell(d, "1m", "tb7+flow", "logreg", 60, "2024-04", "2024-06", FR.LAST_SEL_MONTH)
    assert len(a) == 3
    d2 = {k: v.copy() for k, v in d.items()}
    late = np.asarray(t >= pd.Timestamp("2024-07-01"))
    d2["r"][late] = np.random.default_rng(5).normal(size=late.sum())
    d2["tb7"][late] = 1e6; d2["flow"][late] = -1e6
    b = FR.run_cell(d2, "1m", "tb7+flow", "logreg", 60, "2024-04", "2024-06", FR.LAST_SEL_MONTH)
    assert a == b                                                          # not a number moved
    assert FR.LAST_SEL_MONTH == pd.Period("2024-06", "M")
    assert FR.SEL[1] == "2024-06" and FR.STAB_SEL[1] == "2024-06"


def test_look_refuses_without_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "HERE", str(tmp_path))
    with pytest.raises(SystemExit):
        FR.run("look")
