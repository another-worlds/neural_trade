"""Seconds. The next-open variant never uses a price before the next bar's open for entry.
usage: python -m pytest -q -p no:cacheprovider runs/tactical/nextopen/test_nextopen.py"""
import os, sys, types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); GEOM = os.path.join(os.path.dirname(HERE), "geom2")
sys.path.insert(0, HERE); sys.path.insert(0, GEOM)
import nolib  # noqa: E402


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 3e-3, n)))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, 5e-4, n))       # open differs from the previous close (bounce)
    h = np.maximum(o, c) * 1.001; l = np.minimum(o, c) * 0.999
    return np.stack([o, h, l, c, rng.random(n)], 1)


def test_entry_price_is_the_next_bars_open_and_ignores_every_earlier_price():
    A = _bars(); anch = np.arange(60, 380, 7); H = 15
    r = nolib.label_return(A, anch, H, "nextopen")
    assert np.allclose(nolib.entry_next_open(A, anch), A[anch, 0])
    assert np.allclose(r, A[anch + H - 1, 3] / A[anch, 0] - 1)
    # scramble all prices before the next open, anchor by anchor
    for i in anch[:10]:
        B = A.copy(); B[:i] *= 1.37               # bars 0..i-1 (the signal bar i-1 included)
        assert np.allclose(nolib.label_return(B, np.array([i]), H, "nextopen"), nolib.label_return(A, np.array([i]), H, "nextopen"))
        assert not np.allclose(nolib.label_return(B, np.array([i]), H, "close"), nolib.label_return(A, np.array([i]), H, "close"))


def test_close_entry_is_the_recorded_convention_and_differs_by_the_gap():
    A = _bars(); anch = np.arange(60, 380, 7); H = 15
    rc = nolib.label_return(A, anch, H, "close"); rn = nolib.label_return(A, anch, H, "nextopen")
    assert np.allclose(rc, A[anch + H - 1, 3] / A[anch - 1, 3] - 1)
    assert not np.allclose(rc, rn)
    # same exit price: the two returns differ only through the entry price
    assert np.allclose((1 + rc) * A[anch - 1, 3], (1 + rn) * A[anch, 0])


def test_barrier_labels_use_the_next_open_for_entry_and_nothing_earlier():
    import meta_nextopen as M
    A = _bars(1500, 1); idx = pd.date_range("2024-01-01", periods=len(A), freq="min")
    S = dict(M.SETUP["1min"]); M.SETUP["synth"] = S
    M.bars = lambda kind: pd.DataFrame(A, index=idx, columns=list("ohlcv"))
    T = S["T"]; D, A_, anch = M.labels("synth", T, 1.0, 1.0)
    assert np.allclose(A_, A)
    assert np.allclose(D["rT"], A[anch + T - 1, 3] / A[anch, 0] - 1)
    # pl at a barrier is +-barrier width (unchanged), i.e. relative to the entry price c0 = A[anch, 0]; the bar anch-1 only feeds the window features
    i = 300; k = int(np.where(anch == i)[0][0] if (anch == i).any() else 0); i = anch[k]
    B = A.copy(); B[i - 1, 3] *= 1.05                    # the signal bar's close moves; sigma (window) changes, so compare against the reference formula
    M.bars = lambda kind: pd.DataFrame(B, index=idx, columns=list("ohlcv"))
    D2, _, anch2 = M.labels("synth", T, 1.0, 1.0)
    assert np.isclose(D2["rT"][k], B[i + T - 1, 3] / B[i, 0] - 1) and np.isclose(D2["rT"][k], D["rT"][k])


def test_the_two_meta_modules_differ_only_in_the_entry_price_line():
    a = open(os.path.join(GEOM, "meta.py"), encoding="utf-8").read().splitlines()
    b = open(os.path.join(GEOM, "meta_nextopen.py"), encoding="utf-8").read().splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(diff) == 1 and "A[anch - 1, 3]" in diff[0][0] and "A[anch, 0]" in diff[0][1]
