"""Causality and shape tests for geomlib. usage: python -m pytest runs/tactical/geom2/test_geomlib.py -q -p no:cacheprovider"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import geomlib as G  # noqa: E402


def make(n=6, L=80, seed=0):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 0.1, (n, L)), 1)
    o = c + r.normal(0, 0.02, c.shape); h = np.maximum(o, c) + np.abs(r.normal(0, 0.03, c.shape)); l = np.minimum(o, c) - np.abs(r.normal(0, 0.03, c.shape))
    v = np.abs(r.normal(10, 2, c.shape))
    return np.stack([o, h, l, c, v], -1)


def test_shape_and_names():
    W = make(); F = G.geom_rich(W[:, :60])
    assert F.shape == (6, len(G.geom_names(None, (3, 5, 10, 20)))) and np.isfinite(F).all()
    assert G.geom_base5(W[:, :60]).shape[1] == len(G.geom_names(G.BASE5, (5,)))


def test_channels_are_causal():
    """A channel value at bar t computed from the prefix equals the one from the longer window (so no later bar enters)."""
    W = make(); full = G.channels(W)
    for t in (45, 59, 70):
        pre = G.channels(W[:, :t + 1])
        for name in full:
            np.testing.assert_allclose(pre[name][0][:, -1], full[name][0][:, t], rtol=1e-9, atol=1e-9, err_msg=name)


def test_features_ignore_bars_after_the_window():
    W = make(); F1 = G.geom_rich(W[:, :60]); W2 = W.copy(); W2[:, 60:] = 1e6
    np.testing.assert_array_equal(F1, G.geom_rich(W2[:, :60]))


def test_rows_are_independent():
    W = make(); F = G.geom_rich(W[:, :60]); W2 = W.copy(); W2[2] = make(1, 80, 9)[0]
    F2 = G.geom_rich(W2[:, :60]); keep = [0, 1, 3, 4, 5]
    np.testing.assert_array_equal(F[keep], F2[keep])


def test_chunking_does_not_change_values():
    W = make(20)[:, :60]; old = G.CHUNK
    try:
        G.CHUNK = 7; a = G.geom_rich(W)
    finally:
        G.CHUNK = old
    np.testing.assert_array_equal(a, G.geom_rich(W))
