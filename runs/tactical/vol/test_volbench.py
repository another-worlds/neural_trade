"""Seconds-fast checks of volbench.py: GARCH recovers known parameters; no fitted quantity or forecast reads a later bar.
run: python -m pytest -q -p no:cacheprovider runs/tactical/vol/test_volbench.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volbench as vb  # noqa: E402


def simulate_garch(n, om, al, be, seed=0):
    rng = np.random.default_rng(seed); e = rng.standard_normal(n); r = np.empty(n); h = om / (1 - al - be)
    for t in range(n):
        r[t] = math_sqrt(h) * e[t]; h = om + al * r[t] ** 2 + be * h
    return r


def math_sqrt(x):
    return float(np.sqrt(x))


def test_garch_recovers_parameters_within_10_percent():
    """Four simulated series of 200k bars (omega alone has a ~7% sampling error on one series: seed 0 gives +11%); the mean of the four
    fits is within 10% of the truth for all three parameters, and alpha and beta are within 10% in every single fit."""
    om, al, be = 0.1, 0.1, 0.8
    fits = np.array([vb.garch_fit(simulate_garch(200_000, om, al, be, seed)) for seed in range(4)])
    for i, (name, true) in enumerate((("omega", om), ("alpha", al), ("beta", be))):
        assert abs(fits[:, i].mean() / true - 1) < 0.10, (name, fits[:, i], true)
    assert np.all(np.abs(fits[:, 1] / al - 1) < 0.10) and np.all(np.abs(fits[:, 2] / be - 1) < 0.10), fits


def test_garch_filter_matches_the_recursion_and_the_horizon_formula():
    rng = np.random.default_rng(1); r = rng.standard_normal(500) * 0.01; om, al, be = 1e-6, 0.07, 0.9
    u = vb.garch_filter(r, om, al, be, 1e-4); ref = np.empty(500); prev = 1e-4
    for t in range(500):
        prev = om + al * r[t] ** 2 + be * prev; ref[t] = prev
    assert np.allclose(u, ref, rtol=1e-9)
    h = 7; v = ref[-1]; tot = 0.0
    for k in range(h):
        tot += v; v = om + (al + be) * v
    assert np.isclose(vb.garch_horizon_var(ref[-1], om, al, be, h), tot, rtol=1e-9)


def _toy(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    lr = rng.standard_normal(n) * 5e-4 * (1 + 0.5 * np.sin(np.arange(n) / 300)); lr[0] = 0.0
    return lr


def test_trailing_variance_and_ewma_are_causal():
    lr = _toy(); cs = vb.cum_sq(lr); t0 = 3000
    lr2 = lr.copy(); lr2[t0 + 1:] = 0.123                          # change every bar after t0
    cs2 = vb.cum_sq(lr2)
    for k in vb.TRAILING:
        assert vb.trailing_var(cs, k)[t0] == vb.trailing_var(cs2, k)[t0]
    e1 = vb.ewma_var(lr, 0.98, 1e-7); e2 = vb.ewma_var(lr2, 0.98, 1e-7)
    assert e1[t0] == e2[t0]
    u1 = vb.garch_filter(lr, 1e-8, 0.05, 0.93, 1e-7); u2 = vb.garch_filter(lr2, 1e-8, 0.05, 0.93, 1e-7)
    assert u1[t0] == u2[t0]


def test_minute_forecasts_use_nothing_from_the_val_block_when_fitting():
    """Fitted quantities (EWMA lambda, GARCH parameters, HAR coefficients, scales) must not change when every bar from the val block's
    first window start on is replaced; forecasts at a val bar t must not change when bars after t are replaced."""
    n_tot, vs, nva, n = 9000, 6000, 800, 3000
    rng = np.random.default_rng(3)
    lr = rng.standard_normal(n_tot) * 4e-4 * (1 + 0.6 * np.sin(np.arange(n_tot) / 500)); lr[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(lr)); tod = (np.arange(n_tot) % 1440).astype(float)
    s0 = vs - vb.GAPB - (n - 1)

    def run(close_, lr_):
        cs = vb.cum_sq(lr_); t_tr = s0 + vb.WIN - 1 + np.arange(n)
        rtr = np.stack([close_[t_tr + h] / close_[t_tr] - 1 for h in vb.HZ], 1)
        assert t_tr[-1] + max(vb.HZ) < vs
        return vb.minute_forecasts(close_, tod, lr_, cs, vs, nva, s0, n, rtr, None, None)

    ov, ot, info = run(close, lr)
    lr2 = lr.copy(); lr2[vs:] = rng.standard_normal(n_tot - vs) * 3e-3
    close2 = 100.0 * np.exp(np.cumsum(lr2))
    assert np.array_equal(close2[:vs], close[:vs])                  # identical history before the val block start
    ov2, ot2, info2 = run(close2, lr2)
    assert info["garch"] == info2["garch"] and info["ewma_lambda"] == info2["ewma_lambda"]
    for k in ot:
        assert np.array_equal(ot[k][1], ot2[k][1]), k                # train-span forecasts (and so the scales c) are identical
    # a val forecast at t is unchanged when only bars after t change
    t_va = vs + vb.WIN - 1 + np.arange(nva); j = 300; t = t_va[j]
    lr3 = lr.copy(); lr3[t + 1:] = rng.standard_normal(n_tot - t - 1) * 3e-3; close3 = 100.0 * np.exp(np.cumsum(lr3))
    ov3, _, _ = run(close3, lr3)
    for k in ov:
        assert np.allclose(ov[k][1][: j + 1], ov3[k][1][: j + 1], rtol=1e-9), k
        assert np.allclose(ov[k][1][j], ov3[k][1][j], rtol=1e-9), k
