"""Seconds. FZ0 gradient vs finite differences, FZ0 is minimised by the true (VaR, ES), the fit recovers a known scale law."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import varlib as V  # noqa: E402


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0); X = rng.normal(size=(500, 3)); y = rng.standard_t(5, 500) * np.exp(0.2 * X[:, 0])
    th = rng.normal(0, 0.1, 2 + 6); th[0] = 0.5; th[4] = 0.0
    f, g = V.objective(th, X, y, 0.05, 3)
    for j in range(len(th)):
        e = np.zeros_like(th); e[j] = 1e-6
        # the indicator makes the loss piecewise smooth: with 500 rows a 1e-6 step crosses no kink almost surely
        num = (V.objective(th + e, X, y, 0.05, 3)[0] - V.objective(th - e, X, y, 0.05, 3)[0]) / 2e-6
        assert abs(num - g[j]) < 1e-4 * max(1, abs(g[j])), j


def test_true_var_es_minimise_expected_fz0():
    rng = np.random.default_rng(1); y = rng.normal(size=400000); a = 0.05
    from scipy.stats import norm
    v = norm.ppf(a); e = -norm.pdf(v) / a
    base = V.fz0(y, v, e, a).mean()
    for dv, de in ((0.1, 0), (-0.1, 0), (0, 0.15), (0, -0.15)):
        assert V.fz0(y, v + dv, e + de, a).mean() > base


def test_fit_recovers_scale_law_and_fhs_matches_empirical():
    rng = np.random.default_rng(2); n = 80000; x = rng.normal(size=(n, 1)); y = rng.standard_t(6, n) * np.exp(0.3 * x[:, 0])
    a = 0.05; q, es = V.fhs(y, a); th = V.fit_fz0(x, y, a, q, es)
    assert abs(th[1] - 0.3) < 0.06                       # log-scale coefficient on x
    assert V.fz0(y, *V.predict(th, x), a).mean() < V.fz0(y, q, es, a).mean() - 0.01
    assert abs((y <= q).mean() - a) < 1e-3


def test_ordering_es_below_var():
    rng = np.random.default_rng(3); x = rng.normal(size=(5000, 2)); y = rng.standard_t(5, 5000)
    q, es = V.fhs(y, 0.025); th = V.fit_fz0(x, y, 0.025, q, es); v, e = V.predict(th, x)
    assert (v < 0).all() and (e < v).all()
