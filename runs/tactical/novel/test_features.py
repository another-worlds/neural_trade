"""Tests of the N1-N7 features (seconds, CPU): causality, scale invariance, batch independence, known answers.
run: python -m pytest -q runs/tactical/novel/test_features.py -p no:cacheprovider"""
import os, sys
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402


def series(n=400, seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0007, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0007, n))); v = rng.gamma(2.0, 5.0, n)
    return np.stack([o, h, l, c, v], 1)


def windows(A):
    return np.ascontiguousarray(sliding_window_view(A, 60, axis=0).transpose(0, 2, 1))


@pytest.fixture(scope="module")
def base():
    A = series(); W = windows(A); return A, W, F.compute(W)


def test_shapes_and_finite(base):
    A, W, fe = base
    for f in F.FAMILIES:
        assert fe[f].shape == (len(W), len(F.NAMES[f])) and np.isfinite(fe[f]).all()


def test_causal_future_bars_do_not_change_a_window(base):
    A, W, fe = base
    i = 150; A2 = A.copy(); A2[i + 60:] *= np.random.default_rng(1).uniform(0.5, 2.0, A2[i + 60:].shape)
    fe2 = F.compute(windows(A2))
    for f in F.FAMILIES:
        np.testing.assert_allclose(fe2[f][:i + 1], fe[f][:i + 1], rtol=1e-6, atol=1e-7)


def test_batch_independent(base):
    A, W, fe = base
    one = F.compute(W[37:38]); part = F.compute(W[20:90], chunk=13)
    for f in F.FAMILIES:
        np.testing.assert_allclose(one[f][0], fe[f][37], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(part[f], fe[f][20:90], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("k", [0.001, 7.3, 4e5])
def test_price_scale_invariance(base, k):
    A, W, fe = base
    W64 = W.astype(np.float64); fe = F.compute(W64)       # float32 inputs would round differently after scaling
    W2 = W64.copy(); W2[..., :4] *= k
    fe2 = F.compute(W2)
    for f in F.FAMILIES:
        np.testing.assert_allclose(fe2[f], fe[f], rtol=2e-5, atol=2e-5, err_msg=f)


def test_volume_scale_invariance(base):
    A, W, fe = base
    W2 = W.copy(); W2[..., 4] *= 123.0
    np.testing.assert_allclose(F.compute(W2, ["N2"])["N2"], fe["N2"], rtol=1e-6, atol=1e-7)


def test_flat_window_is_finite_zeroish():
    W = np.ones((3, 60, 5)) * np.array([100.0, 100.0, 100.0, 100.0, 0.0])
    fe = F.compute(W)
    for f in F.FAMILIES:
        assert np.isfinite(fe[f]).all()
    for f in ("N7", "N5", "N2"):
        assert np.allclose(fe[f], 0, atol=1e-6)


# ------------------------------------------------------------ known answers
def test_circle_curvature_is_one_over_r_and_torsion_zero():
    for r in (0.5, 2.0):
        th = np.linspace(0, 2 * np.pi * 0.9, 400)
        P = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], 1)[None]
        kappa, tau, phi, arc, chord = F.frenet(P)
        np.testing.assert_allclose(kappa[0], 1 / r, rtol=1e-3)
        assert np.abs(tau).max() < 1e-6


def test_helix_curvature_and_torsion():
    r, b = 1.0, 0.5; th = np.linspace(0, 6 * np.pi, 3000)
    P = np.stack([r * np.cos(th), r * np.sin(th), b * th], 1)[None]
    kappa, tau, *_ = F.frenet(P)
    np.testing.assert_allclose(kappa[0], r / (r * r + b * b), rtol=1e-3)
    np.testing.assert_allclose(tau[0], b / (r * r + b * b), rtol=1e-2)       # right-handed helix: positive torsion
    P2 = P.copy(); P2[..., 2] *= -1
    assert F.frenet(P2)[1][0].mean() < 0                                      # mirror image: negative torsion


def test_straight_line_has_zero_curvature_torsion_and_turning():
    t = np.linspace(0, 1, 20); P = np.stack([t, 2 * t, -t], 1)[None]
    kappa, tau, phi, arc, chord = F.frenet(P)
    assert np.abs(kappa).max() < 1e-6 and np.abs(tau).max() < 1e-6 and phi.sum() < 1e-6
    np.testing.assert_allclose(arc, chord, rtol=1e-9)


def test_levy_area_of_a_loop():
    th = np.linspace(0, 2 * np.pi, 2000); X, Y = np.cos(th)[None], np.sin(th)[None]
    np.testing.assert_allclose(F.levy(X, Y)[0], np.pi / 8, rtol=1e-3)         # A = pi, TV_X = TV_Y = 4
    np.testing.assert_allclose(F.levy(Y, X)[0], -np.pi / 8, rtol=1e-3)        # clockwise: sign flips
    t = np.linspace(0, 1, 50)[None]
    assert abs(F.levy(t, 3 * t)[0]) < 1e-12                                   # collinear path: zero area
    assert abs(F.levy(t, t ** 2 * 5)[0]) <= 1.0


def test_levy_square_loop_exact():
    X = np.array([[0.0, 1, 1, 0, 0]]); Y = np.array([[0.0, 0, 1, 1, 0]])      # counter-clockwise unit square: A = 1
    np.testing.assert_allclose(F.levy(X, Y)[0], 0.5)


def test_phase_portrait_winding_of_a_sinusoid():
    om = 0.2; t = np.arange(60)
    c = 1000 + 10 * np.sin(om * t); W = np.zeros((1, 60, 5)); W[0, :, :4] = c[:, None]; W[0, :, 4] = 1
    n = F.n3(W)[0]
    np.testing.assert_allclose(n[2], -om, atol=0.03)                          # (v, a) turns clockwise at om rad per bar
    np.testing.assert_allclose(n[3], -om * 30, atol=1.0)


def test_straight_line_phase_portrait_has_no_winding():
    c = 100 + 0.5 * np.arange(60); W = np.zeros((1, 60, 5)); W[0, :, :4] = c[:, None]
    n = F.n3(W)[0]
    assert abs(n[2]) < 1e-3 and abs(n[3]) < 1e-3 and abs(n[1]) < 1e-3


def test_hull_known_answers():
    t = np.arange(59.0)
    area, nv, last = F.upper_hull((100 + t)[None])
    assert nv[0] == 2 and abs(last[0] - 1.0) < 1e-9
    np.testing.assert_allclose(area[0], (100 + 100 + 58) / 2 * 58)
    tent = np.minimum(t, 58 - t)[None]
    area, nv, last = F.upper_hull(tent)
    assert nv[0] == 3 and abs(last[0] + 1.0) < 1e-9
    zig = (np.arange(59) % 2)[None].astype(float)                              # peaks at odd t: hull is the flat line at 1
    area, nv, last = F.upper_hull(zig); assert nv[0] == 4                                                         # 0 -> 1 ... 57 -> 58 (ties: farthest)


def test_hull_position_zero_and_one_at_the_projected_lines():
    t = np.arange(60.0); W = np.zeros((3, 60, 5))
    for i, close59 in enumerate((158.0, 159.0, 158.5)):
        h = 100 + t; l = 99 + t; c = (h + l) / 2; c[-1] = close59
        W[i, :, 0] = c; W[i, :, 1] = h; W[i, :, 2] = l; W[i, :, 3] = c; W[i, :, 4] = 1
    pos = F.n4(W)[:, 2]
    np.testing.assert_allclose(pos, [0.0, 1.0, 0.5], atol=1e-9)               # lines at 158 (support) and 159 (resistance)
    assert (F.n4(W)[:, 6] == 4).all()                                         # two vertices on each hull


def test_w1_equals_the_shift_in_units_of_sd():
    rng = np.random.default_rng(3); base_r = rng.normal(0, 0.001, 29); delta = 0.002
    r = np.concatenate([base_r, [0.0], base_r + delta]); c = 100 * np.cumprod(1 + np.concatenate([[0.0], r]))
    W = np.zeros((1, 60, 5)); W[0, :, :4] = c[:, None]
    rr = np.diff(c) / c[:-1]; sd = rr.std() + 1e-9
    assert abs(F.n5(W)[0, 2] - F.slog((np.median(rr[30:]) - np.median(rr[:29])) / sd)) < 1e-9
    w1 = np.abs(np.quantile(rr[:29], F._Q) - np.quantile(rr[30:], F._Q)).mean() / sd
    assert abs(F.n5(W)[0, 0] - F.slog(w1)) < 1e-9
    np.testing.assert_allclose(w1, delta / sd, rtol=0.05)


def test_fisher_rao_distances():
    np.testing.assert_allclose(F.fr_dist(0.0, 1.0, 0.0, np.e), np.sqrt(2.0))   # pure scale change: sqrt(2) ln(ratio)
    assert F.fr_dist(0.3, 0.7, 0.3, 0.7) == 0
    d1 = F.fr_dist(0.0, 1.0, 1.0, 1.5); d2 = F.fr_dist(1.0, 1.5, 0.0, 1.0)
    assert d1 == d2 > 0


def test_fisher_rao_path_is_zero_for_identical_sub_windows():
    pat = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.3, -0.3]) * 0.001
    r = np.concatenate([np.concatenate([pat, [0.0]])] * 6)[:59]
    c = 100 * np.cumprod(1 + np.concatenate([[0.0], r])); W = np.zeros((1, 60, 5)); W[0, :, :4] = c[:, None]
    assert abs(F.n6(W)[0, 0]) < 0.05 and abs(F.n6(W)[0, 1]) < 0.05


def test_quadratic_coefficient_is_recovered():
    t = np.arange(60.0); a = 3e-5
    y = 4.0 + 0.002 * t + a * t ** 2; c = np.exp(y); W = np.zeros((1, 60, 5)); W[0, :, :4] = c[:, None]
    sd = np.diff(y).std() + 1e-9
    for j, k in enumerate((15, 30, 60)):
        np.testing.assert_allclose(F.n7(W)[0, j] * sd, a, rtol=1e-6)
    W2 = W.copy(); W2[0, :, :4] = np.exp(4.0 + 0.002 * t - a * t ** 2)[:, None]
    assert (F.n7(W2)[0] < 0).all()
