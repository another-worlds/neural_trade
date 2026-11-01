"""Tests for research/pnl/stats.py: bootstrap, HAC t, engine-F daily PnL, p_sel, Deflated Sharpe, Holm, power."""
import os
import sys

import numpy as np
import pytest
from scipy.stats import kstest, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics  # noqa: E402
import stats as S  # noqa: E402


# --------------------------------------------------------------------------------------------- bootstrap

def test_bootstrap_shape_and_seed():
    x = np.random.default_rng(0).standard_normal(300)
    a = S.stationary_bootstrap(x, B=2500, seed=3)                 # not a multiple of the chunk size
    assert a.shape == (2500,)
    assert np.array_equal(a, S.stationary_bootstrap(x, B=2500, seed=3))
    assert not np.array_equal(a, S.stationary_bootstrap(x, B=2500, seed=4))


def test_stationary_indices_block_structure():
    rng = np.random.default_rng(1)
    T, mb = 1800, 10
    idx = S._sb_indices(rng, 1000, T, 1.0 / mb)
    assert idx.min() >= 0 and idx.max() < T
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % T).mean()          # P(continue the block) = 1 - 1/mb (+ 1/T by chance)
    assert abs(cont - (1 - 1 / mb)) < 0.005
    counts = np.bincount(idx.ravel(), minlength=T)                # every day is equally likely
    assert counts.std() / counts.mean() < 0.05


def test_bootstrap_variance_closed_form():
    # Politis and Romano (1994): Var*(mean*) = (1/T)[C0 + 2 sum_{k=1}^{T-1} (1 - k/T) q^k C_k], q = 1 - 1/mean_block,
    # C_k the circular sample autocovariances. For iid data it is close to, but not equal to, var/T.
    for seed, mb in ((2, 10), (3, 20), (4, 1)):
        x = np.random.default_rng(seed).standard_normal(500)
        T, q = len(x), 1 - 1 / mb
        e = x - x.mean()
        C = np.array([np.mean(e * np.roll(e, -k)) for k in range(T)])
        k = np.arange(1, T)
        v = (C[0] + 2 * ((1 - k / T) * q ** k) @ C[1:]) / T
        d = S.stationary_bootstrap(x, B=10000, mean_block=mb, seed=seed)
        assert abs(d.mean() - x.mean()) < 4 * np.sqrt(v / 10000)
        assert abs(d.std() / np.sqrt(v) - 1) < 0.03                 # MC SE of the SD ~0.7%


def test_bootstrap_mean_ci_coverage_iid_normal():
    rng = np.random.default_rng(5)
    reps, T, mu = 300, 500, 0.1
    hit = 0
    for i in range(reps):
        x = mu + rng.standard_normal(T)
        lo, hi = S.percentile_ci(S.stationary_bootstrap(x, B=1000, mean_block=10, seed=i), 0.95)
        hit += lo <= mu <= hi
    cov = hit / reps
    assert 0.90 <= cov <= 0.98, cov                               # nominal 0.95; 300 reps give SE ~0.013


def test_one_sided_bound_and_pvalue():
    rng = np.random.default_rng(6)
    x = 0.2 + rng.standard_normal(400)
    d = S.stationary_bootstrap(x, B=5000, seed=1)
    assert S.lower_bound(d, 0.95) == pytest.approx(np.quantile(d, 0.05))
    assert S.upper_bound(d, 0.95) == pytest.approx(np.quantile(d, 0.95))
    assert S.lower_bound(d) > 0 and S.boot_pvalue(d) < 0.05
    y = rng.standard_normal(400) - 0.2
    e = S.stationary_bootstrap(y, B=5000, seed=1)
    assert S.lower_bound(e) < 0 and S.boot_pvalue(e) > 0.05
    assert S.prob_positive(d) == pytest.approx(1 - np.mean(d <= 0))


def test_paired_bootstrap():
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal(500), rng.standard_normal(500)
    assert np.all(S.paired_bootstrap(x, x, B=500) == 0)
    assert np.allclose(S.paired_bootstrap(x + 0.01, x, B=500), 0.01)
    # mean difference on the same resampled days = bootstrap of the daily difference with the same seed
    assert np.allclose(S.paired_bootstrap(x, y, B=1500, seed=9), S.stationary_bootstrap(x - y, B=1500, seed=9))
    with pytest.raises(ValueError):
        S.paired_bootstrap(x, y[:-1], B=10)


def test_bootstrap_stats_vectorised_and_fallback():
    x = np.random.default_rng(8).normal(0.001, 0.02, 400)
    for stat in (S.sharpe_ann, S.max_drawdown):
        fast = S.stationary_bootstrap(x, B=300, seed=2, stat=stat)
        slow = S.stationary_bootstrap(x, B=300, seed=2, stat=lambda v: float(stat(v)))   # no axis -> row by row
        assert np.allclose(fast, slow)
    assert np.allclose(S.sharpe_ann(x), metrics.sharpe(x))
    assert np.allclose(S.max_drawdown(x), metrics.max_drawdown(x))


# --------------------------------------------------------------------------------------------- HAC t

def test_newey_west_hand_example():
    # x = 1..4, lag 1: e = (-1.5, -.5, .5, 1.5); g0 = 1.25, g1 = 0.3125, S = 1.25 + 2 * 0.5 * 0.3125 = 1.5625
    assert S.newey_west_se([1, 2, 3, 4], lag=1) == pytest.approx(np.sqrt(1.5625 / 4))
    assert S.newey_west_t([1, 2, 3, 4], lag=1) == pytest.approx(4.0)


def test_newey_west_iid_close_to_plain_t():
    rng = np.random.default_rng(10)
    for _ in range(5):
        x = 0.05 + rng.standard_normal(2000)
        t_plain, t_hac = S.plain_t(x), S.newey_west_t(x, lag=10)
        assert abs(t_hac - t_plain) < 0.1 * max(1.0, abs(t_plain))
        T = len(x)                                                 # lag 0: only the ddof differs
        assert S.newey_west_t(x, lag=0) == pytest.approx(t_plain * np.sqrt(T / (T - 1)))


def test_newey_west_autocorrelated_wider():
    rng = np.random.default_rng(11)
    e = rng.standard_normal(5000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.5 * x[i - 1] + e[i]
    ratio = S.newey_west_se(x, lag=20) / (x.std(ddof=1) / np.sqrt(len(x)))
    assert 1.4 < ratio < 1.9                                       # long-run factor sqrt((1+phi)/(1-phi)) = 1.73


def test_t_vectorised_and_degenerate():
    Y = np.random.default_rng(12).standard_normal((3, 200))
    assert np.allclose(S.newey_west_t(Y, 5), [S.newey_west_t(y, 5) for y in Y])
    assert S.newey_west_t(np.zeros(50)) == 0 and S.plain_t(np.zeros(50)) == 0


# --------------------------------------------------------------------------------------------- engine F and p_sel

def test_engine_f_daily_hand_example():
    R = np.array([0.01, 0.02, -0.01, 0.03, 0.05])
    r = np.log1p(R)
    f = np.array([0.0, 0.0, 0.001, 0.0, 0.0])
    x = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    # bar 1: enter +1 (3 bps); bar 2: long pays -1% and 10 bps funding; bar 3: +3%, flip 5 + 3 bps; bar 4: short -5%, exit 5
    per_bar = [0.0, -3e-4, -0.011, 0.03 - 8e-4, -0.05 - 5e-4]
    d = S.engine_f_daily(x, r, f, np.array([0, 0, 0, 1, 1]))
    assert d.shape == (1, 2)
    assert np.allclose(d[0], [sum(per_bar[:3]), sum(per_bar[3:])])
    # the shift moves returns and funding together (costs stay): rolling by 1 puts R[1] = 0.02 on bar 2, and
    # R[2] with its funding print (-0.011) on bar 3
    d1 = S.engine_f_daily(x, r, f, np.zeros(5), shift=1)
    assert d1[0, 0] == pytest.approx(-3e-4 + 0.02 + (-0.011 - 8e-4) + (0.03 * -1 - 5e-4))


def _synthetic(rng, D, M, K, sd=1e-3, hold=20):
    n = D * M
    r = rng.standard_normal(n) * sd
    f = np.zeros(n)
    f[::max(M // 3, 1)] = 1e-4
    X = np.empty((K, n), np.float32)
    for k in range(K):
        ch = rng.random(n) < 1.0 / hold
        X[k] = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=ch.sum() + 1)[np.cumsum(ch)]
    return X, r, f, np.repeat(np.arange(D), M)


@pytest.mark.parametrize("irregular", [False, True])
def test_p_sel_matches_reference(irregular):
    rng = np.random.default_rng(13)
    X, r, f, day = _synthetic(rng, D=100, M=48, K=3)
    if irregular:                                                  # drop bars so days have different lengths
        keep = rng.random(len(r)) > 0.2
        X, r, f, day = X[:, keep], r[keep], f[keep], day[keep]
    res = S.p_sel(X, r, f, day, n_shifts=20, min_shift=30 * 48, seed=4, dtype=np.float64)
    assert np.allclose(res["daily_obs"], S.engine_f_daily(X, r, f, day), atol=1e-12)
    assert np.allclose(res["t_obs"], S.newey_west_t(S.engine_f_daily(X, r, f, day), 10))
    for b in (0, 7, 19):
        ref = S.engine_f_daily(X, r, f, day, shift=int(res["shifts"][b]))
        assert np.allclose(res["null_t"][b], S.newey_west_t(ref, 10), atol=1e-9)
    assert np.all((res["shifts"] >= 30 * 48) & (res["shifts"] <= len(r) - 30 * 48))
    # float32 fast path agrees with the float64 one
    res32 = S.p_sel(X, r, f, day, n_shifts=20, min_shift=30 * 48, seed=4)
    assert np.allclose(res32["null_t"], res["null_t"], atol=1e-3)
    plain = S.p_sel(X, r, f, day, n_shifts=5, min_shift=30 * 48, seed=4, tstat="plain", dtype=np.float64)
    assert np.allclose(plain["t_obs"], S.plain_t(S.engine_f_daily(X, r, f, day)))


def test_p_sel_definition():
    rng = np.random.default_rng(14)
    X, r, f, day = _synthetic(rng, D=90, M=48, K=4)
    res = S.p_sel(X, r, f, day, n_shifts=99, min_shift=30 * 48, seed=1)
    assert res["null_t"].shape == (99, 4) and res["null_max"].shape == (99,)
    assert np.allclose(res["null_max"], res["null_t"].max(1))
    assert res["p_sel"] == (1 + np.sum(res["null_max"] >= res["t_obs"].max())) / 100
    assert np.all(res["p_single"] <= 1) and res["p_sel"] >= res["p_single"][res["best"]] - 1e-12
    # same seed, same shifts; a cash candidate gets t = 0 and never breaks the max
    Xc = np.vstack([X, np.zeros((1, X.shape[1]), np.float32)])
    resc = S.p_sel(Xc, r, f, day, n_shifts=99, min_shift=30 * 48, seed=1)
    assert np.array_equal(resc["shifts"], res["shifts"]) and resc["t_obs"][-1] == 0
    with pytest.raises(ValueError):
        S.p_sel(X, r, f, day, min_shift=len(r) // 2)


def test_p_sel_uniform_under_null():
    # random positions, independent of iid returns: p_sel should be ~U(0, 1)
    rng = np.random.default_rng(15)
    ps = []
    for i in range(150):
        X, r, f, day = _synthetic(rng, D=90, M=48, K=3)
        ps.append(S.p_sel(X, r, f, day, n_shifts=99, min_shift=30 * 48, seed=i)["p_sel"])
    ps = np.array(ps)
    assert abs(ps.mean() - 0.5) < 0.07
    assert np.mean(ps <= 0.10) < 0.18 and np.mean(ps <= 0.5) > 0.35
    assert kstest(ps, "uniform").pvalue > 0.01


def test_p_sel_planted_signal():
    # candidate 0 knows the sign of the next return (plus noise of the same size); two random candidates
    rng = np.random.default_rng(16)
    X, r, f, day = _synthetic(rng, D=90, M=96, K=3)
    n = len(r)
    nxt = np.r_[r[1:], 0.0]
    X[0] = np.sign(nxt + rng.standard_normal(n) * 1e-3)
    res = S.p_sel(X, r, f, day, n_shifts=999, min_shift=30 * 96, seed=0)
    assert res["best"] == 0 and res["t_obs"][0] > 5
    assert res["p_sel"] < 0.01 and res["p_sel"] == pytest.approx(0.001)


# --------------------------------------------------------------------------------------------- Deflated Sharpe

def test_dsr_formula_spot_checks():
    # N = 1, normal returns: DSR = Phi(SR sqrt(T-1) / sqrt(1 + SR^2/2))
    assert S.expected_max_sharpe(1, 0.3) == 0
    assert S.deflated_sharpe(0.1, 1, 0.0, 0.0, 3.0, 101) == pytest.approx(0.8407413278013518, abs=1e-12)
    assert S.deflated_sharpe(0.1, 1, 0.0, 0.0, 3.0, 101) == pytest.approx(norm.cdf(1 / np.sqrt(1.005)))
    # expected max of N = 100 standard trials: (1-g) Z^-1(0.99) + g Z^-1(1 - 1/(100 e))
    g = 0.5772156649015329
    assert S.expected_max_sharpe(100, 1.0) == pytest.approx((1 - g) * 2.3263478740408408 + g * 2.6802104449668875)
    assert S.expected_max_sharpe(100, 4.0) == pytest.approx(2 * S.expected_max_sharpe(100, 1.0))
    # skew and kurtosis enter as 1 - skew SR + (kurt - 1)/4 SR^2
    sr, sr0, T, sk, ku = 0.08, 0.03, 500, -1.0, 6.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    assert S.probabilistic_sharpe(sr, sr0, T, sk, ku) == pytest.approx(norm.cdf(z))


def test_dsr_paper_example():
    # Bailey and Lopez de Prado (2014), numerical example: annual SR 2.5, N = 100, V[SR_n] = 1/2 (annual),
    # T = 1250 days, skew -3, kurtosis 10 -> SR0 ~ 0.1132 per day, DSR ~ 0.9004
    sr0 = S.expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    assert S.deflated_sharpe(2.5 / np.sqrt(250), 100, 0.5 / 250, -3.0, 10.0, 1250) == pytest.approx(0.9004, abs=1e-4)


def test_dsr_monotone():
    base = dict(var_sr=0.002, skew=0.0, kurt=3.0, T=1765)
    assert S.deflated_sharpe(0.1, 10, **base) > S.deflated_sharpe(0.1, 100, **base)   # more trials, lower DSR
    assert S.deflated_sharpe(0.1, 10, **base) < S.deflated_sharpe(0.12, 10, **base)


# --------------------------------------------------------------------------------------------- Holm

def test_holm_hand_example():
    # sorted: d .005*4 = .02, a .01*3 = .03, c .03*2 = .06, b .04*1 = .04 -> running max .06
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert list(adj) == ["a", "b", "c", "d"]
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})
    assert sorted(k for k, p in adj.items() if p <= 0.05) == ["a", "d"]
    assert S.holm({"x": 0.4, "y": 0.6}) == pytest.approx({"x": 0.8, "y": 0.8})
    assert S.holm({"only": 0.03}) == {"only": 0.03}
    assert S.holm({"a": 0.5, "b": 0.9, "c": 0.7}) == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0})


# --------------------------------------------------------------------------------------------- power (SPEC POWER)

def test_power_track1_numbers():
    # per-trade SD 25-40 bps, design effect 1.3, one-sided 5%, 80% power
    mde = lambda sd, n: S.min_detectable_mean(sd, n, deff=1.3)
    assert (round(mde(25, 1000), 1), round(mde(40, 1000), 1)) == (2.2, 3.6)
    assert (round(mde(25, 3000), 1), round(mde(40, 3000), 1)) == (1.3, 2.1)
    assert (round(mde(25, 10000), 1), round(mde(40, 10000), 1)) == (0.7, 1.1)
    # true net -3 bps per trade: power to reject net >= 0 is 0.67 at 1,000 trades and 0.97 at 3,000 (SD 40)
    assert round(S.power_mean(3, 40, 1000, deff=1.3), 2) == 0.67
    assert round(S.power_mean(3, 40, 3000, deff=1.3), 2) == 0.97
    assert S.power_mean(S.min_detectable_mean(30, 500), 30, 500) == pytest.approx(0.8)


def test_power_track2_numbers():
    assert round(S.min_detectable_sharpe(363 / 365), 2) == 2.49            # seal
    assert round(S.min_detectable_sharpe(0.86), 2) == 2.68                  # clean seal
    assert round(S.min_detectable_sharpe(1765 / 365), 2) == 1.13            # dev OOS
    assert round(S.min_detectable_sharpe(1765 / 365 + 1), 2) == 1.03        # dev OOS + seal
    assert round(S.power_sharpe(0.8, 1.0), 2) == 0.20
    assert round(S.power_sharpe(0.8, 1765 / 365), 2) == 0.55
