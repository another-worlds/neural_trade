"""Seconds-long tests of the sizing study. usage: python -m pytest -q -p no:cacheprovider runs/tactical/sizing/test_sizing.py"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sizelib as S  # noqa: E402


def synth(n=24 * 420, seed=0):
    rng = np.random.default_rng(seed)
    times = (pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(n), unit="h")).values.astype("datetime64[ns]")
    rv = np.exp(rng.normal(-11, 0.6, n)) * (1 + 0.5 * np.sin(np.arange(n) / 90.0) ** 2)
    return rv, times


def test_inputs_at_bar_j_ignore_the_future():
    rv, times = synth(); h = 4
    base = S.state(rv, times, h, "2021-12", "2022-02")
    j = int(np.flatnonzero(pd.DatetimeIndex(times) >= pd.Timestamp("2022-01-15"))[0])
    rv2 = rv.copy(); rv2[j + 1:] = rv2[j + 1:] * 50 + 1.0          # corrupt every bar after j
    alt = S.state(rv2, times, h, "2021-12", "2022-02")
    for k in ("pred", "sigma", "dlv", "target", "thr"):
        a, b = base[k][: j + 1], alt[k][: j + 1]
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0), k
    assert np.isfinite(base["sigma"][j]) and np.isfinite(base["thr"][j])
    w = S.base_weight(base["sigma"][j], base["target"][j])
    assert S.W_MIN <= w <= S.W_MAX


def test_har_training_rows_end_before_the_month_start():
    # the model of a month must not change when bars from the month start on (their targets are the future of its training rows) change:
    # predictions for the month's FIRST bar use features of bars <= j only, so corrupt bars strictly after j and the training set too
    rv, times = synth(); h = 4; m0 = pd.Timestamp("2022-01-01"); t = pd.DatetimeIndex(times)
    j = int(np.flatnonzero(t >= m0)[0])
    a, _ = S.har_walk_forward(rv, times, h, "2022-01", "2022-01")
    rv2 = rv.copy(); rv2[j + 1:] = 99.0
    b, _ = S.har_walk_forward(rv2, times, h, "2022-01", "2022-01")
    assert a[j] == b[j]                                             # same beta (trained before m0) and same features
    y = S.har_target(rv, h)
    tr = np.isfinite(y) & (t >= m0 - pd.Timedelta(days=365)) & (t + pd.Timedelta(hours=h + 1) <= m0)
    last = np.flatnonzero(tr).max()
    assert t[last + h] + pd.Timedelta(hours=1) <= m0                # the last training target bar closed by the month start


def test_brake_weight_ignores_the_trade_itself_and_the_future():
    rng = np.random.default_rng(1); n = 300
    g = rng.normal(0, 0.004, (1, n)); w0 = np.ones(n)
    p1, w1 = S.simulate(w0, g, 0.0, True)
    for i in (50, 120, 250):
        g2 = g.copy(); g2[:, i:] = rng.normal(0, 0.05, (1, n - i))
        p2, w2 = S.simulate(w0, g2, 0.0, True)
        assert np.array_equal(w1[:, : i + 1], w2[:, : i + 1])       # weights up to and including trade i do not see trade i's P&L


def test_brake_hand_path():
    # equity 1 -> 1.03 (peak) -> 1.00 (dd 2.9% >= 2%: brake) -> next trade at half size -> new peak restores
    g = np.array([[0.03, -0.03, 0.02, 0.02, 0.02]])
    pnl, w = S.simulate(np.ones(5), g, 0.0, True)
    # step by step: after trade0 eq 1.03 new peak; trade1 weight 1 -> eq 1.00, dd 2.91% -> on; trade2 w .5 -> 1.01; trade3 .5 -> 1.02
    # (still below 1.03: stays on); trade4 .5 -> 1.03 (not above the peak 1.03: stays on)
    assert np.allclose(w[0], [1, 1, 0.5, 0.5, 0.5]) and np.allclose(pnl[0], [0.03, -0.03, 0.01, 0.01, 0.01])


def test_cost_scales_with_weight_and_hit_is_cost_sign():
    g = np.array([[0.0003]]); pnl, w = S.simulate(np.array([2.0]), g, 5.0, False)
    assert np.allclose(pnl, 2.0 * (0.0003 - 0.0005)) and (pnl < 0).all()


def test_non_overlap_after_the_filter():
    rng = np.random.default_rng(2); n = 400
    bar = np.arange(n) * 1; entry = (np.sort(rng.integers(0, 200, n)) * S.HOUR); exit_ = entry + rng.integers(1, 6, n) * S.HOUR
    bar = np.arange(n)
    st = {"dlv": rng.normal(size=n), "thr": np.zeros(n), "sigma": np.ones(n), "target": np.ones(n)}
    for rule in S.RULES:
        idx, w0 = S.apply_rule(rule, bar, entry, exit_, st)
        assert (entry[idx][1:] >= exit_[idx][:-1]).all(), rule
    # the filter drops dlv > thr signals, then the free position is used by later ones
    idx_f, _ = S.apply_rule("R2_volchange_filter", bar, entry, exit_, st)
    assert (st["dlv"][idx_f] <= st["thr"][idx_f]).all()
    idx_0, _ = S.apply_rule("R0_fixed", bar, entry, exit_, st)
    assert len(idx_f) > 0 and len(idx_0) > 0


def test_cutoff_bounds():
    assert pd.Timestamp(S.C.CUTOFF) == pd.Timestamp("2025-07-20 23:59")
    rv = S.hourly_rv(end="2021-03-01")
    assert rv.index.max() < pd.Timestamp("2021-03-01") and rv.index.max() <= pd.Timestamp(S.C.CUTOFF)
    full = S.hourly_rv(end="2030-01-01")
    assert full.index.max() <= pd.Timestamp(S.C.CUTOFF)
    import pytest
    with pytest.raises(AssertionError):
        S.C.assert_cut(pd.DatetimeIndex(["2025-07-21 00:00"]), "x")


def test_r0_equals_the_frozen_sequence_of_fwdlib():
    sys.path.insert(0, os.path.join(S.TAC, "fwd")); import fwdlib as F  # noqa
    rng = np.random.default_rng(3); n = 600; H = 4
    t = (np.arange(n) * S.HOUR).astype(np.int64); idx = np.arange(n); p = rng.random(n); r = rng.normal(0, 0.003, n)
    ent, ex, side, ret, mon = F.trades_from([("2021-01", idx, p, r)], t, H, 0.1)
    k = np.abs(p - 0.5) >= np.quantile(np.abs(p - 0.5), 0.9)
    entry = t[k] + S.HOUR; exit_ = t[k] + (H + 1) * S.HOUR
    st = {"dlv": np.zeros(n), "thr": np.zeros(n), "sigma": np.ones(n), "target": np.ones(n)}
    sel, _ = S.apply_rule("R0_fixed", np.flatnonzero(k), entry, exit_, st)
    assert np.array_equal(entry[sel], ent) and np.array_equal(exit_[sel], ex)
