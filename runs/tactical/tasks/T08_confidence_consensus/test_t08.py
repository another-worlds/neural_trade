"""T08 tests (SPEC "Tests"): percentile confidence from the previous month only, consensus rules on synthetic votes, non-overlapping 4-hour holds,
next-open, selection.json committed before the looks, the never-read span (+ costs, the null, the bootstrap, Holm, deciles)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pytest

import t08_lib as L
import t07_lib as T
import t08_scores as SC


def hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


# ------------------------------------------------------------------ percentile confidence
def test_percentile_uses_only_the_previous_month():
    rng = np.random.default_rng(0)
    ht = hours("2020-01-01", 24 * 91); months = L.month_labels(ht)                    # Jan, Feb (29 d), Mar
    s = rng.normal(size=len(ht)); c = L.prev_month_percentile(s, months)
    jan, feb, mar = (months == "2020-01"), (months == "2020-02"), (months == "2020-03")
    assert np.isnan(c[jan]).all()                                                      # no previous month, no confidence
    prev = np.sort(np.abs(s[jan])); k = np.flatnonzero(feb)[10]
    assert c[k] == pytest.approx(np.searchsorted(prev, abs(s[k]), side="right") / len(prev))
    s2 = s.copy(); other = feb.copy(); other[k] = False; s2[other] *= 3.0              # change every other February value
    c2 = L.prev_month_percentile(s2, months)
    assert c2[k] == c[k]                                                               # the rest of February does not enter February's ranks
    assert not np.allclose(c2[mar], c[mar])                                            # but February is March's reference
    s3 = s.copy(); s3[mar] += 5.0                                                      # a later month never changes an earlier one
    c3 = L.prev_month_percentile(s3, months)
    assert np.array_equal(c3[jan | feb], c[jan | feb], equal_nan=True)


def test_percentile_is_a_rank_in_zero_one_and_skips_nan():
    ht = hours("2020-01-01", 24 * 62); months = L.month_labels(ht)
    s = np.linspace(-1, 1, len(ht)); s[100] = np.nan
    c = L.prev_month_percentile(s, months)
    ok = np.isfinite(c); assert ((c[ok] >= 0) & (c[ok] <= 1)).all() and np.isnan(c[100])
    big = np.flatnonzero(months == "2020-02")[0]
    s_small = s.copy(); s_small[big] = 0.0
    assert L.prev_month_percentile(s_small, months)[big] == 0.0                         # smaller than every previous-month value
    s_big = s.copy(); s_big[big] = 100.0
    assert L.prev_month_percentile(s_big, months)[big] == 1.0                           # larger than every previous-month value


# ------------------------------------------------------------------ consensus rules on synthetic votes
def mk(signs, cs):
    """S [6, 1] with the given signs and C [6, 1] with the given percentile confidences (None = no score)."""
    S = np.array([[np.nan if c is None else sg * 0.1] for sg, c in zip(signs, cs)]); C = np.array([[np.nan if c is None else c] for c in cs])
    return S, C


def test_k1_needs_m1_m2_m3_to_vote_with_one_sign():
    q = 0.8
    S, C = mk([1, 1, 1, -1, -1, -1], [.9, .9, .9, .1, .1, .1]); assert L.signal_consensus(S, C, "K1", q)[0] == 1
    S, C = mk([-1, -1, -1, 1, 1, 1], [.85, .9, .99, 0, 0, 0]); assert L.signal_consensus(S, C, "K1", q)[0] == -1
    S, C = mk([1, 1, -1, 1, 1, 1], [.9, .9, .9, .9, .9, .9]); assert L.signal_consensus(S, C, "K1", q)[0] == 0   # M3 disagrees
    S, C = mk([1, 1, 1, 1, 1, 1], [.9, .9, .79, .9, .9, .9]); assert L.signal_consensus(S, C, "K1", q)[0] == 0   # M3 below q
    S, C = mk([1, 1, 1, 1, 1, 1], [.9, None, .9, .9, .9, .9]); assert L.signal_consensus(S, C, "K1", q)[0] == 0   # a member without a score


def test_k2_three_voters_one_sign_and_no_dissent():
    q = 0.9
    S, C = mk([1, 1, 1, 0, 0, 0], [.95, .95, .95, .0, .0, .0]); S[3:] = np.nan; C[3:] = np.nan
    assert L.signal_consensus(S, C, "K2", q)[0] == 1                                    # exactly 3 voters, nobody else
    S, C = mk([1, 1, 1, 1, 1, 1], [.95, .95, .6, .2, .2, .2]); assert L.signal_consensus(S, C, "K2", q)[0] == 0   # only 2 voters
    S, C = mk([1, 1, 1, -1, 1, 1], [.95, .95, .95, .4, .2, .2]); assert L.signal_consensus(S, C, "K2", q)[0] == 1   # opposing at c < 0.5: no veto
    S, C = mk([1, 1, 1, -1, 1, 1], [.95, .95, .95, .5, .2, .2]); assert L.signal_consensus(S, C, "K2", q)[0] == 0   # opposing at c >= 0.5: veto
    S, C = mk([1, 1, -1, 1, 1, 1], [.95, .95, .95, .95, .2, .2]); assert L.signal_consensus(S, C, "K2", q)[0] == 0  # a voter with the other sign
    S, C = mk([-1, -1, -1, -1, 1, 1], [.95, .95, .95, .95, .45, .3]); assert L.signal_consensus(S, C, "K2", q)[0] == -1   # four short voters
    S, C = mk([-1, -1, -1, -1, 1, 1], [.95, .95, .95, .95, .95, .3]); assert L.signal_consensus(S, C, "K2", q)[0] == 0   # a long voter too: mixed


def test_k3_needs_all_six():
    S, C = mk([1] * 6, [.96] * 6); assert L.signal_consensus(S, C, "K3", 0.95)[0] == 1
    S, C = mk([1] * 5 + [-1], [.96] * 6); assert L.signal_consensus(S, C, "K3", 0.95)[0] == 0
    S, C = mk([1] * 6, [.96] * 5 + [.94]); assert L.signal_consensus(S, C, "K3", 0.95)[0] == 0
    S, C = mk([-1] * 6, [1.0] * 6); assert L.signal_consensus(S, C, "K3", 0.95)[0] == -1


def test_family_a_trades_when_c_reaches_q_in_the_members_direction():
    S = np.array([[0.2, -0.3, 0.1, 0.4]] * 6); C = np.array([[0.89, 0.9, 0.97, 0.99]] * 6)
    sig = L.signal_member(S, C, 2, 0.90); assert list(sig) == [0, -1, 1, 1] and L.signal_member(S, C, 2, 0.98)[3] == 1 and L.signal_member(S, C, 2, 0.98)[2] == 0


def test_grid_sizes():
    a, b = L.config_names(); assert len(a) == 18 and len(b) == 9 and len(set(a + b)) == 27


# ------------------------------------------------------------------ 4-hour holds
def test_holds_are_four_hours_and_never_overlap():
    sig = np.zeros(40, np.int8); sig[[2, 3, 5, 6, 7, 20, 38, 39]] = [1, -1, 1, 1, -1, -1, 1, 1]
    ent, side = L.simulate(sig)
    assert list(ent) == [2, 6, 20] and list(side) == [1, 1, -1]                        # 3, 5 skipped while 2's trade is open; 6 is the first free hour; 38, 39 would exit past the data
    assert (np.diff(ent) >= L.HOLD).all()
    ent2, _ = L.simulate(np.where(np.arange(40) % 4 == 0, 1, 0).astype(np.int8))
    assert list(ent2[:3]) == [0, 4, 8]                                                  # back to back at the exit hour is allowed


def test_position_is_held_exactly_four_hours():
    nh = 24; Hr = np.arange(1, nh + 1, dtype=float) * 1e-4
    ent = np.array([3]); side = np.array([-1]); rh, pos = L.hourly_returns(ent, side, Hr, 0.0)
    assert list(np.flatnonzero(pos)) == [3, 4, 5, 6] and (pos[[3, 4, 5, 6]] == -1).all()
    assert rh[3:7] == pytest.approx(-Hr[3:7]) and (rh[:3] == 0).all() and (rh[7:] == 0).all()


# ------------------------------------------------------------------ next-open
def test_entry_earns_the_entry_hours_return_not_the_signal_hours():
    nh = 12; Hr = np.zeros(nh); Hr[2] = 0.05                                            # a jump in hour 2 (the bar that produced the signal)
    rh, pos = L.hourly_returns(np.array([3]), np.array([1]), Hr, 0.0)
    assert rh.sum() == 0.0 and pos[2] == 0                                              # the signal bar's own return is never earned
    Hr[3] = 0.02; rh, _ = L.hourly_returns(np.array([3]), np.array([1]), Hr, 0.0); assert rh[3] == pytest.approx(0.02)


def test_rule_scores_of_hour_k_see_only_closed_bars_before_k():
    rng = np.random.default_rng(1); n = 400
    hc = 100 * np.exp(np.cumsum(rng.normal(0, 3e-3, n))); hi = hc * 1.002; lo = hc * 0.998
    class G: pass
    g = G(); g.hc = hc
    base = SC.rule_scores(g, hi, lo)
    hc2, hi2, lo2 = hc.copy(), hi.copy(), lo.copy(); hc2[250:] *= 1.4; hi2[250:] *= 1.4; lo2[250:] *= 1.4
    g2 = G(); g2.hc = hc2
    alt = SC.rule_scores(g2, hi2, lo2)
    assert np.array_equal(np.nan_to_num(base[:, :251], nan=9), np.nan_to_num(alt[:, :251], nan=9))     # score of hour 250 uses bars < 250
    assert not np.allclose(np.nan_to_num(base[:, 251]), np.nan_to_num(alt[:, 251]))


def test_hourly_model_scores_land_on_the_entry_hour():
    raw = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=pd.date_range("2020-01-01", periods=24 * 60, freq="min"))
    g = T.Grid(raw); row = np.full(g.nh, np.nan)
    t_sig = pd.DatetimeIndex(["2020-01-01 05:00"]); t_entry = (t_sig + pd.Timedelta(hours=1)).values.astype("datetime64[m]")
    SC.put(row, g, t_entry, np.array([0.8]))
    assert np.flatnonzero(np.isfinite(row)).tolist() == [6] and row[6] == pytest.approx(0.3)           # the bar closed at 05:59, filled at the 06:00 open
    row2 = np.full(g.nh, np.nan); SC.put(row2, g, np.array(["2020-01-01T06:10"], "datetime64[m]"), np.array([0.9]))
    assert np.isnan(row2).all()                                                         # an anchor that is not on an hour start gives no hourly score


# ------------------------------------------------------------------ costs, the null, the bootstrap, Holm, deciles
def test_cost_is_a_round_trip_per_trade_even_back_to_back():
    Hr = np.zeros(20)
    rh, _ = L.hourly_returns(np.array([2, 6]), np.array([1, 1]), Hr, 2.0)               # the second trade starts where the first ends
    assert rh.sum() == pytest.approx(-2 * 2.0 * 1e-4) and rh[2] == pytest.approx(-1e-4) and rh[6] == pytest.approx(-2e-4) and rh[10] == pytest.approx(-1e-4)
    nb = L.trade_net_bps(np.array([2]), np.array([-1]), np.full(20, 1e-4), 2.0); assert nb[0] == pytest.approx(-4.0 - 2.0)


def test_null_flips_signs_only_and_keeps_entries_and_costs():
    rng = np.random.default_rng(2); Hr = rng.normal(0, 1e-3, 200); ent = np.array([5, 20, 60, 61 + 3, 100]); ent = ent[np.r_[True, np.diff(ent) >= 4]]
    side = np.array([1, -1, 1, 1, -1])[:len(ent)]
    flips = np.array([side.astype(float), -side.astype(float), np.ones(len(ent))])
    cum = L.null_cum(ent, side, Hr, 2.0, 0, 200, flips=flips)
    for f, got in zip(flips, cum):
        rh, _ = L.hourly_returns(ent, f.astype(int), Hr, 2.0)
        assert got == pytest.approx(np.prod(1 + rh) - 1)
    actual, _ = L.hourly_returns(ent, side, Hr, 2.0)
    assert cum[0] == pytest.approx(np.prod(1 + actual) - 1)
    big = L.null_cum(ent, side, Hr, 2.0, 0, 200, draws=50); assert len(big) == 50


def test_null_p95_below_perfect_foresight():
    Hr = np.random.default_rng(3).normal(0, 2e-3, 400); ent = np.arange(0, 396, 4);
    cs = np.concatenate([[0.0], np.cumsum(Hr)]); side = np.where(cs[ent + 4] - cs[ent] > 0, 1, -1)
    perfect, _ = L.hourly_returns(ent, side, Hr, 0.0); cum = L.null_cum(ent, side, Hr, 0.0, 0, 400, draws=500)
    assert np.prod(1 + perfect) - 1 > np.quantile(cum, 0.95)


def test_week_bootstrap_detects_a_positive_mean_and_not_noise():
    t = pd.date_range("2025-11-03", periods=200, freq="12h")
    pos = L.week_bootstrap(np.full(200, 5.0) + np.random.default_rng(0).normal(0, 1, 200), t, draws=2000)
    assert pos["lo"] > 0 and pos["p"] < 0.01 and pos["n_weeks"] >= 14
    noise = L.week_bootstrap(np.random.default_rng(1).normal(0, 20, 200), t, draws=2000)
    assert noise["lo"] < 0 < noise["hi"]
    assert np.isnan(L.week_bootstrap(np.array([]), t[:0])["mean"])


def test_holm_step_down():
    assert L.holm({"a": 0.02, "b": 0.04}) == {"a": True, "b": True}
    assert L.holm({"a": 0.02, "b": 0.06}) == {"a": True, "b": False}
    assert L.holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}                  # the smaller p must clear 0.025


def test_decile_table_bins_and_hits():
    c = np.array([0.05, 0.05, 0.95, 0.95, 1.0, np.nan]); d = np.array([1, -1, 1, 1, -1, 1]); f = np.array([10.0, 10.0, 10.0, -10.0, -30.0, 5.0])
    rows = L.decile_table(c, d, f, cost_bps=2.0)
    assert rows[0]["n"] == 2 and rows[0]["hit"] == pytest.approx(0.5) and rows[9]["n"] == 3
    assert rows[9]["hit"] == pytest.approx(2 / 3) and rows[9]["hit_net"] == pytest.approx(2 / 3) and rows[4]["n"] == 0


# ------------------------------------------------------------------ selection.json before the looks
def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True)


def test_selection_must_be_committed_before_looks(tmp_path):
    _git(tmp_path, "init", "-q"); _git(tmp_path, "config", "user.email", "t@t"); _git(tmp_path, "config", "user.name", "t")
    p = tmp_path / "selection.json"; p.write_text('{"finalist_A": "x"}')
    with pytest.raises(AssertionError, match="not tracked"):
        L.assert_selection_committed(str(p))
    _git(tmp_path, "add", "selection.json")
    with pytest.raises(AssertionError, match="modified or staged"):
        L.assert_selection_committed(str(p))
    _git(tmp_path, "commit", "-q", "-m", "x"); assert len(L.assert_selection_committed(str(p))) == 64
    p.write_text('{"finalist_A": "y"}')
    with pytest.raises(AssertionError, match="modified or staged"):
        L.assert_selection_committed(str(p))


def test_run_looks_refuses_without_a_committed_selection(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("t08_run_looks", os.path.join(HERE, "run_looks.py")); run_looks = importlib.util.module_from_spec(spec); spec.loader.exec_module(run_looks)
    monkeypatch.setattr(run_looks, "SEL", str(tmp_path / "selection.json"))
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(AssertionError, match="not tracked"):
        run_looks.main()


def test_selection_commit_precedes_results_commit():
    sel, res = os.path.join(HERE, "selection.json"), os.path.join(HERE, "results.json")
    assert os.path.exists(sel)
    root = _git(HERE, "rev-parse", "--show-toplevel").stdout.strip()

    def first_commit(f):
        out = _git(root, "log", "--diff-filter=A", "--format=%H", "--", f).stdout.split()
        return out[-1] if out else None
    a = first_commit(sel)
    assert a is not None                                                                # selection.json is committed
    b = first_commit(res)
    if b is None:
        pytest.skip("results.json not committed yet")
    assert a != b and subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=root).returncode == 0


def test_selection_used_only_data_before_2024_07():
    import json
    sel = json.load(open(os.path.join(HERE, "selection.json")))
    assert pd.Timestamp(sel["data_last_bar_read"]) == pd.Timestamp("2024-06-30 23:59")
    assert pd.Timestamp(T.read_main(T.SEL_END).index.max()) < pd.Timestamp("2024-07-01")


# ------------------------------------------------------------------ the never-read span
def test_never_read_span_refused():
    idx = pd.DatetimeIndex(["2025-09-30 23:59", "2025-10-01 00:00"])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    with pytest.raises(AssertionError):
        T.check_fwd(df)
    assert len(T.check_fwd(df.iloc[1:])) == 1


def test_forward_file_loader_stays_inside_the_window():
    d = T.read_fwd()
    assert d.index.min() == pd.Timestamp("2025-10-01 00:00") and d.index.max() == T.FWD_LAST
    assert not ((d.index >= T.NEVER_LO) & (d.index <= T.NEVER_HI)).any()


def test_main_file_loader_stops_at_the_cutoff():
    d = T.read_main(None); assert d.index.max() == pd.Timestamp("2025-07-20 23:59")
