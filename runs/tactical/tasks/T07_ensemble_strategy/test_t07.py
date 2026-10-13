"""T07 tests (SPEC "Tests"): next-open fills, thresholds from the previous month only, models fitted only before the month, the never-read span,
selection.json committed before the looks, costs on turnover, the null keeps the exposure path (+ the combination rules)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pytest

import t07_lib as T
import members as MB


def synth_raw(start="2020-01-01", hours=24 * 30, seed=0, drift=0.0):
    rng = np.random.default_rng(seed); n = hours * 60
    idx = pd.date_range(start, periods=n, freq="min")
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 2e-4, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": np.ones(n)}, index=idx)


# ------------------------------------------------------------------ next-open fills
def test_next_open_fill_hourly_member():
    g = T.Grid(synth_raw(hours=48))
    t_sig = pd.DatetimeIndex([pd.Timestamp("2020-01-01 05:00")])                      # window's last bar opens 05:00 and closes 05:59
    t_entry = (t_sig + pd.Timedelta(hours=1)).values.astype("datetime64[m]")        # the next bar's open: 06:00
    pos, tr = MB.trades_to_position(g, np.array([0.9]), t_entry, np.array(["2020-01"]), {"2020-01": 0.1}, 4 * 60)
    e = g.minute_of(np.datetime64("2020-01-01T06:00"))
    assert pos[e - 1] == 0 and pos[e] == 1 and pos[e + 239] == 1 and pos[e + 240] == 0
    gross = (pos * g.r).sum()                                                          # earns exactly the minutes after the fill
    assert gross == pytest.approx(g.r[e:e + 240].sum())
    assert g.r[e - 1] != 0 and (pos[:e] == 0).all()                                    # the signal bar's own return is never earned


def test_next_open_fill_rules_use_closed_hours_only():
    g = T.Grid(synth_raw(hours=400, seed=3))
    for f in (T.pos_trend, T.pos_meanrev, T.pos_breakout):
        full = f(g.hc)
        hc2 = g.hc.copy(); hc2[200:] *= 1.5                                           # change every close from hour 200 on
        alt = f(hc2)
        assert (full[:201] == alt[:201]).all()                                        # position of hour k depends on closes of hours < k


# ------------------------------------------------------------------ thresholds
def test_thresholds_from_previous_month_only():
    rng = np.random.default_rng(0)
    conf = {m: rng.random(500) * 0.2 for m in ("2020-01", "2020-02", "2020-03")}
    thr = T.month_thresholds(conf, ["2020-01", "2020-02", "2020-03"])
    assert "2020-01" not in thr and thr["2020-02"] == pytest.approx(np.quantile(conf["2020-01"], 0.9)) and thr["2020-03"] == pytest.approx(np.quantile(conf["2020-02"], 0.9))
    conf2 = dict(conf); conf2["2020-02"] = conf["2020-02"] * 2                         # change month 2 only
    thr2 = T.month_thresholds(conf2, ["2020-01", "2020-02", "2020-03"])
    assert thr2["2020-02"] == thr["2020-02"] and thr2["2020-03"] != thr["2020-03"]
    # a month without a threshold (the first) takes no trade
    ent, ex, side = T.signal_trades(np.array(["2020-01-05T00:00"], "datetime64[m]"), np.array([0.99]), np.array(["2020-01"]), thr, 60)
    assert len(ent) == 0


def test_signal_trades_non_overlapping_and_side():
    t = np.array(["2020-02-01T00:00", "2020-02-01T01:00", "2020-02-01T04:00"], "datetime64[m]")
    ent, ex, side = T.signal_trades(t, np.array([0.9, 0.1, 0.8]), np.array(["2020-02"] * 3), {"2020-02": 0.2}, 4 * 60)
    assert list(ent) == [t[0], t[2]] and list(side) == [1, 1]                           # the 01:00 short is skipped while the 00:00 long is open


# ------------------------------------------------------------------ models fitted only before the month
def hourly_features(seed=0, n_days=900):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-01-01", periods=24 * n_days, freq="h")
    X = rng.normal(size=(len(t), 5)).astype(np.float32)
    r4 = 0.002 * X[:, 0] + rng.normal(0, 0.01, len(t)); r1 = 0.001 * X[:, 0] + rng.normal(0, 0.005, len(t))
    return dict(X=X, t_sig=t, valid=np.ones(len(t), bool), r={4: r4, 1: r1})


@pytest.mark.parametrize("kind,H,Td", [("logreg", 4, 730), ("l1", 1, 365)])
def test_hourly_model_sees_only_labels_closed_before_the_month(kind, H, Td):
    F = hourly_features(); month = "2020-04"; m0 = pd.Period(month, "M").start_time
    base = MB.hourly_month_preds(F, H, kind, Td, [month])[month]
    F2 = {**F, "r": {k: v.copy() for k, v in F["r"].items()}, "X": F["X"].copy()}
    late = np.asarray(pd.DatetimeIndex(F["t_sig"]) + pd.Timedelta(hours=H + 1) > m0)   # labels not closed by the month start
    F2["r"][H][late] = -F2["r"][H][late] * 5 + 0.01                                    # destroy them
    again = MB.hourly_month_preds(F2, H, kind, Td, [month])[month]
    assert np.array_equal(base[0], again[0]) and np.allclose(base[1], again[1], atol=1e-12)
    # and the training rows really end before the month
    import wf
    tr, te = wf.split_masks(pd.DatetimeIndex(F["t_sig"]), m0, (pd.Period(month, "M") + 1).start_time, H, 60, Td)
    assert (pd.DatetimeIndex(F["t_sig"])[tr].max() + pd.Timedelta(hours=H + 1)) <= m0 and not (tr & te).any()


def test_m1_train_rows_end_before_the_month():
    import m1_core
    t = pd.date_range("2020-01-01", periods=60 * 24 * 120, freq="10min").values.astype("datetime64[m]")
    m0 = pd.Timestamp("2020-04-01")
    tr = m1_core.train_mask(t, m0)
    assert pd.Timestamp(t[tr].max()) + pd.Timedelta(minutes=16) <= m0                  # entry +1, exit +16 (15-minute label bar)
    assert pd.Timestamp(t[tr].min()) >= m0 - pd.Timedelta(days=60)


# ------------------------------------------------------------------ the never-read span
def test_never_read_span_refused():
    idx = pd.DatetimeIndex(["2025-09-30 23:59", "2025-10-01 00:00"])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    with pytest.raises(AssertionError):
        T.check_fwd(df)
    ok = df.iloc[1:]
    assert len(T.check_fwd(ok)) == 1


def test_forward_file_loader_stays_inside_the_window():
    d = T.read_fwd()
    assert d.index.min() == pd.Timestamp("2025-10-01 00:00") and d.index.max() == T.FWD_LAST
    assert not ((d.index >= T.NEVER_LO) & (d.index <= T.NEVER_HI)).any()


def test_selection_loader_stops_before_2024_07():
    d = T.read_main(T.SEL_END)
    assert d.index.max() < pd.Timestamp("2024-07-01") and d.index.max() == pd.Timestamp("2024-06-30 23:59")


# ------------------------------------------------------------------ selection.json before the looks
def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True)


def test_selection_must_be_committed_before_looks(tmp_path):
    _git(tmp_path, "init", "-q"); _git(tmp_path, "config", "user.email", "t@t"); _git(tmp_path, "config", "user.name", "t")
    p = tmp_path / "selection.json"; p.write_text('{"chosen": "C1"}')
    with pytest.raises(AssertionError, match="not tracked"):
        T.assert_selection_committed(str(p))
    _git(tmp_path, "add", "selection.json")
    with pytest.raises(AssertionError, match="modified or staged"):
        T.assert_selection_committed(str(p))
    _git(tmp_path, "commit", "-q", "-m", "x")
    assert len(T.assert_selection_committed(str(p))) == 64
    p.write_text('{"chosen": "C2"}')
    with pytest.raises(AssertionError, match="modified or staged"):
        T.assert_selection_committed(str(p))


def test_selection_commit_precedes_results_commit():
    sel, res = os.path.join(HERE, "selection.json"), os.path.join(HERE, "results.json")
    if not (os.path.exists(sel) and os.path.exists(res)):
        pytest.skip("no results yet")
    root = _git(HERE, "rev-parse", "--show-toplevel").stdout.strip()

    def first_commit(f):
        out = _git(root, "log", "--diff-filter=A", "--format=%H", "--", f).stdout.split()
        return out[-1] if out else None
    a, b = first_commit(sel), first_commit(res)
    if b is None:
        pytest.skip("results.json not committed yet")
    assert a is not None and a != b
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=root)
    assert anc.returncode == 0                                                         # selection.json's commit is an ancestor of results.json's


# ------------------------------------------------------------------ costs on turnover
def test_costs_charged_on_turnover():
    g = T.Grid(synth_raw(hours=10))
    zero = np.zeros(g.n); w = zero.copy(); w[60:180] = 1.0                              # in at hour 1, out at hour 3: a round trip
    gr, u, _ = T.book(w, g.r)
    assert u.sum() == pytest.approx(2.0)
    for c in (0, 1, 2, 5):
        assert T.net(gr, u, c).sum() == pytest.approx(gr.sum() - c * 1e-4)             # c bps per round trip = c/2 per unit of turnover
    w2 = zero.copy(); w2[60:120] = 1.0; w2[120:180] = -1.0                              # a flip costs two units
    _, u2, _ = T.book(w2, g.r)
    assert u2.sum() == pytest.approx(1 + 2 + 1)
    const = np.ones(g.n); _, u3, _ = T.book(const, g.r)
    assert u3.sum() == pytest.approx(1.0)                                              # holding costs nothing after the entry
    st = T.trade_stats(w, g.r, 2)
    assert st["n_trades"] == 1


# ------------------------------------------------------------------ the null keeps the exposure path
def test_null_flips_signs_not_exposure():
    g = T.Grid(synth_raw(hours=24 * 20, seed=5))
    rng = np.random.default_rng(1)
    w = np.zeros(g.n)
    for s in range(20):                                                                 # 20 trades of different signs and sizes
        a = 600 + s * 1400; w[a:a + 500] = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
    s_, rid, st = T.runs_of(w); nr = int(rid.max())
    assert nr == 20
    fl = np.where(rng.random((7, nr + 1)) < 0.5, 1.0, -1.0)
    for k in range(7):
        wf_ = T.flip_runs(w, np.concatenate([[1.0], fl[k, 1:]]))
        assert np.array_equal(np.abs(wf_), np.abs(w))                                   # same exposure path
    cum = T.null_cum(w, g, 2, slice(0, g.nh), flips=fl)
    for k in range(7):
        wf_ = T.flip_runs(w, np.concatenate([[1.0], fl[k, 1:]]))
        gr, _, _ = T.book(wf_, g.r); _, u, _ = T.book(w, g.r)                            # cost of the ORIGINAL path
        ref = np.expm1(np.log1p(T.net(gr, u, 2)).sum())
        assert cum[k] == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_null_p95_below_a_perfect_foresight_path():
    g = T.Grid(synth_raw(hours=24 * 20, seed=6))
    sign = np.sign(g.r); sign[sign == 0] = 1
    w = np.zeros(g.n)
    for a in range(0, g.n - 60, 120):
        w[a:a + 60] = np.sign(g.r[a:a + 60].sum())                                       # an oracle side per 60-minute trade
    cum = T.null_cum(w, g, 2, slice(0, g.nh), draws=200)
    gr, u, _ = T.book(w, g.r)
    actual = np.expm1(np.log1p(T.net(gr, u, 2)).sum())
    assert actual > np.quantile(cum, 0.95) and abs(np.median(cum)) < abs(actual)


# ------------------------------------------------------------------ combinations
def test_vote_needs_three_agreeing_members():
    p = np.array([[1, 1, 1, 1], [1, 1, 0, -1], [1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], float)
    v = T.vote_hourly(p)
    assert list(v) == [1.0, 0.0, 0.0, 0.0]                                              # hour 0: three agree; hour 1: two only; hour 3: sum -2 but two agree
    p2 = np.array([[1, 1], [1, 1], [1, -1], [-1, 0], [0, 0], [0, 0]], float)
    assert list(T.vote_hourly(p2)) == [1.0, 0.0]                                        # hour 0: sum +2, three agree with it


def test_inverse_loss_weight_zero_for_a_losing_member():
    rng = np.random.default_rng(0); H = 4000
    nets = rng.normal(0, 0.002, (3, H)); nets[0] += 0.0004; nets[1] -= 0.0006; nets[2] += 0.0002
    a = T.inverse_loss_weights(nets)
    k = H - 10
    assert a[1, k] == 0.0 and a[0, k] > 0 and a[2, k] > 0
    assert a[:, 100].sum() > 0                                                          # before 14 days the equal-risk fallback


def test_equal_risk_scales_to_the_same_vol_and_cap():
    rng = np.random.default_rng(1); H = 3000
    nets = np.stack([rng.normal(0, 0.001, H), rng.normal(0, 0.003, H), rng.normal(0, 0.002, H)])
    a = T.equal_risk_weights(nets)
    k = H - 1
    contrib = a[:, k] * 3 * np.array([nets[i, k - 720:k].std() for i in range(3)])     # a_i * M * sigma_i = sigma_ref
    assert np.allclose(contrib, contrib[0], rtol=1e-9)
    tiny = nets.copy(); tiny[0] *= 1e-4
    assert T.equal_risk_weights(tiny)[0, k] * 3 <= T.SCALE_CAP + 1e-12


def test_gross_cap_and_har_scale():
    pos = np.ones((3, 120), np.int8); a = np.full((3, 2), 1.0)
    assert np.abs(T.combine_linear(pos, a)).max() == 1.0                                # capped at 1x notional
    s = T.har_scale(np.array([1.0, 4.0, np.nan]), 2.0)
    assert list(s) == [1.0, 0.5, 1.0]


def test_rules_are_causal_on_a_prefix():
    g = T.Grid(synth_raw(hours=600, seed=9))
    for f in (T.pos_trend, T.pos_meanrev, T.pos_breakout):
        full = f(g.hc); part = f(g.hc[:400])
        assert (full[:400] == part).all()


def test_rsi_meanrev_hysteresis():
    hc = np.concatenate([np.linspace(100, 140, 60), np.linspace(140, 60, 80), np.linspace(60, 130, 120)])
    p = T.pos_meanrev(hc)
    assert set(np.unique(p)) <= {-1, 0, 1}
    i = np.flatnonzero(p == -1)
    assert len(i) and (p[i[0]:i[0] + 1] == -1).all()


def test_minute_vote_diagnostic_matches_the_hourly_rule_on_constant_positions():
    pos = np.repeat(np.array([[1, 1, 0], [1, 1, 0], [1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]], np.int8), 60, axis=1)
    assert list(T.vote_minute(pos)[::60]) == [1.0, 0.0, 0.0]
    assert list(T.vote_hourly(pos.reshape(6, 3, 60).mean(2))) == list(T.vote_minute(pos)[::60])
