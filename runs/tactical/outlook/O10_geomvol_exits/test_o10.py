"""O10 tests (SPEC "Tests"): no exit signal uses the bar it acts on or later bars, each exit rule on hand-made bars, E4's training rows, the E0 reproduction of O9's
ROBUST trades on span B (trades only), the choice gate, the freeze and score_P refusals. pytest in this folder."""
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

HERE0 = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE0)
import o10_lib as O  # noqa: E402
import run_o10 as R  # noqa: E402
L9 = O.L9; L8 = O.L8; P5 = O.P5; P2 = O.P2; O3 = O.O3; T = O.T
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("score_P10", os.path.join(HERE0, "score_P.py")); SP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(SP)
GC = O.gcols()


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def hand(o, h, l, c, a_bars=(), slope=None, dv=None, s15=0.005, start="2024-01-01"):
    """A context from hand-made bars. Decision rows at `a_bars` (bar indices) with the given slope / dV values."""
    n = len(o); s = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=pd.date_range(start, periods=n, freq="min"))
    a = np.array(a_bars, int); nr = len(a); G = np.zeros((nr, 328), np.float32)
    if slope is not None:
        G[:, O.SLOPE_COL] = slope
    DF = {"a": a, "m": s["open"].values[a], "ta": s.index.values[a], "G": G}
    S = {"sig": np.full(nr, 0.01), "dv": np.zeros(nr) if dv is None else np.asarray(dv, float), "p": np.full(nr, 0.5), "s15": np.full(nr, s15) if np.isscalar(s15) else np.asarray(s15, float)}
    return s, DF, S, O.make_ctx(s, DF, S, GC)


def flat(n, px=100.0):
    return [np.full(n, px) for _ in range(4)]


# ------------------------------------------------------------------ the geometry feature and the causal EMA
def test_slope_column_is_the_o1_feature_g15_ema20_slope3():
    top = json.load(open(os.path.join(O.OUTLOOK, "O1_geom_vol", "interp.json")))["top"][0]
    assert top["name"] == "g15.ema20.slope3" and O.SLOPE_COL == top["col"] - O.P5.N_S3


def test_decision_slope_does_not_depend_on_bars_at_or_after_the_anchor():
    s = synth(30000); D = L8.build_dframe(s, 3000); r = len(D["a"]) // 2; a = int(D["a"][r])
    s2 = s.copy(); s2.iloc[a:] = np.random.default_rng(1).uniform(1, 1000, s2.iloc[a:].shape)
    D2 = L8.build_dframe(s2, 3000); r2 = int(np.where(D2["a"] == a)[0][0])
    assert D["G"][r, O.SLOPE_COL] == D2["G"][r2, O.SLOPE_COL] and np.array_equal(D["G"][r, GC], D2["G"][r2, GC])


def test_ema_cross_flags_are_causal_and_match_the_definition():
    s = synth(3000); S = {"sig": np.full(1, 0.01), "dv": np.zeros(1), "p": np.zeros(1), "s15": np.zeros(1)}
    DF = {"a": np.array([100]), "m": s["open"].values[[100]], "ta": s.index.values[[100]], "G": np.zeros((1, 328), np.float32)}
    ctx = O.make_ctx(s, DF, S, GC); j = 1500
    s2 = s.copy(); s2.iloc[j + 1:] = np.random.default_rng(2).uniform(1, 1000, s2.iloc[j + 1:].shape); c2 = O.make_ctx(s2, DF, S, GC)
    assert np.array_equal(ctx["evL"][:j + 1], c2["evL"][:j + 1]) and np.array_equal(ctx["evS"][:j + 1], c2["evS"][:j + 1])
    e = s["close"].ewm(span=20, adjust=False).mean().values; c = s["close"].values
    i = np.arange(1, 3000)
    assert np.array_equal(ctx["evL"][1:], (c[i] < e[i]) & (c[i - 1] >= e[i - 1])) and np.array_equal(ctx["evS"][1:], (c[i] > e[i]) & (c[i - 1] <= e[i - 1]))


# ------------------------------------------------------------------ E1: the trailing stop
@pytest.mark.parametrize("side", [1, -1])
def test_trailing_level_only_moves_in_the_positions_favour_and_never_crosses_the_fixed_stop(side):
    rng = np.random.default_rng(0); p0 = 100.0; stop = 99.0 if side == 1 else 101.0
    for _ in range(50):
        m = 40; ext = p0 + side * np.cumsum(rng.normal(0.05, 0.4, m - 1)); sg = rng.uniform(0.001, 0.01, m)
        lv = O.trail_levels(side, p0, stop, ext, sg, 1.0)
        d = np.diff(lv); assert (d >= -1e-12).all() if side == 1 else (d <= 1e-12).all()
        assert (lv >= stop).all() if side == 1 else (lv <= stop).all()
        i = 20; ext2 = ext.copy(); ext2[i:] = p0 + side * 50; sg2 = sg.copy(); sg2[i + 1:] = 0.5      # later bars cannot change the level at bar i
        assert np.array_equal(lv[:i + 1], O.trail_levels(side, p0, stop, ext2, sg2, 1.0)[:i + 1])


def test_e1_long_exits_at_the_trailing_level_and_at_the_open_when_gapped():
    o, h, l, c = flat(40)
    o[6], h[6], l[6], c[6] = 100, 101, 100, 101; o[7], h[7], l[7], c[7] = 101, 102, 101, 102; o[8], h[8], l[8], c[8] = 102, 102, 101.0, 101.2
    o[9:] = h[9:] = l[9:] = c[9:] = 101.2
    s, DF, S, ctx = hand(o, h, l, c, [0, 15, 30], s15=0.005)
    sp = O.spec_of({"exit": "E1c1", "T": 20})
    xb, xp, why, op = O.exit_trade(ctx, sp, 5, 1, 100.0, 0.01, 0.01, True)
    assert (xb, why, op) == (8, "trail", False) and xp == pytest.approx(102 * (1 - 0.005))        # the best price of the CLOSED bars before bar 8 is 102
    o2, h2, l2, c2 = [x.copy() for x in (o, h, l, c)]; o2[8], h2[8], l2[8] = 100.5, 101.0, 100.2        # gapped below the level: filled at the open
    s, DF, S, ctx2 = hand(o2, h2, l2, c2, [0, 15, 30], s15=0.005)
    xb, xp, why, _ = O.exit_trade(ctx2, sp, 5, 1, 100.0, 0.01, 0.01, True); assert (xb, why, xp) == (8, "trail", 100.5)
    spw = O.spec_of({"exit": "E1c2", "T": 20})                                                           # c = 2: the level 102 * 0.99 = 100.98 is below bar 8's low 101.0: no hit
    assert O.exit_trade(ctx, spw, 5, 1, 100.0, 0.01, 0.01, True)[:3] == (25, 101.2, "time")


def test_e1_short_mirrors_the_long_and_the_entry_bar_stop_is_pessimistic():
    o, h, l, c = flat(40)
    o[6], h[6], l[6], c[6] = 100, 100, 99, 99; o[7], h[7], l[7], c[7] = 99, 99, 98, 98; o[8], h[8], l[8], c[8] = 98, 99.0, 98, 98.8
    o[9:] = h[9:] = l[9:] = c[9:] = 98.8
    s, DF, S, ctx = hand(o, h, l, c, [0, 15, 30], s15=0.005)
    xb, xp, why, _ = O.exit_trade(ctx, O.spec_of({"exit": "E1c1", "T": 20}), 5, -1, 100.0, 0.01, 0.01, True)
    assert (xb, why) == (8, "trail") and xp == pytest.approx(98 * 1.005)
    o3, h3, l3, c3 = flat(40); l3[5] = 98.0; c3[5] = 99.5                                                 # the entry bar itself trades through a long's stop
    s, DF, S, ctx3 = hand(o3, h3, l3, c3, [0, 15, 30])
    assert O.exit_trade(ctx3, O.spec_of({"exit": "E1c1", "T": 20}), 5, 1, 100.0, 0.01, 0.01, True)[:3] == (5, 99.0, "stop")      # min(stop, close)


# ------------------------------------------------------------------ E2a, E2b, E3, E4
def _ema_bars():
    c = np.full(100, 100.0); c[30:35] = [100.5, 101, 101.5, 102, 102.5]; c[35] = 100.0
    o = np.concatenate([[100.0], c[:-1]]); h = np.maximum(o, c) + 0.1; l = np.minimum(o, c) - 0.1
    return o, h, l, c


def test_e2a_closes_at_the_next_open_after_the_cross_and_ignores_the_bars_after_the_signal():
    o, h, l, c = _ema_bars(); s, DF, S, ctx = hand(o, h, l, c, [0, 15, 30, 45])
    assert ctx["evL"][35] and not ctx["evL"][33:35].any()
    sp = O.spec_of({"exit": "E2a", "T": 60})
    xb, xp, why, op = O.exit_trade(ctx, sp, 33, 1, 101.5, 0.02, 0.01, True)
    assert (xb, xp, why, op) == (36, o[36], "ema", True)
    o2, h2, l2, c2 = [x.copy() for x in (o, h, l, c)]; h2[36:] = 500; l2[36] = 50; c2[36:] = 3                      # bar 36 and later are replaced (its open is the fill)
    s, DF, S, ctx2 = hand(o2, h2, l2, c2, [0, 15, 30, 45])
    assert O.exit_trade(ctx2, sp, 33, 1, 101.5, 0.02, 0.01, True)[:3] == (36, o[36], "ema")          # the open comes before anything that happens inside bar 36
    # a short is not closed by a cross below
    assert O.exit_trade(ctx, sp, 33, -1, 101.5, 0.02, 0.01, True)[2] != "ema"


def test_e2b_and_e3_act_only_at_decision_times_and_not_at_the_entry_bar():
    o, h, l, c = flat(200); bars = [0, 15, 30, 45, 60, 75]
    s, DF, S, ctx = hand(o, h, l, c, bars, slope=[0, -1, -1, 1, -1, 1])                                    # long: against at bars 15, 30 and 60
    sp = O.spec_of({"exit": "E2b", "T": 90})
    assert O.exit_trade(ctx, sp, 32, 1, 100.0, 0.02, 0.01, True)[:3] == (60, 100.0, "geo")                  # bar 45 favourable, bar 60 against; nothing at 33..44
    assert O.exit_trade(ctx, sp, 30, 1, 100.0, 0.02, 0.01, True)[:3] == (60, 100.0, "geo")                  # the entry bar's own decision row (bar 30) is not used
    assert O.exit_trade(ctx, sp, 32, -1, 100.0, 0.02, 0.01, True)[0] == 45                                  # a short: bar 45 points against it
    s, DF, S, ctx = hand(o, h, l, c, bars, dv=[0, -1, -1, 1, -1, 1])
    sp3 = O.spec_of({"exit": "E3", "T": 90})
    assert O.exit_trade(ctx, sp3, 32, 1, 100.0, 0.02, 0.01, True)[:3] == (60, 100.0, "vol") and O.exit_trade(ctx, sp3, 32, -1, 100.0, 0.02, 0.01, True)[:3] == (60, 100.0, "vol")
    s, DF, S, ctx = hand(o, h, l, c, bars, slope=[0, 1, 1, 1, 1, 1])
    assert O.exit_trade(ctx, sp, 32, 1, 100.0, 0.02, 0.01, True)[2] == "time"                               # never against: the time stop


def _model(icpt):
    return {"mean": np.zeros(O.NG + 5), "scale": np.ones(O.NG + 5), "coef": np.zeros(O.NG + 5), "icpt": icpt, "n": 1, "last_label_end": "x"}


def test_e4_closes_when_the_probability_is_below_one_half_with_models_by_month_and_logs_and_shuffles():
    o, h, l, c = flat(200); s, DF, S, ctx = hand(o, h, l, c, [0, 15, 30, 45, 60, 75])
    c4 = {"exit": "E4", "T": 90}
    assert O.exit_trade(ctx, O.spec_of(c4, models={"*": _model(-5.0)}), 32, 1, 100.0, 0.02, 0.01, True)[:3] == (45, 100.0, "learn")
    assert O.exit_trade(ctx, O.spec_of(c4, models={"*": _model(5.0)}), 32, 1, 100.0, 0.02, 0.01, True)[2] == "time"
    assert O.exit_trade(ctx, O.spec_of(c4, models={"2023-12": _model(-5.0)}), 32, 1, 100.0, 0.02, 0.01, True)[2] == "time"      # no model for the month: hold
    assert O.exit_trade(ctx, O.spec_of(c4, models={"2024-01": _model(-5.0)}), 32, 1, 100.0, 0.02, 0.01, True)[2] == "learn"
    log = []; O.exit_trade(ctx, O.spec_of(c4, models={"*": _model(-5.0)}, plog=log), 32, 1, 100.0, 0.02, 0.01, True); assert log == [("2024-01", pytest.approx(1 / (1 + np.exp(5))))]
    sh = O.spec_of(c4, models={"*": _model(-5.0)}, shuf={"2024-01": np.array([0.9])})                   # SHUF draws from the FULL arm's multiset: P = 0.9, no exit
    assert O.exit_trade(ctx, sh, 32, 1, 100.0, 0.02, 0.01, True)[2] == "time"


def test_e4_features_use_only_bars_before_the_checkpoint():
    s = synth(3000); DFa = np.array([0, 15, 30, 45, 60]); G = np.random.default_rng(0).normal(size=(5, 328)).astype(np.float32)
    DF = {"a": DFa, "m": s["open"].values[DFa], "ta": s.index.values[DFa], "G": G}; S = {"sig": np.full(5, 0.01), "dv": np.ones(5), "p": np.zeros(5), "s15": np.full(5, 0.005)}
    ctx = O.make_ctx(s, DF, S, GC); x = O.cp_x(ctx, 3, 45, 33, 1, 100.0, 0.01)
    s2 = s.copy(); s2.iloc[45:] = np.random.default_rng(3).uniform(1, 1000, s2.iloc[45:].shape); ctx2 = O.make_ctx(s2, DF, S, GC)
    assert np.array_equal(x, O.cp_x(ctx2, 3, 45, 33, 1, 100.0, 0.01)) and x.shape == (O.NG + 5,)
    assert np.array_equal(x[:O.NG], G[3][GC].astype(float)) and np.array_equal(O.cp_x(ctx, 3, 45, 33, -1, 100.0, 0.01)[:O.NG], -x[:O.NG])        # geometry signed by the side


def test_e4_records_labels_ends_and_checkpoint_range():
    o, h, l, c = flat(140); o[60] = 101.0; o[75] = 99.0; s, DF, S, ctx = hand(o, h, l, c, range(0, 140, 15))
    tr = {"ebar": np.array([32, 32]), "xbar": np.array([70, 90]), "side": np.array([1, 1]), "epx": np.array([100.0, 100.0]), "xpx": np.array([98.0, 100.0]),
          "reason": np.array(["stop", "time"]), "sig": np.array([0.01, 0.01])}
    rec = O.e4_records(ctx, tr)
    # trade 1 (stop in bar 70): checkpoints at 45, 60 (bars 32 < a <= 70); trade 2 (time exit at the open of 90): 45, 60, 75
    assert len(rec["y"]) == 5
    assert list(rec["y"]) == [True, False, True, False, True]     # trade 1: 45->60 up (100->101), 60->exit down (101->98); trade 2: 45->60 up, 60->75 down (101->99), 75->exit up (99->100)
    ix = ctx["idx"]
    assert rec["end"][0] == ix[60] and rec["end"][1] == ix[70] + np.timedelta64(1, "m") and rec["end"][4] == ix[90]


def test_e4_fit_uses_only_labels_known_before_the_month_and_needs_enough_rows():
    rng = np.random.default_rng(0); m0 = pd.Timestamp("2024-02-01 00:00"); n = 450
    end = np.concatenate([(pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 40000, 400), "m")).values, [np.datetime64(m0, "ns")] * 20,
                          (m0 + pd.to_timedelta(rng.integers(0, 4000, 30), "m")).values]).astype("datetime64[ns]")
    rec = {"X": rng.normal(size=(n, O.NG + 5)), "y": rng.random(n) > 0.5, "end": end}
    a = O.e4_fit(rec, m0); assert a["n"] == 400 and pd.Timestamp(a["last_label_end"]) < m0
    rec2 = {k: v.copy() for k, v in rec.items()}; rec2["X"][400:] = 99.0; rec2["y"][400:] = ~rec2["y"][400:]
    b = O.e4_fit(rec2, m0); assert np.array_equal(a["coef"], b["coef"]) and a["icpt"] == b["icpt"]
    small = {k: v[:100] for k, v in rec.items()}; assert O.e4_fit(small, m0) is None
    ms = O.e4_models(rec, ["2024-02", "2024-03"]); assert set(ms) == {"2024-02", "2024-03"} and ms["2024-03"]["n"] > ms["2024-02"]["n"]


# ------------------------------------------------------------------ the engine
def _syn_ctx(n=20000, seed=0, p_lo=0.4, p_hi=0.6):
    s = synth(n, seed=seed); a = np.arange(100, n - 400); a = a[a % 15 == 0]; rng = np.random.default_rng(seed + 1); nr = len(a)
    G = rng.normal(size=(nr, 328)).astype(np.float32); DF = {"a": a, "m": s["open"].values[a], "ta": s.index.values[a], "G": G}
    S = {"sig": np.full(nr, 0.003) * rng.uniform(0.8, 1.2, nr), "dv": rng.normal(size=nr), "p": rng.uniform(p_lo, p_hi, nr), "s15": np.full(nr, 0.0015) * rng.uniform(0.8, 1.2, nr)}
    return s, DF, S, O.make_ctx(s, DF, S, GC)


def test_every_exit_kind_trades_and_exits_are_causal_end_to_end():
    s, DF, S, ctx = _syn_ctx(); rows = np.arange(len(DF["a"])); aL, aS = L9.gate_masks(S["p"], S["dv"], 0.02, None)
    for c in O.grid():
        sp = O.spec_of(c, models={"*": _model(-0.2)} if c["exit"] == "E4" else None); tr = O.simulate10(ctx, rows, sp, S["sig"], S["sig"], aL, aS)
        assert len(tr["dec"]) > 30, c
        assert (tr["xbar"] >= tr["ebar"]).all() and (tr["xbar"] - tr["ebar"] <= c["T"]).all()
        assert np.isfinite(tr["ret"]).all()
    # a signal exit at the open of bar x depends on the bars BEFORE x (and the open itself as the fill price), not on the bars from x on
    sp = O.spec_of({"exit": "E2a", "T": 60}); tr = O.simulate10(ctx, rows, sp, S["sig"], S["sig"], aL, aS); k = np.where(tr["reason"] == "ema")[0]; assert len(k) > 10
    for i in k[:10]:
        x = int(tr["xbar"][i]); s2 = s.copy(); j = s2.columns.get_indexer(["high", "low", "close"]); s2.iloc[x:, j] = np.random.default_rng(5).uniform(1, 1000, (len(s2) - x, 3))
        s2.iloc[x + 1:, 0] = np.random.default_rng(6).uniform(1, 1000, len(s2) - x - 1)
        tr2 = O.simulate10(O.make_ctx(s2, DF, S, GC), rows, sp, S["sig"], S["sig"], aL, aS); k2 = np.where(tr2["ebar"] == int(tr["ebar"][i]))[0]
        assert len(k2) and tr2["xbar"][k2[0]] == x and tr2["reason"][k2[0]] == "ema" and tr2["xpx"][k2[0]] == tr["xpx"][i]
    for kind, key in (("E2b", "slope"), ("E3", "dvx")):                                                  # decision-time signals: rows after the exit row are irrelevant
        sp = O.spec_of({"exit": kind, "T": 60}); tr = O.simulate10(ctx, rows, sp, S["sig"], S["sig"], aL, aS); k = np.where(tr["reason"] == ("geo" if kind == "E2b" else "vol"))[0]; assert len(k) > 10
        for i in k[:10]:
            x = int(tr["xbar"][i]); r = int(ctx["barrow"][x]); c2 = dict(ctx); c2[key] = ctx[key].copy(); c2[key][r + 1:] = np.random.default_rng(7).normal(size=len(c2[key]) - r - 1)
            tr2 = O.simulate10(c2, rows, sp, S["sig"], S["sig"], aL, aS); k2 = np.where(tr2["ebar"] == int(tr["ebar"][i]))[0]
            assert len(k2) and tr2["xbar"][k2[0]] == x and tr2["xpx"][k2[0]] == tr["xpx"][i]


def test_a_forbidden_side_places_no_order_and_one_position_at_a_time():
    s, DF, S, ctx = _syn_ctx(); rows = np.arange(len(DF["a"])); n = len(rows)
    sp = O.spec_of({"exit": "E0", "T": 60}); none = np.zeros(n, bool); allow = np.ones(n, bool)
    assert len(O.simulate10(ctx, rows, sp, S["sig"], S["sig"], none, none)["dec"]) == 0
    tr = O.simulate10(ctx, rows, sp, S["sig"], S["sig"], allow, none); assert (tr["side"] == 1).all() and len(tr["dec"]) > 30
    assert (tr["ebar"][1:] >= tr["xbar"][:-1]).all()                                                    # flat before the next entry


def test_a_data_gap_inside_the_order_and_holding_window_removes_the_decision():
    s = synth(20000); s2 = s.drop(s.index[5000:5003]); a = np.arange(100, len(s2) - 400); a = a[a % 15 == 0]; nr = len(a); rng = np.random.default_rng(1)
    DF = {"a": a, "m": s2["open"].values[a], "ta": s2.index.values[a], "G": rng.normal(size=(nr, 328)).astype(np.float32)}
    S = {"sig": np.full(nr, 0.003), "dv": np.zeros(nr), "p": np.full(nr, 0.5), "s15": np.full(nr, 0.0015)}; ctx = O.make_ctx(s2, DF, S, GC); rows = np.arange(nr); allow = np.ones(nr, bool)
    for Tt in (60, 240):
        tr = O.simulate10(ctx, rows, O.spec_of({"exit": "E0", "T": Tt}), S["sig"], S["sig"], allow, allow); assert len(tr["dec"]) > 30
        assert not ((tr["dec"] > 5000 - 15 - Tt) & (tr["dec"] < 5000)).any(), Tt                    # the window decision .. decision + 15 + T would cross the gap
        assert ((ctx["tmin"][tr["xbar"]] - ctx["tmin"][tr["ebar"]]) == (tr["xbar"] - tr["ebar"])).all()


def test_e0_reproduces_o9_robust_trades_on_span_b():
    """Same entries, exits and returns as O9's ROBUST/FULL PESS trades of span B (a comparison of trades only; no performance number is read or printed)."""
    ref_p = os.path.join(O.O9DIR, "trades_B.csv.gz"); sig_p = os.path.join(R.O9SCR, "sig_B.npz")
    if not (os.path.exists(ref_p) and os.path.exists(sig_p) and os.path.exists(os.path.join(L8.SCRATCH, "dframe.npz"))):
        pytest.skip("O9 caches are not on this machine")
    s = L8.load_bars_cut(); DF = L8.load_dframe(); z = np.load(sig_p); months = R.span_months("B")
    S = {"sig": z["L1"], "dv": z["dv"], "p": z["p"], "s15": np.full(len(z["L1"]), np.nan)}; ctx = O.make_ctx(s, DF, S, GC)
    rows = np.where(np.isin(ctx["mo_row"], months) & np.isfinite(S["sig"]))[0]; aL, aS = L9.gate_masks(S["p"], S["dv"], 0.02, None)
    ref = pd.read_csv(ref_p, float_precision="round_trip"); ref = ref[(ref.arm == "ROBUST/FULL") & (ref.fills == "PESS")].reset_index(drop=True)
    tf = O.trades10(ctx, rows, O.spec_of({"exit": "E0", "T": 60}), S, S["sig"], aL, aS, True)[0]
    assert len(tf) == len(ref) > 100
    assert (pd.to_datetime(ref["decision_time"]).values == tf["decision_time"].values).all() and (pd.to_datetime(ref["entry_time"]).values == tf["entry_time"].values).all()
    assert (pd.to_datetime(ref["exit_time"]).values == tf["exit_time"].values).all() and (ref["reason"].values == tf["reason"].values).all()
    for k in ("side", "entry_px", "exit_px", "ret_gross", "sigma"):
        assert np.array_equal(ref[k].values, tf[k].values), k


# ------------------------------------------------------------------ SHUF, grid, statistics
def test_grid_is_the_14_configurations_with_distinct_names_and_e1_parses_c():
    g = O.grid(); assert len(g) == 14 and len({O.cfg_name(c) for c in g}) == 14 and {c["T"] for c in g} == {60, 240}
    assert O.spec_of({"exit": "E1c2", "T": 60})["c"] == 2 and O.spec_of({"exit": "E1c1", "T": 60})["kind"] == "E1" and O.spec_of({"exit": "E2b", "T": 60})["c"] is None


def test_shuf_permutes_each_exit_signal_within_the_month_and_leaves_the_rest():
    s, DF, S, ctx = _syn_ctx(60000); months = ["2024-01", "2024-02"]; mo = ctx["mo_row"]
    for kind, key in (("E1", "s15"), ("E2b", "slope"), ("E3", "dvx")):
        c2, sx = O.make_shuf(ctx, S["sig"], kind, months)
        assert np.array_equal(sx, S["sig"])
        for m in months:
            r = mo == m
            assert np.array_equal(np.sort(c2[key][r]), np.sort(ctx[key][r])) and not np.array_equal(c2[key][r], ctx[key][r])
        for other in ("s15", "slope", "dvx"):
            if other != key:
                assert np.array_equal(c2[other], ctx[other])
    c2, _ = O.make_shuf(ctx, S["sig"], "E2a", months)
    for k in ("evL", "evS"):
        for m in months:
            lo, hi = np.searchsorted(ctx["idx"], np.datetime64(P5.month_bounds(m)[0], "ns")), np.searchsorted(ctx["idx"], np.datetime64(P5.month_bounds(m)[1], "ns"))
            assert c2[k][lo:hi].sum() == ctx[k][lo:hi].sum() and not np.array_equal(c2[k][lo:hi], ctx[k][lo:hi])
    c0, sx0 = O.make_shuf(ctx, S["sig"], "E0", months)                                                    # E0: the exit levels' sigma is permuted, the entries keep the real sigma
    assert not np.array_equal(sx0, S["sig"]) and np.array_equal(np.sort(sx0[mo == "2024-01"]), np.sort(S["sig"][mo == "2024-01"]))
    a, b = O.make_shuf(ctx, S["sig"], "E1", months)[0]["s15"], O.make_shuf(ctx, S["sig"], "E1", months)[0]["s15"]; assert np.array_equal(a, b)      # seeded


def test_paired_monthly_and_summary_on_a_hand_made_frame():
    x = np.array([0.01, 0.02, -0.01, 0.03]); r = O.paired_monthly(x, np.zeros(4)); assert r["mean"] == pytest.approx(0.0125) and r["months_better"] == 3 and r["lo"] < r["mean"] < r["hi"]
    tf = pd.DataFrame({"entry_time": pd.to_datetime(["2024-01-05", "2024-01-09", "2024-02-02"]), "exit_time": pd.to_datetime(["2024-01-05", "2024-01-09", "2024-02-02"]), "ret_gross": [0.01, -0.005, 0.004],
                       "w": 1.0, "side": [1, -1, 1], "reason": ["trail", "stop", "time"], "hold": [30, 10, 60]})
    d = O.summarise10(tf, ["2024-01", "2024-02"])
    assert d["n_trades"] == 3 and d["trades_per_month"] == 1.5 and d["mean_hold_min"] == pytest.approx(100 / 3) and d["hit_2bps"] == pytest.approx(2 / 3)
    assert d["avg_win_bps"] == pytest.approx(((0.01 - 2e-4) + (0.004 - 2e-4)) / 2 * 1e4) and d["avg_loss_bps"] == pytest.approx(-52.0) and d["reasons"] == {"trail": 1, "stop": 1, "time": 1}
    assert d["mean_m_2bps"] == pytest.approx((0.01 - 0.005 - 4e-4 + 0.004 - 2e-4) / 2)


# ------------------------------------------------------------------ the gate, the freeze, the scoring
def _mini_repo(tmp_path, monkeypatch):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (tmp_path / "keep.txt").write_text("x"); g("add", "keep.txt"); g("commit", "-q", "-m", "init")
    monkeypatch.setattr(R, "HERE", str(tmp_path)); monkeypatch.setattr(R, "CHOICE", str(tmp_path / "choice.json"))
    return g


def test_choice_gate_refuses_describe_and_freeze_without_a_committed_unchanged_choice(tmp_path, monkeypatch):
    g = _mini_repo(tmp_path, monkeypatch)
    for fn in (R.describe, R.freeze, R.choice_gate):
        with pytest.raises(SystemExit, match="choice.json is not tracked"):
            fn()
    (tmp_path / "choice.json").write_text(json.dumps({"config": "x"}))
    with pytest.raises(SystemExit, match="not tracked"):
        R.describe()
    g("add", "choice.json")
    with pytest.raises(SystemExit, match="not in HEAD|differs"):
        R.freeze()
    g("commit", "-q", "-m", "c1"); assert R.choice_gate()["config"] == "x"
    (tmp_path / "choice.json").write_text(json.dumps({"config": "z"}))
    with pytest.raises(SystemExit, match="choice.json differs"):
        R.describe()


def test_freeze_reads_no_bar_after_the_cut_and_the_cut_file_refuses_the_forbidden_gap(tmp_path):
    import inspect
    assert "load_freeze_bars" in inspect.getsource(R.freeze) and "assert bars.index.max() <= P5.B_HI" in inspect.getsource(R.freeze)
    n = int((P5.B_HI - P5.B_LO).total_seconds() // 60) + 1 + 3000
    idx = pd.date_range(P5.B_LO, periods=n, freq="min"); rng = np.random.default_rng(0); c = 100 + np.cumsum(rng.normal(0, 0.01, n))
    f = tmp_path / "fwd.csv"
    pd.DataFrame({"timestamp": idx, "open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1.0}).to_csv(f, index=False)
    b = P5.load_freeze_bars(str(f)); assert b.index.max() == P5.B_HI and len(b) == n - 3000
    idx2 = pd.date_range("2025-07-19 23:00", periods=40 * 60, freq="min")
    g = tmp_path / "cut.csv"; pd.DataFrame({"timestamp": idx2, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(g, index=False)
    with pytest.raises(AssertionError):
        T.load_bars(str(g), pd.Timestamp("2025-07-19 23:00"), pd.Timestamp("2025-07-22 00:00"))


@pytest.fixture(scope="module")
def frozen_synth():
    s = synth(61000); m0 = pd.Timestamp("2024-02-05 00:00"); old = O.E4_MIN; O.E4_MIN = 20
    try:
        art = O.freeze_fit10(s[s.index < m0], m0, "E4_T60")
    finally:
        O.E4_MIN = old
    return s, m0, art


def test_artefact_roundtrip_tamper_and_no_label_after_the_cut(frozen_synth):
    s, m0, art = frozen_synth; art_j = json.loads(json.dumps(art))
    assert P5.sha_of(art_j) == art["sha256"] and art["o9_frozen_sha256"] != art["sha256"]
    for k in ("sigma", "dv", "pup", "sigma15"):
        assert pd.Timestamp(art_j[k]["last_label_end"]) <= m0
    assert pd.Timestamp(art["last_bar"]) < m0 and set(art["e4"]) == {"60", "240"}
    e4 = [j for j in art["e4"].values() if j is not None]; assert e4, "no E4 model on the synthetic bars"
    for j in e4:
        assert pd.Timestamp(j["last_label_end"]) < m0 and len(j["coef"]) == O.NG + 5
    DF, DX = P2.make_dframe_for(s, 6145)
    S1 = O.signals_frozen10(O.load_artefact10(art), DF, DX, ["2024-02"]); S2 = O.signals_frozen10(O.load_artefact10(art_j), DF, DX, ["2024-02"])
    for k in S1:
        assert np.array_equal(S1[k], S2[k], equal_nan=True)
    assert np.isfinite(S1["s15"]).sum() > 100 and np.isfinite(S1["sig"]).sum() > 100
    bad = json.loads(json.dumps(art)); bad["e4"]["60"]["icpt"] += 1e-9
    with pytest.raises(ValueError, match="sha256"):
        O.load_artefact10(bad)


def test_freeze_is_independent_of_bars_after_the_cut(frozen_synth):
    s, m0, art = frozen_synth
    s2 = s.copy(); s2.loc[s2.index >= m0, :] = np.random.default_rng(2).uniform(1, 1000, s2[s2.index >= m0].shape)
    old = O.E4_MIN; O.E4_MIN = 20
    try:
        a2 = O.freeze_fit10(s2[s2.index < m0], m0, "E4_T60")
    finally:
        O.E4_MIN = old
    assert a2["sha256"] == art["sha256"]


def test_complete_months_and_score_refusals(tmp_path, monkeypatch):
    t = pd.Timestamp
    assert SP.complete_months(t("2027-02-01 01:15"), 60) == [pd.Period(m, "M") for m in ("2026-11", "2026-12", "2027-01")]
    assert len(SP.complete_months(t("2027-02-01 01:14"), 60)) == 2 and SP.complete_months(t("2026-10-12 23:59"), 240) == []
    assert len(SP.complete_months(t("2027-02-01 04:14"), 240)) == 2 and len(SP.complete_months(t("2027-02-01 04:15"), 240)) == 3          # T 240 needs 255 minutes of tail
    if os.path.exists(T.FWD_CSV) and os.path.exists(SP.FROZEN) and os.path.exists(SP.CHOICE):
        monkeypatch.setattr(sys, "argv", ["score_P.py"]); monkeypatch.setattr(SP.T, "load_bars", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a bar was read")))
        with pytest.raises(SystemExit, match="REFUSED: 0 complete months"):
            SP.main()
    res = tmp_path / "results_P.json"; res.write_text("{}"); monkeypatch.setattr(SP, "RESULTS", str(res))
    with pytest.raises(SystemExit, match="exists"):
        SP.main()
    monkeypatch.setattr(SP, "RESULTS", str(tmp_path / "none.json"))
    s = synth(61000); m0 = pd.Timestamp("2024-02-05 00:00"); art = P2.freeze_fit(s[s.index < m0], m0); art["m0"] = "tampered"
    fz = tmp_path / "frozen_P.json"; fz.write_text(json.dumps(art)); monkeypatch.setattr(SP, "FROZEN", str(fz))
    with pytest.raises(SystemExit, match="sha256"):
        SP.main()


@pytest.mark.parametrize("cfg", [{"exit": "E1c1", "T": 240}, {"exit": "E4", "T": 60}, {"exit": "E2a", "T": 60}, {"exit": "E0", "T": 60}])
def test_score_core_on_synthetic_bars_structure_and_paired_series(cfg):
    s = synth(110000, "2026-06-10"); m0 = pd.Timestamp("2026-07-10 00:00"); old = O.E4_MIN; O.E4_MIN = 20
    try:
        art = O.freeze_fit10(s[s.index < m0], m0, O.cfg_name(cfg))
    finally:
        O.E4_MIN = old
    res = SP.score_core(art, s, cfg, ["2026-07", "2026-08"])
    assert res["months"] == ["2026-07", "2026-08"] and res["c1_CHOICE"]["n_days"] == 62 and res["CHOICE_minus_E0"]["n_days"] == 62
    assert set(res) >= {"c1_CHOICE", "CHOICE_minus_E0", "CHOICE_minus_SHUF", "exits_add", "strategy_works", "owner_target_met"} and ("CHOICE_minus_E0T" in res) == (cfg["T"] != 60)
    assert res["stats"]["CHOICE"]["n_trades"] > 5 and res["stats"]["E0"]["n_trades"] > 5 and res["stats"]["SHUF"]["n_trades"] >= 0
    if cfg["exit"] == "E0":                                                                              # the choice is E0: it adds nothing over itself
        assert res["CHOICE_minus_E0"]["mean"] == 0 and res["exits_add"] is False
