"""Tests of the hill-climb v2 (SPEC.md). usage: python -m pytest runs/tactical/hc2/test_hc2.py -q
Data tests need the caches built by run_preds.py / run_paths.py (D:/nt/nt_hc2_cache) and are skipped when they are missing."""
import os, sys
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hc2lib as H  # noqa: E402

MO = H.MO
HAVE_DEV = all(os.path.exists(H.pred_path("dev", ("rich", "logreg", 730, h))) for h in H.HS) and all(os.path.exists(H.paths_file("dev", h)) for h in H.HS)
needs_dev = pytest.mark.skipif(not HAVE_DEV, reason="dev caches not built")


# ------------------------------------------------------------------ the lockbox draw
def test_lockbox_is_12_months_two_per_year_and_never_the_spent_period():
    lb = list(MO.LOCKBOX)
    assert len(lb) == 12 and lb == sorted(lb)
    for y in range(2019, 2025):
        assert sum(m.startswith(str(y)) for m in lb) == 2
    assert all("2019-01" <= m <= "2024-06" for m in lb)          # 2024-07 onward is out of the climb and the lockbox entirely
    assert list(MO.draw_lockbox(MO.SEED)) == lb                    # fixed seed, reproducible
    assert lb == ["2019-05", "2019-12", "2020-03", "2020-07", "2021-02", "2021-10", "2022-09", "2022-10", "2023-01", "2023-04", "2024-01", "2024-06"]


def test_dev_months_are_all_minus_lockbox():
    assert len(MO.DEV) == 54 and not set(MO.DEV) & set(MO.LOCKBOX)
    assert set(MO.DEV) | set(MO.LOCKBOX) == set(MO.ALL_MONTHS) and MO.WARMUP not in MO.ALL_MONTHS


def test_reference_month_skips_lockbox_months():
    ex = set(MO.LOCKBOX)
    assert MO.ref_month("2019-06", ex) == "2019-04"                # 2019-05 is lockbox
    assert MO.ref_month("2022-11", ex) == "2022-08"                # 2022-10 and 2022-09 are lockbox
    assert MO.ref_month("2019-06", set()) == "2019-05"
    assert MO.ref_month("2019-01", ex) == "2018-12"


# ------------------------------------------------------------------ a lockbox month is never predicted or scored in the dev phase
def test_predict_set_refuses_lockbox_in_dev_and_dev_in_look():
    class Dummy:
        pass
    with pytest.raises(ValueError):
        H.predict_set(Dummy(), "rich", "logreg", 730, 4, ("2019-04", "2019-05"), look=False)
    with pytest.raises(ValueError):
        H.predict_set(Dummy(), "rich", "logreg", 730, 4, ("2019-04",), look=True)


@needs_dev
def test_dev_store_cannot_be_built_as_look_and_holds_no_lockbox_prediction():
    inp = H.Inputs()
    with pytest.raises(AssertionError):
        H.Store(inp, "dev", look=True)
    lb = np.concatenate([inp.mon_idx[m] for m in MO.LOCKBOX])
    for h in H.HS:
        assert np.isnan(np.load(H.pred_path("dev", ("rich", "logreg", 730, h)))[lb]).all()
        for exitr in H.EXIT_ORDER:
            assert np.isnan(H.load_paths("dev", h)[exitr]["gross"][lb]).all()      # no lockbox path was simulated either


@needs_dev
def test_every_dev_trade_has_entry_and_exit_in_dev_months():
    inp = H.Inputs(); st = H.Store(inp); ev = H.Evaluator(st)
    lbm = set(MO.LOCKBOX)
    for cfg in (H.BASELINE, dict(H.BASELINE, exit="tp2", H=1), dict(H.BASELINE, side="agree", exit="tp1.5", q=0.2)):
        seq = ev.sequence(cfg)
        assert len(seq["idx"]) > 100
        mon = lambda ns: pd.DatetimeIndex(np.asarray(ns, np.int64).astype("datetime64[ns]")).to_period("M").astype(str)
        assert not set(mon(seq["entry"])) & lbm and not set(mon(seq["exit"] - 1)) & lbm
        assert not set(inp.mon[seq["idx"]]) & lbm and (seq["mpos"] >= 0).all()
        # the last trades of a dev month before a lockbox month are dropped, not held into it
        assert (seq["exit"] <= pd.Timestamp("2024-07-01").value).all()


# ------------------------------------------------------------------ execution: next open, non-overlap
@needs_dev
def test_baseline_enters_at_next_open_and_time_exit_equals_the_frozen_return():
    inp = H.Inputs(); st = H.Store(inp); ev = H.Evaluator(st)
    seq = ev.sequence(H.BASELINE); idx = seq["idx"]
    assert (seq["entry"] == inp.t[idx] + H.HOUR).all()
    frozen = inp.r[4][idx] * seq["side"]                            # cl[anch+3]/op[anch]-1: next-open entry, close of the 4th hour
    assert np.abs(seq["gross"] - frozen).max() < 1e-9
    assert (seq["exit"] == inp.t[idx] + 5 * H.HOUR).all()


@needs_dev
def test_sequences_never_overlap():
    inp = H.Inputs(); st = H.Store(inp); ev = H.Evaluator(st)
    for cfg in (H.BASELINE, dict(H.BASELINE, exit="tp1.5"), dict(H.BASELINE, H=1, q=0.2, exit="tp2")):
        seq = ev.sequence(cfg)
        assert (seq["entry"][1:] >= seq["exit"][:-1]).all() and (np.diff(seq["entry"]) > 0).all()
        assert (seq["exit"] > seq["entry"]).all()


def test_scan_skip_while_open_and_equal_taken():
    e = np.array([0, 1, 5, 6, 10], np.int64); x = np.array([5, 3, 6, 9, 11], np.int64)
    assert H.scan(e, x).tolist() == [0, 2, 3, 4]        # 1 skipped (open until 5); 5 == previous exit is taken; 6 == exit 6 taken


# ------------------------------------------------------------------ information at entry only
def test_weights_use_closed_trades_only():
    net = np.array([-0.03, -0.03, 0.01, 0.02, -0.01, 0.0])
    w = H.weights(net, True); w2 = H.weights(np.concatenate([net[:4], [9.9, -9.9]]), True)
    assert (w[:5] == w2[:5]).all() and w[0] == 1.0 and w[2] == H.BRAKE_F
    assert (H.weights(net, False) == 1).all()


@needs_dev
def test_threshold_is_the_reference_month_quantile_and_ignores_the_future():
    inp = H.Inputs(); st = H.Store(inp); key = ("rich", "logreg", 730, 4)
    p = st.preds(key); idx, _ = st.signals(key, 0.10)
    m = "2021-08"; ref = MO.ref_month(m, st.excluded); assert ref == "2021-07"
    conf = np.abs(p - 0.5); thr = np.quantile(conf[inp.mon_idx[ref]], 0.9)
    in_m = idx[inp.mon[idx] == m]
    want = inp.mon_idx[m][conf[inp.mon_idx[m]] >= thr]
    assert np.array_equal(in_m, want)
    # changing the predictions of every later month leaves month m's signals unchanged
    st2 = H.Store(inp); p2 = p.copy(); later = np.concatenate([inp.mon_idx[x] for x in MO.DEV if x > m]); p2[later] = 0.5 + 0.4 * np.random.default_rng(0).random(len(later))
    st2._pred[key] = p2; idx2, _ = st2.signals(key, 0.10)
    assert np.array_equal(idx2[inp.mon[idx2] == m], in_m)
    # and the month after a lockbox month takes the nearest earlier non-lockbox month (2019-06 -> 2019-04), never the lockbox month 2019-05
    assert MO.ref_month("2019-06", st.excluded) == "2019-04"


@needs_dev
def test_the_pipeline_reproduces_the_frozen_h62_predictions():
    inp = H.Inputs(); p = np.load(H.pred_path("dev", ("rich", "logreg", 730, 4)))
    got = list(F_month(inp, "2021-08"))[0]
    idx = got[1]
    assert np.allclose(p[idx], got[2], atol=1e-6)


def F_month(inp, m):
    return H.F.month_predictions(inp.X["rich"], inp.t, inp.r[4], "logreg", 730, 4, m, m)


# ------------------------------------------------------------------ the climb rule
class FakeEv:
    def __init__(self, table):
        self.table = table
    def eval(self, cfg):
        return self.table(cfg)


def _mk(cal, mb):
    return {"calmar": cal, "cum": cal, "dd": 1.0, "n": 600, "mbps": np.asarray(mb, float)}


def test_climb_accepts_only_when_calmar_up_and_paired_mean_positive():
    base_mb = np.array([1.0, 1.0, 1.0, 1.0])

    def table(cfg):
        if cfg["size"] == "brake":                                  # higher Calmar, but a negative paired monthly mean: refused
            return _mk(5.0, base_mb - 2.0)
        if cfg["exit"] == "tp2" and cfg["feat"] == "rich":          # higher Calmar and a positive paired mean: accepted
            return _mk(3.0, base_mb + 1.0)
        return _mk(1.0, base_mb)
    inc, r, traj = H.climb(FakeEv(table), dict(H.BASELINE), np.random.default_rng(0))
    assert inc["size"] == "fixed" and inc["exit"] == "tp2" and r["calmar"] == 3.0
    assert [t["calmar"] for t in traj] == [1.0, 3.0] and traj[1]["lever"] == "exit"


def test_climb_stops_at_a_local_optimum_and_takes_the_best_candidate_of_a_lever():
    mb = np.ones(3)

    def table(cfg):
        return _mk({"time": 1.0, "tp1.5": 2.0, "tp2": 4.0}[cfg["exit"]], mb + {"time": 0, "tp1.5": 1, "tp2": 1}[cfg["exit"]])
    inc, r, traj = H.climb(FakeEv(table), dict(H.BASELINE), np.random.default_rng(1))
    assert inc["exit"] == "tp2" and len(traj) == 2


def test_space_size_and_baseline_in_space():
    assert H.N_CONFIGS == 1944 and len(H.all_configs()) == 1944
    assert all(H.BASELINE[k] in v for k, v in H.LEVERS) and len({H.ckey(c) for c in H.all_configs()}) == 1944


# ------------------------------------------------------------------ the null world
@needs_dev
def test_null_flip_takes_the_opposite_side_path_and_is_deterministic():
    inp = H.Inputs(); st = H.Store(inp)
    ev = H.Evaluator(st); evf = H.Evaluator(st, flips=-np.ones(len(st.months), int))
    a = ev.sequence(H.BASELINE); b = evf.sequence(H.BASELINE)
    assert np.array_equal(a["idx"][:1], b["idx"][:1]) and (b["side"][:5] == -a["side"][:5]).all()
    # time exit: the opposite side's return is exactly minus the model side's on the same signal
    common = np.intersect1d(a["idx"], b["idx"])
    ga = dict(zip(a["idx"], a["gross"])); gb = dict(zip(b["idx"], b["gross"]))
    assert all(abs(ga[i] + gb[i]) < 1e-12 for i in common[:200])


def test_data_end_is_before_the_spent_period():
    inp = H.Inputs(); assert pd.Timestamp(inp.t.max()) < pd.Timestamp("2024-07-01") and H.DATA_END == "2024-07-01"
