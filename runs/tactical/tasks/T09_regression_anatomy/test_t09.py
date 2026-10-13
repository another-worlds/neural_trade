"""T09 tests (seconds): the data cutoff, the never-read span, walk-forward causality and the next-open labels.
usage: python -m pytest -q runs/tactical/tasks/T09_regression_anatomy/test_t09.py -p no:cacheprovider"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pytest

import guard
import lib

DATA_OK = os.path.exists(guard.CUT_CSV) and os.path.exists(guard.FWD_CSV)


# ---------------------------------------------------------------- the cutoff and the never-read span
def _tail_head(path, n=3):
    with open(path, "rb") as f:
        head = f.readlines()[:n]
        f.seek(0, 2); size = f.tell(); f.seek(max(size - 4000, 0)); tail = f.read().decode().strip().splitlines()[-n:]
    return [x.decode().strip() for x in head], tail


@pytest.mark.skipif(not DATA_OK, reason="data files not on this machine")
def test_cut_file_ends_at_the_cutoff():
    _, tail = _tail_head(guard.CUT_CSV)
    assert tail[-1][:16] == "2025-07-20 23:59", tail[-1]
    assert pd.Timestamp(tail[-1][:19]) <= guard.CUT_END


@pytest.mark.skipif(not DATA_OK, reason="data files not on this machine")
def test_forward_file_starts_after_the_never_read_span():
    head, _ = _tail_head(guard.FWD_CSV)
    assert head[1][:10] == "2025-10-01"
    assert pd.Timestamp(head[1][:19]) > guard.NEVER_TO


def test_guard_rejects_a_bar_in_the_never_read_span():
    for ts in ("2025-07-21 00:00", "2025-08-15 12:00", "2025-09-30 23:59"):
        with pytest.raises(AssertionError):
            guard.check_index(pd.DatetimeIndex([pd.Timestamp("2025-07-20 23:59"), pd.Timestamp(ts)]))
    guard.check_index(pd.DatetimeIndex(["2025-07-20 23:59", "2025-10-01 00:00"]))       # the two edges are fine


def test_guard_allows_only_the_two_files_and_never_the_original():
    assert guard.ALLOWED == {os.path.normpath(guard.CUT_CSV), os.path.normpath(guard.FWD_CSV)}
    assert "neural_trade" not in guard.CUT_CSV and "neural_trade" not in guard.FWD_CSV
    for p in glob.glob(os.path.join(HERE, "*.py")):
        if os.path.basename(p) in ("test_t09.py", "guard.py"):
            continue
        src = open(p, encoding="utf-8").read()
        assert "Bitcoin_BTCUSDT.csv" not in src and "neural_trade/Bitcoin" not in src, p     # only the cut copy is ever named


@pytest.mark.skipif(not DATA_OK, reason="data files not on this machine")
def test_loaded_bars_hold_no_bar_of_the_never_read_span():
    for which, lo, hi in (("cut", None, guard.CUT_END), ("fwd", guard.FWD_FIRST, guard.FWD_LAST)):
        s = guard.load_ohlcv(which)
        assert not ((s.index >= guard.NEVER_FROM) & (s.index <= guard.NEVER_TO)).any()
        assert s.index.max() <= hi
        if lo is not None:
            assert s.index.min() >= lo


@pytest.mark.skipif(not os.path.exists(f"{lib.CACHE}/ds_cut.npz"), reason="no dataset cache")
def test_built_datasets_have_no_anchor_in_the_never_read_span():
    for which in ("cut", "fwd"):
        if not os.path.exists(f"{lib.CACHE}/ds_{which}.npz"):
            continue
        D = lib.build_dataset(which); t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]"))
        assert not ((t >= guard.NEVER_FROM) & (t <= guard.NEVER_TO)).any()



# ---------------------------------------------------------------- walk-forward causality
def _synthetic(n_days=130, seed=0):
    """A small minute-anchor dataset (anchors every 10 minutes) with a weak real signal, in the lib.walk input format."""
    rng = np.random.default_rng(seed); n = n_days * 144
    t = (pd.Timestamp("2020-01-01 00:59") + pd.to_timedelta(np.arange(n) * 10, unit="min")).values.astype("datetime64[ns]").astype(np.int64)
    F = rng.normal(size=(n, 13)); r = 4e-4 * (-F[:, 3] + 0.5 * rng.normal(size=n)); R = np.full((n, len(lib.HS)), np.nan); R[:, lib.HS.index(15)] = r
    return {"F": F, "t": t, "r_close": R, "r_nopen": R.copy(), "r_mid": R.copy()}


def test_training_labels_are_finished_before_the_month_starts():
    D = _synthetic(); t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]"))
    for m in lib.month_range(D["t"], first="2020-03", last="2020-04"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = lib.split(D["t"], m0, m1, 60)
        # the last training anchor's label ends H = 15 minutes (close entry) / H + 1 (next-open entry) after its last bar, plus the bar's own length
        assert (t[tr].max() + pd.Timedelta(minutes=lib.H0 + 2)) <= m0
        assert t[tr].max() < m0 and t[te].min() >= m0 and (t[tr].min() >= m0 - pd.Timedelta(days=60))
        assert not (tr & te).any()


def test_a_months_fit_never_sees_its_own_month_or_later_data():
    """Replace every anchor from the month's start onward by garbage (features and returns): the month's coefficients, scaler and training
    threshold stay bit-identical, because the fit uses only earlier anchors."""
    D = _synthetic(); W0 = lib.walk(D, "close", first="2020-04", last="2020-04", min_train=500, min_test=300)
    assert len(W0["months"]) == 1
    D2 = {k: np.array(v, copy=True) for k, v in D.items()}; t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); late = np.asarray(t >= pd.Timestamp("2020-04-01"))
    rng = np.random.default_rng(1); D2["F"][late] = rng.normal(size=(late.sum(), 13)) * 50
    for e in ("close", "nopen", "mid"):
        D2["r_" + e][late] = rng.normal(size=D2["r_" + e][late].shape) * 1e-2
    W1 = lib.walk(D2, "close", first="2020-04", last="2020-04", min_train=500, min_test=300)
    a, b = W0["months"][0], W1["months"][0]
    assert a["coef"] == b["coef"] and a["intercept"] == b["intercept"] and a["thr"] == b["thr"] and a["n_train"] == b["n_train"]
    # and the converse: changing the TRAINING span changes the fit (the test above is not vacuous)
    D3 = {k: np.array(v, copy=True) for k, v in D.items()}; early = np.asarray(t < pd.Timestamp("2020-03-15")) & np.asarray(t >= pd.Timestamp("2020-02-15"))
    D3["F"][early, 3] = -D3["F"][early, 3]
    W2 = lib.walk(D3, "close", first="2020-04", last="2020-04", min_train=500, min_test=300)
    assert W2["months"][0]["coef"] != a["coef"]


def test_predictions_exist_only_for_the_test_month():
    D = _synthetic(); W = lib.walk(D, "close", first="2020-04", last="2020-04", min_train=500, min_test=300)
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); has = np.isfinite(W["p"])
    assert has.sum() > 0 and (t[has] >= pd.Timestamp("2020-04-01")).all() and (t[has] < pd.Timestamp("2020-05-01")).all()


# ---------------------------------------------------------------- labels
def test_next_open_close_and_mid_labels_by_hand():
    #            bar:    0     1     2     3     4     5     6
    o = np.array([100., 101., 102., 103., 104., 105., 106.])
    h = np.array([101., 103., 104., 105., 106., 107., 108.])
    l = np.array([99., 100., 101., 102., 103., 104., 105.])
    c = np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5])
    a = np.array([2, 3])            # window last bar = 1 and 2; the next bar is 2 and 3
    R = lib.entry_returns(o, h, l, c, a)
    j1, j5 = lib.HS.index(1), lib.HS.index(5)
    assert R["close"][0, j1] == pytest.approx(c[1 + 1] / c[1] - 1)          # enter at the last bar's close, exit one bar later
    assert R["nopen"][0, j1] == pytest.approx(c[2 + 1] / o[2] - 1)          # enter at the NEXT bar's open (bar 2), exit at the close of bar a+h
    assert R["mid"][0, j1] == pytest.approx(c[2 + 1] / ((h[2] + l[2]) / 2) - 1)
    assert R["nopen"][1, j1] == pytest.approx(c[4] / o[3] - 1)
    # the next-open entry never uses the signal bar's close: changing c[a-1] leaves it unchanged, and changing o[a] leaves the close entry unchanged
    c2 = c.copy(); c2[1] += 5; R2 = lib.entry_returns(o, h, l, c2, a)
    assert R2["nopen"][0, j1] == R["nopen"][0, j1] and R2["close"][0, j1] != R["close"][0, j1]
    o2 = o.copy(); o2[2] += 5; R3 = lib.entry_returns(o2, h, l, c, a)
    assert R3["close"][0, j1] == R["close"][0, j1] and R3["nopen"][0, j1] != R["nopen"][0, j1]
    assert np.isfinite(R["close"][0, j5]) and np.isnan(R["nopen"][0, j5]) and np.isnan(R["close"][1, j5])      # exit beyond the series


def test_the_three_labels_share_the_exit_bar_and_differ_only_in_entry():
    o, h, l, c = (np.array(x) for x in ([10., 11., 12., 13.], [11., 12., 13., 14.], [9., 10., 11., 12.], [10.5, 11.5, 12.5, 13.5]))
    R = lib.entry_returns(o, h, l, c, np.array([2]))
    j = lib.HS.index(1)
    assert (1 + R["nopen"][0, j]) * o[2] == pytest.approx(c[3]) and (1 + R["mid"][0, j]) * ((h[2] + l[2]) / 2) == pytest.approx(c[3])


def test_deadband_scales_with_the_horizon():
    assert lib.deadband(15) == pytest.approx(5e-4) and lib.deadband(60) == pytest.approx(1e-3) and lib.deadband(1) < lib.deadband(5)


def test_corwin_schultz_is_zero_for_flat_bars_and_positive_for_a_wide_pair():
    flat = lib.cs_series(np.full(5, 100.0), np.full(5, 100.0)); assert (flat == 0).all()
    h = np.array([100.0, 101.0, 100.6]); l = np.array([99.0, 99.8, 100.1])
    assert lib.cs_series(h, l)[1] >= 0


# ---------------------------------------------------------------- the sanity record
@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "sanity.json")), reason="sanity.json not produced yet")
def test_sanity_reproduces_h51_and_h59_within_0_001():
    s = json.load(open(os.path.join(HERE, "sanity.json")))
    for e in ("close", "nopen"):
        assert s[e]["max_abs_diff"] < 1e-3 and s[e]["n_months"] >= 90 and s[e]["n_over_0.001"] == 0


# ---------------------------------------------------------------- seams and the UTC hour (coordinator note: the cut file is shifted, Bitstamp)
def test_seam_mask_flags_dst_days_and_the_splice_only():
    ts = pd.DatetimeIndex(["2018-03-11 08:00", "2019-11-03 06:00", "2024-09-13 08:30", "2019-06-17 12:00", "2021-05-05 03:00", "2018-03-13 12:00"])
    m = lib.seam_mask(ts.values.astype("datetime64[ns]").astype(np.int64))
    assert m.tolist() == [True, True, True, False, False, False]
    assert not lib.seam_mask(ts.values.astype("datetime64[ns]").astype(np.int64), "fwd").any()          # the forward file (UTC, Binance) has none


def test_utc_hour_converts_the_shifted_file_and_leaves_the_rest():
    f = lambda s, w="cut": int(lib.utc_hour(pd.DatetimeIndex([s]).values.astype("datetime64[ns]").astype(np.int64), w)[0])
    assert f("2019-07-01 12:00") == 8            # US daylight saving: file = UTC + 4
    assert f("2019-01-15 12:00") == 7            # otherwise file = UTC + 5
    assert f("2019-01-15 03:00") == 22           # wraps across midnight
    assert f("2025-06-01 12:00") == 12           # after the 2024-09-13 splice the file is UTC
    assert f("2026-02-01 05:00", "fwd") == 5     # the forward file is UTC


def test_analysis_excludes_seam_anchors_from_the_impulse_response():
    import analysis as A
    n = 3000; t = (pd.Timestamp("2018-03-11 00:00") + pd.to_timedelta(np.arange(n) * 10, unit="min")).values.astype("datetime64[ns]").astype(np.int64)
    rng = np.random.default_rng(0); R = {e: rng.normal(size=(n, len(lib.HS))) * 1e-3 for e in lib.ENTRIES}
    D = {"t": t, "lastret": rng.normal(size=(n, len(lib.KS))) * 1e-3, "seam": lib.seam_mask(t), **{"r_" + e: R[e] for e in lib.ENTRIES}}
    assert D["seam"].any()
    a = A.impulse(D, first_month="2018-03", entries=("close",), exclude_seam=True); b = A.impulse(D, first_month="2018-03", entries=("close",), exclude_seam=False)
    assert a["close"]["n"].sum() < b["close"]["n"].sum()
