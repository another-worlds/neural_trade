"""Seconds-long checks of the forward test: protected span, training end, next-open, previous-month threshold."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdtest_lib as T


def _write(tmp_path, start, days, name):
    ts = pd.date_range(start, periods=days * 1440, freq="min"); c = 100 + np.arange(len(ts)) * 0.01
    p = tmp_path / name; pd.DataFrame({"timestamp": ts, "open": c - .005, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}).to_csv(p, index=False); return str(p)


def test_forward_loader_refuses_protected_bar(tmp_path):
    try:
        T.load_fwd_bars(_write(tmp_path, "2025-09-29", 5, "a.csv")); assert False
    except AssertionError as e:
        assert "protected" in str(e)
    s = T.load_fwd_bars(_write(tmp_path, "2025-10-01", 5, "b.csv")); assert s.index.min() >= pd.Timestamp("2025-10-01")


def test_train_loader_refuses_bar_after_training_end(tmp_path):
    try:
        T.load_train_bars(_write(tmp_path, "2025-07-19", 3, "c.csv")); assert False
    except AssertionError as e:
        assert "training end" in str(e)


def test_real_files_respect_the_spans():
    s = T.load_train_bars(); assert s.index.max() <= T.TRAIN_END
    f = T.load_fwd_bars(); assert f.index.min() >= T.PROT1


def test_training_span_ends_2025_07_20_and_label_closed():
    X, t, r, _ = T.anchors(T.load_train_bars())
    ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); end = T.TRAIN_END + pd.Timedelta(minutes=1)
    ok = (ti + pd.Timedelta(minutes=(T.H + 1) * T.BAR_MIN) <= end)
    assert ti[ok].max() <= pd.Timestamp("2025-07-20 19:00")


def test_next_open_and_first_forward_anchor_after_warmup():
    f = T.load_fwd_bars(); X, t, r, anch = T.anchors(f)
    assert pd.Timestamp(int(t.min())) >= pd.Timestamp("2025-10-03")        # 60 hourly bars of the new file are needed
    A = f.values; i = 7; assert np.isclose(r[i], A[anch[i] + T.H - 1, 3] / A[anch[i], 0] - 1)


def test_threshold_uses_only_previous_month_and_signals_next_open():
    h = 3600 * 10 ** 9; t = np.arange(10, dtype=np.int64) * h
    thr = {"B": T.month_threshold(np.array([.1, .2, .3, .4]))}
    assert np.isclose(thr["B"], np.quantile([.1, .2, .3, .4], .9))
    ix = np.arange(5, 10); p = np.array([.9, .5, .5, .2, .5])
    ent, ex, side, idx, mon = T.signals({"B": (ix, p), "A": (np.arange(5), np.full(5, .99))}, t, thr)
    assert list(mon) == ["B"] and ent[0] == t[5] + h and side[0] == 1
