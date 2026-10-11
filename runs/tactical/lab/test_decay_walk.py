"""The walk-forward never trains on the scored month or later (seconds)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decay_lib as D


def test_train_labels_end_before_month_and_no_test_leak():
    for bar in D.BAR_MIN:
        t = pd.date_range("2020-01-01", "2021-01-01", freq=f"{D.BAR_MIN[bar]}min")
        for H in (1, 15, 30):
            Htd = pd.Timedelta(minutes=H * D.BAR_MIN[bar])
            for m in pd.period_range("2020-04", "2020-12", freq="M"):
                m0, m1 = m.start_time, (m + 1).start_time
                tr, te = D.split_masks(t, m0, m1, Htd, 60)
                assert tr.any() and te.any()
                assert (t[tr] + Htd).max() <= m0              # the last training label ends by the month start
                assert t[tr].max() < m0 and t[te].min() >= m0 and t[te].max() < m1
                assert not (tr & te).any()
                assert t[tr].min() >= m0 - pd.Timedelta(days=60)


def test_corrupting_the_future_does_not_change_a_scored_month():
    rng = np.random.default_rng(0)
    n = 6000; t = pd.date_range("2020-01-01", periods=n, freq="15min")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 3e-3, n)))
    anch = np.arange(60, n - 1); F = rng.normal(size=(len(anch), 13)).astype(np.float32)
    F = F + (rng.normal(size=(len(anch), 1)) * 0)               # features independent of the future by construction
    sig = np.full(len(anch), 3e-3); tt = t[anch - 1]
    a = D.walk_forward("15m", 1, 20, data=(F, sig, anch, tt, close), first="2020-02", last="2020-02")
    F2 = F.copy(); F2[np.asarray(tt >= pd.Timestamp("2020-03-02"))] += 5.0
    c2 = close.copy(); c2[np.asarray(t >= pd.Timestamp("2020-03-02"))] *= 1.5
    b = D.walk_forward("15m", 1, 20, data=(F2, sig, anch, tt, c2), first="2020-02", last="2020-02")
    assert a and b and a[0]["month"] == b[0]["month"] == "2020-02"
    assert abs(a[0]["auc"] - b[0]["auc"]) < 1e-9
