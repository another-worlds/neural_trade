"""Seconds. The walk-forward never trains on the scored month or later; every feature block is unchanged when later bars change."""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wf  # noqa: E402
import feats  # noqa: E402


def test_training_labels_close_before_the_month_starts():
    for bar in wf.BAR_MIN:
        bm, H = wf.BAR_MIN[bar], wf.HOR[bar]
        t = pd.date_range("2020-01-01", "2021-01-01", freq=f"{bm}min")
        for m in pd.period_range("2020-04", "2020-12", freq="M"):
            m0, m1 = m.start_time, (m + 1).start_time
            for T in (60, 365):
                tr, te = wf.split_masks(t, m0, m1, H, bm, T)
                assert tr.any() and te.any() and not (tr & te).any()
                assert (t[tr] + pd.Timedelta(minutes=(H + 1) * bm)).max() <= m0       # the last label bar has CLOSED by the month start
                assert t[tr].max() < m0 and t[te].min() >= m0 and t[te].max() < m1
                assert t[tr].min() >= m0 - pd.Timedelta(days=T)


def _synth(seed=0, n=9000):
    rng = np.random.default_rng(seed); t = pd.date_range("2020-01-01", periods=n, freq="15min")
    X = rng.normal(size=(n, 6)).astype(np.float32); r = 1e-3 * (0.3 * X[:, 0] + rng.normal(size=n))
    return X, t, r


def test_corrupting_the_scored_months_future_and_the_border_labels_changes_nothing():
    X, t, r = _synth(); H, bm = 3, 15
    for kind in wf.MODELS:
        a = wf.walk_forward(X, t, r, "1m", kind, 20, "2020-02", "2020-02", H=H, bar_min=bm, db=0.0)
        assert len(a) == 1 and a[0]["month"] == "2020-02"
        m0, m1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")
        X2, r2 = X.copy(), r.copy()
        fut = np.asarray(t >= m1); X2[fut] += 5.0; r2[fut] *= -3.0                        # everything after the scored month
        border = np.asarray((t >= m0 - pd.Timedelta(minutes=bm * H)) & (t < m0))           # anchors whose label closes after the month start
        assert border.sum() >= H
        r2[border] = -r2[border] * 7                                                       # their labels reach into the scored month
        b = wf.walk_forward(X2, t, r2, "1m", kind, 20, "2020-02", "2020-02", H=H, bar_min=bm, db=0.0)
        assert abs(a[0]["auc"] - b[0]["auc"]) < 1e-9 and abs(a[0]["top10_bps"] - b[0]["top10_bps"]) < 1e-9


def test_every_feature_block_ignores_later_bars():
    rng = np.random.default_rng(1); n = 3000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 5e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 5e-4, n)))
    A = np.stack([o, h, l, c, rng.uniform(1, 10, n)], 1)
    idx = pd.date_range("2020-01-01", periods=n, freq="1min")
    tod = (idx.hour * 60 + idx.minute).to_numpy(float); dow = idx.dayofweek.to_numpy(float)
    anch = np.arange(100, n - 1, 37); b = 2000
    A2 = A.copy(); A2[b + 1:] = A2[b + 1:] * rng.uniform(0.5, 1.5, (n - b - 1, 5))
    F1 = feats.blocks_at(A, tod, dow, anch); F2 = feats.blocks_at(A2, tod, dow, anch)
    ok = anch - 1 <= b
    assert ok.sum() > 40 and (~ok).sum() > 10
    assert set(F1) == set(feats.BLOCKS)
    for k in feats.BLOCKS:
        assert F1[k].shape[1] > 0
        assert np.array_equal(F1[k][ok], F2[k][ok]), k
        assert not np.array_equal(F1[k][~ok], F2[k][~ok]), k         # the test can see a change
