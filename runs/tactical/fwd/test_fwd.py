"""Seconds-long checks: no bar on or after 2024-07-01 is loaded; next-open entry; non-overlap; purge."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdlib as F


def _csv(tmp_path):
    ts = pd.date_range("2024-06-29 00:00", periods=4 * 1440, freq="min")   # runs to 2024-07-02
    c = 100 + np.arange(len(ts)) * 0.01
    df = pd.DataFrame({"timestamp": ts, "open": c - 0.005, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})
    p = tmp_path / "b.csv"; df.to_csv(p, index=False); return str(p)


def test_no_bar_on_or_after_cutoff(tmp_path):
    s = F.load_bars(_csv(tmp_path))
    assert s.index.max() < pd.Timestamp("2024-07-01") and s.index.max() >= pd.Timestamp("2024-06-30 23:00")
    # and the cached sets used by the grid
    if os.path.exists(f"{F.CACHE}/t.npy"):
        t = np.load(f"{F.CACHE}/t.npy"); assert pd.Timestamp(int(t.max())) < pd.Timestamp("2024-07-01")
        assert len(np.load(f"{F.CACHE}/open.npy")) == len(np.load(f"{F.CACHE}/close.npy"))
        assert pd.Timestamp(int(np.load(f"{F.CACHE}/t.npy").max())) + pd.Timedelta(hours=F.HMAX + 1) <= pd.Timestamp("2024-07-01") + pd.Timedelta(hours=1)


def test_next_open_entry():
    op = np.array([10., 11, 12, 13, 14, 15]); cl = np.array([10.5, 11.5, 12.5, 13.5, 14.5, 15.5]); an = np.array([2, 3])
    assert np.allclose(F.next_open_ret(op, cl, an, 1), [12.5 / 12 - 1, 13.5 / 13 - 1])
    assert np.allclose(F.next_open_ret(op, cl, an, 2), [13.5 / 12 - 1, 14.5 / 13 - 1])


def test_trades_nonoverlapping_and_entry_time():
    h = 3600 * 10 ** 9; t = np.arange(10, dtype=np.int64) * h
    p = np.array([.9, .9, .9, .1, .5, .5, .5, .5, .5, .9]); r = np.full(10, 0.001)
    ent, ex, side, ret, mon = F.trades_from([("2019-01", np.arange(10), p, r)], t, 2, 0.4)
    assert (ent[1:] >= ex[:-1]).all() and ent[0] == t[0] + h and ex[0] == t[0] + 3 * h
    assert side[0] == 1 and ret[0] == 0.001


def test_forward_refuses_spent_months(tmp_path):
    import forward
    try:
        forward.run_forward(_csv(tmp_path), "2025-09", "2025-12"); assert False
    except SystemExit as e:
        assert "spent" in str(e)


def test_forward_runs_on_synthetic(tmp_path):
    import forward
    rng = np.random.default_rng(0); n = 90 * 1440
    ts = pd.date_range("2021-01-01", periods=n, freq="min"); c = 100 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    df = pd.DataFrame({"timestamp": ts, "open": np.r_[c[0], c[:-1]], "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": rng.random(n) + 1})
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    cfg = dict(forward.CFG, T=30)
    orig = (forward.F.__dict__.get("x"),)
    import wf
    wf.MIN_TRAIN, wf.MIN_TEST = 100, 100
    res = forward.run_forward(str(p), "2021-03", "2021-03", min_month="2021-01", cfg=cfg, out_dir=str(tmp_path))
    assert res["n_months"] == 1 and res["n_trades"] > 0 and "PASS" in res
