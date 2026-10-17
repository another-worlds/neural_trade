"""M3 step 1 (mm_env): the Kronos inputs. For every OK anchor ta (hourly UTC grid) the last 400 complete five-minute OHLCV bars strictly before ta
(harness `bars_before`, nothing else is read), as raw arrays for the legacy env (no parquet there).
Output D:/nt/nt_t14_cache/m3/inputs_{source}.npz: ta_ns [n], x [n,400,5] (open high low close volume, float64), ts_ns [n,400] (bar open time, UTC)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import anchors as AN

CTX = 400
OUT = os.path.join(AN.CACHE_DIR, "m3")


def anchor_input(source, ta, ctx=CTX):
    b = AN.bars_before(source, ta, ctx, "5min")
    return b


def build(source):
    anc = AN.load_anchors(source); ta = anc.index[anc["ok"].values]
    X = np.full((len(ta), CTX, 5), np.nan); T = np.zeros((len(ta), CTX), dtype=np.int64); good = np.zeros(len(ta), bool)
    for i, t in enumerate(ta):
        b = anchor_input(source, t)
        if len(b) == CTX:
            X[i] = b[["open", "high", "low", "close", "volume"]].to_numpy(float); T[i] = b.index.values.astype("datetime64[ns]").astype(np.int64); good[i] = True
            assert b.index[-1] < t and b.index[-1] + pd.Timedelta(minutes=5) <= t
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"inputs_{source}.npz"), ta_ns=ta.values.astype("datetime64[ns]").astype(np.int64)[good], x=X[good], ts_ns=T[good])
    print(source, "ok anchors", len(ta), "with 400 bars", int(good.sum()))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["cut", "fwd"]:
        build(s)
