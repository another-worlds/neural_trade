"""T06: the per-bar series cache (vec / seq / close) built from the CUT file only (no bar after 2025-07-20 23:59), with the frozen
newnet/data.py feature code, into D:/nt/nt_t06_cache. usage: python build_cache.py"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "common")); sys.path.insert(0, os.path.join(TAC, "newnet")); sys.path.insert(0, os.path.join(TAC, "lab"))
import numpy as np
import cutoff as C
import lab
import data as dm

OUT = os.environ.get("SERIES_CACHE", "D:/nt/nt_t06_cache")


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    d = C.read_csv(usecols=["timestamp", "open", "high", "low", "close", "volume"])     # asserts the cutoff
    t = __import__("pandas").to_datetime(d["timestamp"])
    A = d[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
    tod = (t.dt.hour * 60 + t.dt.minute).to_numpy(np.float64); dow = t.dt.dayofweek.to_numpy(np.float64)
    print("bars", len(A), "last", t.iloc[-1], flush=True)
    vec, seq = dm.bar_arrays(A, tod, dow, lab, log=lambda s: print(s, flush=True))
    np.save(f"{OUT}/series_vec.npy", vec); np.save(f"{OUT}/series_seq.npy", seq); np.save(f"{OUT}/series_close.npy", A[:, 3])
    np.save(f"{OUT}/series_ohlcv.npy", A)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
