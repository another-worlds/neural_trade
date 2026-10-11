"""usage: python build_cache.py real [workers]   -> D:/nt/nt_shape_cache/real.npz (anchors from 2017 to the 2025-07-20 cutoff)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shapelib as S

if __name__ == "__main__":
    t0 = time.time(); w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    df = S.load_bars()
    print(f"bars {len(df)} {df.index.min()} .. {df.index.max()} (cutoff {S.CUTOFF}) loaded in {time.time() - t0:.0f}s", flush=True)
    assert df.index.max() <= S.CUTOFF
    ds = S.build_dataset(df, workers=w)
    S.save_dataset(ds, "real")
    print(f"anchors {len(ds['t'])}; done in {time.time() - t0:.0f}s", flush=True)
