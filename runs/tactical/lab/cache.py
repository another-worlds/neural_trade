"""Logistic-regression lab, step 1: cache each slice's train and val windows and targets once (owner, 2026-10-09:
"a big step toward the logistic-regression approach: instant results and speed").
Same data as the network's long-block runs (make_hc4.WEEK: excerpt, MAX_SEQUENCE_COUNT 126,000, fold -2, train -> fit,
val -> score), so lab numbers and network numbers are on identical blocks.
usage: python cache.py [--final]   (--final caches the held-out FINAL slices; only for a winner's one-time check)
out:   runs/tactical/lab/cache/<data_end>.npz  Wtr, Wva [N, 60, 5] float32 OHLCV windows; rtr, rva [N, 3] returns per horizon;
       deadband (fraction). Not committed (heavy)."""
import os, sys, time
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.chdir(r"D:\nt\nt_tactical")
import neural_trade  # noqa: F401
from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.experiments import screen as S

sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical\hc4")
import make_hc4 as H

OUT = "runs/tactical/lab/cache"; os.makedirs(OUT, exist_ok=True)
slices = H.FINAL if "--final" in sys.argv else H.CLIMB
if "--many" in sys.argv:  # 30 slices for resolution (AUDIT.md: judge on 20-40): the 6 CLIMB plus 24 more climb slices, never FINAL
    pool = [s for s in H.climb if s not in H.FINAL and s not in H.CLIMB]
    slices = H.CLIMB + [pool[round(i * (len(pool) - 1) / 23)] for i in range(24)]
    from datetime import datetime as _D  # a slice spans ~87.5 days: drop any within 90 days of a FINAL slice (18 extra remain)
    slices = [s for s in slices if s in H.CLIMB or min(abs((_D.fromisoformat(s) - _D.fromisoformat(f)).days) for f in H.FINAL) >= 90]
for de in slices:
    p = f"{OUT}/{de[:13].replace(':', '')}.npz"
    if os.path.exists(p):
        continue
    t = time.time()
    ov = dict(H.WEEK)
    if de not in H.CLIMB and de not in H.FINAL:  # the excerpt covers only the 12 hill-climb slices: others need the full file
        ov["CSV_PATH"] = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
    cfg = Config.from_yaml("configs/default.yaml").override(**ov, DATA_END=de, SEED=0)
    cache = {}; S._load_cached(cfg, cache)
    X_seq, y_seq, lc_seq, ext, X_model = S._windowed_cached(cfg, cache)
    dp = DataProcessor(cfg); dp.prepare_datasets_from_windows(X_seq, y_seq, lc_seq, ext, X_model=X_model)
    fo = dp.fold
    ret = lambda idx: (y_seq[idx] / lc_seq[idx].reshape(-1, 1)).astype(np.float32)
    np.savez(p, Wtr=X_model[fo.train].astype(np.float32), Wva=X_model[fo.val].astype(np.float32),
             rtr=ret(fo.train), rva=ret(fo.val), deadband=float(cfg.DIR_DEADBAND_BPS) / 1e4, data_end=de)
    print(f"{de}: train {len(fo.train)} val {len(fo.val)} in {time.time() - t:.1f}s", flush=True)
