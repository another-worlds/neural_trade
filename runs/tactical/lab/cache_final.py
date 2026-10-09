"""newnet2, criterion 3: the held-out FINAL slices' val blocks, built exactly as cache.py --final does (same Config override,
same excerpt CSV, same DataProcessor fold), written to THIS worktree (lab/cache_final/<slice>.npz: Wva, rva, deadband; the training
windows are not needed by the new network, which builds its training span from the full file).
usage: python cache_final.py"""
import os, sys, time
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import neural_trade  # noqa: F401
from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.experiments import screen as S

sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical\hc4")
import make_hc4 as H   # chdir()s into D:/nt/nt_tactical on import (read-only use); back to the worktree right after
os.chdir(WT)
OUT = os.path.join(HERE, "cache_final"); os.makedirs(OUT, exist_ok=True)
for de in H.FINAL:
    p = f"{OUT}/{de[:13].replace(':', '')}.npz"
    if os.path.exists(p):
        continue
    t = time.time()
    cfg = Config.from_yaml("configs/default.yaml").override(**dict(H.WEEK), DATA_END=de, SEED=0)
    cache = {}; S._load_cached(cfg, cache)
    X_seq, y_seq, lc_seq, ext, X_model = S._windowed_cached(cfg, cache)
    dp = DataProcessor(cfg); dp.prepare_datasets_from_windows(X_seq, y_seq, lc_seq, ext, X_model=X_model)
    fo = dp.fold
    ret = lambda idx: (y_seq[idx] / lc_seq[idx].reshape(-1, 1)).astype(np.float32)
    np.savez(p, Wva=X_model[fo.val].astype(np.float32), rva=ret(fo.val), deadband=float(cfg.DIR_DEADBAND_BPS) / 1e4, data_end=de)
    print(f"{de}: val {len(fo.val)} in {time.time() - t:.1f}s", flush=True)
