"""geom2 step 6: per-bar geometry features for the whole minute file (geomlib.geom_rich on the 60-bar window ending at each bar; the first
59 bars are 0), float16 [T, 164], written to cache/series_geom.npy. Causal: a bar's row reads only bars up to and including it.
usage: python build_geom.py [--check]   (--check: recompute 200 random rows and compare)"""
import os, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "geom2"))
import data as dm  # noqa: E402
import geomlib as G  # noqa: E402

OUT = os.path.join(dm.CACHE, "series_geom.npy"); WIN = dm.WIN; CHUNK = 20000
A, tod, dow = dm.load_series()
if "--check" in sys.argv:
    g = np.load(OUT, mmap_mode="r"); idx = np.random.default_rng(0).integers(WIN, len(A), 200)
    W = np.stack([A[i - WIN + 1:i + 1] for i in idx]).astype(np.float32); ref = G.geom_rich(W)
    print("max abs diff", float(np.abs(np.asarray(g[idx], np.float32) - ref).max()), "rows", len(g), "cols", g.shape[1])
    sys.exit()
os.makedirs(dm.CACHE, exist_ok=True)
n = len(A); F = G.geom_rich(np.stack([A[:WIN]] * 2).astype(np.float32)).shape[1]
out = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16, shape=(n, F)); out[:] = 0
for a in range(WIN - 1, n, CHUNK):
    b = min(a + CHUNK, n); W = sliding_window_view(A[a - WIN + 1:b], WIN, axis=0).transpose(0, 2, 1)
    f = G.geom_rich(np.ascontiguousarray(W).astype(np.float32))
    out[a:b] = np.clip(f, -6e4, 6e4).astype(np.float16)
    if (a // CHUNK) % 40 == 0:
        print(a, "/", n, flush=True)
out.flush(); print("written", OUT, out.shape)
