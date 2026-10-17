"""M6 FAISS retrieval smoke: 1,000 windows x 20 descriptors from the cut-file tail (5-min anchors), exact L2 index, query k=50 neighbours.
Run with D:/nt/nt_mm_env/python.exe (CPU only)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, faiss
from common_data import bars_5m

def descriptors(r, n_desc=20):
    lr = r.logret.values; rv = r.logrv.values; X = []
    for i in range(24, len(r)):
        w = lr[i - 24:i]; X.append(np.r_[w[-20:], rv[i - 1]][:n_desc])
    return np.asarray(X, "float32")

if __name__ == "__main__":
    t0 = time.time(); r = bars_5m(1100); X = descriptors(r)[:1000]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    idx = faiss.IndexFlatL2(X.shape[1]); idx.add(X)
    D, I = idx.search(X[-1:], 50)
    print("index", idx.ntotal, "dim", X.shape[1], "nn idx[:5]", I[0][:5], "dist[:3]", D[0][:3].round(3), "secs", round(time.time() - t0, 2), "faiss", faiss.__version__)
