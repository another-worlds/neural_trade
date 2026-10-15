import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import o11_vol_lib as L
from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception: pass
t0 = time.time(); F = L.load_cut(); print("cut", len(F["ta"]), F["ta"][0], F["ta"][-1], "oka frac", F["oka"].mean(), time.time() - t0, flush=True)
G = L.load_fwd_frame(); import numpy as np
print("fwd", len(G["ta"]), G["ta"][0], G["ta"][-1], "oka frac", G["oka"].mean(), "G finite", np.isfinite(G["G"]).all(), time.time() - t0, flush=True)
