"""T06: the lab-style training block of a slice rebuilt from the CUT series, so the held-out FINAL slices (whose lab cache holds only the
val block) can be scored by the lab members, and checked against the lab cache on the 24 slices.
Block rule (rebuild/longtrain.py): the training windows are the NT consecutive 1-minute windows that end 81 bars before the val block's
first window (the cache's 80-bar purge gap); NT = 16560 (11.5 days) = the cache's own length.
usage: python blocks.py --final   (writes D:/nt/nt_t06_cache/final_blocks/*.npz)      python blocks.py --verify (the 24 lab slices)"""
import glob, os, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.environ.get("SERIES_CACHE", "D:/nt/nt_t06_cache")
LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
FINAL_CACHE = os.path.join(TAC, "lab", "cache_final")
NT, GAPB, WIN, HZ = 16560, 81, 60, (10, 15, 20)


def windows(A, s0, n):
    return sliding_window_view(A[s0:s0 + n + WIN - 1], WIN, axis=0).transpose(0, 2, 1)


def locate(A, Wva):
    w = Wva[0].astype(np.float64)
    for i in np.where(np.abs(A[:, 3] - w[0, 3]) < 1e-2)[0]:
        if i + WIN <= len(A) and np.allclose(A[i:i + WIN], w, rtol=1e-5, atol=1e-6):
            return int(i)
    raise RuntimeError("val block not found in the series")


def labels(A, starts):
    last = np.asarray(starts) + WIN - 1
    return np.stack([A[last + h, 3] / A[last, 3] - 1 for h in HZ], 1).astype(np.float32)


def build_block(A, Wva, nt=NT):
    """(Wtr, rtr, vs) of the slice whose val windows are Wva [N,60,5], from the OHLCV series A [T,5]."""
    vs = locate(A, Wva); s0 = vs - GAPB - (nt - 1)
    assert s0 >= 0
    return np.ascontiguousarray(windows(A, s0, nt)).astype(np.float32), labels(A, s0 + np.arange(nt)), vs


def main():
    A = np.load(f"{CACHE}/series_ohlcv.npy", mmap_mode="r")
    if "--verify" in sys.argv:
        bad = 0
        for p in sorted(glob.glob(f"{LAB_CACHE}/*.npz")):
            D = np.load(p); Wtr, rtr, vs = build_block(np.asarray(A), D["Wva"], len(D["Wtr"]))
            ok = np.allclose(Wtr, D["Wtr"], rtol=1e-5, atol=1e-6) and np.allclose(rtr, D["rtr"], rtol=1e-4, atol=2e-6)
            bad += not ok; print(os.path.basename(p), "train block equals the lab cache:", ok, "n", len(D["Wtr"]), flush=True)
        print("mismatches:", bad); sys.exit(1 if bad else 0)
    os.makedirs(f"{CACHE}/final_blocks", exist_ok=True); A = np.asarray(A)
    for p in sorted(glob.glob(f"{FINAL_CACHE}/*.npz")):
        D = np.load(p); Wtr, rtr, vs = build_block(A, D["Wva"])
        assert np.allclose(labels(A, vs + np.arange(len(D["Wva"]))), D["rva"], rtol=1e-4, atol=2e-6), "rva differs from the cut series"
        np.savez(f"{CACHE}/final_blocks/{os.path.basename(p)}", Wtr=Wtr, Wva=D["Wva"], rtr=rtr, rva=D["rva"], deadband=D["deadband"],
                 data_end=D["data_end"], vs=vs)
        print(os.path.basename(p), "val", len(D["Wva"]), "vs", vs, flush=True)


if __name__ == "__main__":
    main()
