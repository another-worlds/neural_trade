"""Sanity: the monthly tb7 AUC of this study against the stored H51 (close) and H59 (next-open) values."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import lib

def main():
    t0 = time.time(); D = lib.build_dataset("cut"); print("dataset", len(D["t"]), f"{time.time()-t0:.0f}s", flush=True)
    out = {}
    for entry, ref in (("close", "decay.json"), ("nopen", "decay_nextopen.json")):
        W = lib.walk(D, entry); mine = {r["month"]: r["auc"] for r in W["months"]}
        old = {r["month"]: r["auc"] for r in json.load(open(os.path.join(lib.TAC, "lab", ref)))}
        common = sorted(m for m in mine if m in old and m <= "2025-06")
        d = np.array([mine[m] - old[m] for m in common])
        out[entry] = {"n_months": len(common), "max_abs_diff": float(np.abs(d).max()), "mean_diff": float(d.mean()),
                      "worst_month": common[int(np.abs(d).argmax())], "n_over_0.001": int((np.abs(d) > 0.001).sum()),
                      "mean_auc_mine": float(np.mean([mine[m] for m in common])), "mean_auc_stored": float(np.mean([old[m] for m in common])),
                      "months_mine_not_stored": [m for m in mine if m not in old], "months_stored_not_mine": [m for m in old if m not in mine and m <= "2025-06"]}
        print(entry, out[entry], flush=True)
    json.dump(out, open(os.path.join(HERE, "sanity.json"), "w"), indent=1)
    print(f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
