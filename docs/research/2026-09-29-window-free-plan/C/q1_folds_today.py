"""Q1: how today's walk-forward folds overlap (bundled 30-day file), and what the label-overlap and
finite-window invariants say for every pair of blocks, within and across folds.

Imports the repo's make_purged_splits read-only (D:/neural_trade/src). CPU only.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_folds_today.py
Writes q1_folds_today.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, "D:/neural_trade/src")
from neural_trade.data.splits import make_purged_splits  # noqa: E402

OUT = Path(__file__).with_name("q1_folds_today.json")
N_SEQ = 43_421          # sequences of the bundled file at the default config (runs/.../artifacts/meta.json)
LOOKBACK, H = 60, [10, 15, 20]
HMAX = max(H)


def span(b):
    return int(b[0]), int(b[-1])


def main():
    folds = make_purged_splits(N_SEQ, lookback=LOOKBACK, horizon_steps=H, n_folds=5,
                               val_fraction=0.066, cal_fraction=0.066)
    out = {"n_seq": N_SEQ, "folds": [], "within_fold": [], "cross_fold_overlaps": []}
    for k, f in enumerate(folds):
        name = f"fold {f.fold} (index {k - len(folds)})"
        out["folds"].append({"name": name, "gap": f.gap,
                             **{b: span(getattr(f, b)) for b in ("train", "val", "cal", "test")},
                             "sizes": {b: len(getattr(f, b)) for b in ("train", "val", "cal", "test")}})
        blocks = [("train", f.train), ("val", f.val), ("cal", f.cal), ("test", f.test)]
        for (na, a), (nb, b) in itertools.combinations(blocks, 2):
            last_label_a = int(a[-1]) + HMAX - 1        # label increments of anchor i: bars i..i+H-1
            first_label_b = int(b[0])
            out["within_fold"].append({
                "fold": name, "pair": f"{na}->{nb}",
                "label_distance_bars": first_label_b - last_label_a,          # > 0: disjoint
                "label_disjoint_with_embargo_H": first_label_b - last_label_a > HMAX,
                "finite_window_clear(D-005)": first_label_b - LOOKBACK > last_label_a,
            })
    # cross-fold: which blocks of an earlier fold sit inside a later fold's blocks
    for (i, fa), (j, fb) in itertools.combinations(list(enumerate(folds)), 2):
        for na in ("train", "val", "cal", "test"):
            a = getattr(fa, na)
            for nb in ("train", "val", "cal", "test"):
                b = getattr(fb, nb)
                lo, hi = max(a[0], b[0]), min(a[-1], b[-1])
                if lo <= hi:
                    out["cross_fold_overlaps"].append({
                        "earlier": f"fold index {i - len(folds)} {na}", "later": f"fold index {j - len(folds)} {nb}",
                        "overlap_seq": [int(lo), int(hi)], "n": int(hi - lo + 1)})
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for fo in out["folds"]:
        print(fo["name"], {b: fo[b] for b in ("train", "val", "cal", "test")}, fo["sizes"])
    print("\nwithin-fold minimum label distance:",
          min(w["label_distance_bars"] for w in out["within_fold"]),
          "; all label-disjoint with embargo:", all(w["label_disjoint_with_embargo_H"] for w in out["within_fold"]),
          "; all D-005 finite-window clear:", all(w["finite_window_clear(D-005)"] for w in out["within_fold"]))
    print("\ncross-fold overlaps involving an earlier fold's evaluation blocks:")
    for o in out["cross_fold_overlaps"]:
        if not o["earlier"].endswith("train"):
            print(" ", o)


if __name__ == "__main__":
    main()
