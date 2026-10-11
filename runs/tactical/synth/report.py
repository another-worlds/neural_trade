"""Report for SPEC.md: tables per arm and target, paired differences with 95% t-intervals, Holm over the 32 pre-registered tests,
the win rule. usage: python report.py > report.txt"""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["tb7", "har", "c3", "C1", "C2a", "C2b", "B", "A"]


def load(name, key):
    p = os.path.join(HERE, name)
    return sorted((json.loads(l) for l in open(p)), key=lambda r: r[key]) if os.path.exists(p) else []


def metric(row, arm, tgt):
    r = row["res"].get(arm, {}).get(tgt)
    if r is None:
        return np.nan
    if tgt == "dir":
        return float(r[0])
    return float(r["spearman"])


def table(rows, tgt):
    out = {}
    for a in ARMS:
        v = np.array([metric(r, a, tgt) for r in rows])
        if np.isfinite(v).any():
            out[a] = v
    return out


def ci(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 1.0
    return m, m - t * se, m + t * se, p, n


def main():
    prot = {"lab": load("results_lab.jsonl", "slice"), "wf": load("results_wf.jsonl", "month")}
    print("SPEC: runs/tactical/synth/SPEC.md. dir = mean AUC of h10/15/20 (lab: close-entry labels; wf: next-open labels); "
          "vol = Spearman with next-hour log RV. Intervals: 95% t over slices (lab) / months (wf; months are weakly dependent, "
          "so a little optimistic).\n")
    tests = []
    for P, rows in prot.items():
        if not rows:
            continue
        print(f"=== {P}: {len(rows)} {'slices' if P == 'lab' else 'months'}; seconds per unit {np.mean([r['sec'] for r in rows]):.0f}")
        for tgt in ("dir", "vol"):
            T = table(rows, tgt)
            print(f"-- {tgt}: mean [95% CI]")
            for a, v in T.items():
                m, lo, hi, _, n = ci(v)
                print(f"   {a:5s} {m:.4f} [{lo:.4f},{hi:.4f}]")
            base = "tb7" if tgt == "dir" else "har"
            for X, c2 in (("A", "C2a"), ("B", "C2b")):
                if X not in T:
                    continue
                for nm, bb in ((base, base), ("C1", "C1"), (c2, c2), ("c3", "c3")):
                    if bb in T:
                        m, lo, hi, p, n = ci(T[X] - T[bb])
                        tests.append({"P": P, "tgt": tgt, "X": X, "vs": nm, "m": m, "lo": lo, "hi": hi, "p": p, "n": n})
        print()
    # Holm over the pre-registered family
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"]); run = 0.0; M = len(tests)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (M - r) * tests[i]["p"])); tests[i]["holm"] = run
    print(f"=== Paired differences X - baseline (family of {M} tests, Holm-adjusted two-sided p)")
    print("   protocol target X   vs     diff     95% CI              p       Holm p")
    for t in sorted(tests, key=lambda t: (t["P"], t["tgt"], t["X"], t["vs"])):
        print(f"   {t['P']:5s} {t['tgt']:5s} {t['X']:2s}  {t['vs']:5s} {t['m']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}]  {t['p']:.4f}  {t['holm']:.4f}")
    print("\n=== Win rule (both protocols, all four differences: CI above 0 and Holm p < 0.05)")
    any_win = False
    for X in ("A", "B"):
        for tgt in ("dir", "vol"):
            sel = [t for t in tests if t["X"] == X and t["tgt"] == tgt]
            ok = len(sel) == 8 and all(t["lo"] > 0 and t["holm"] < 0.05 for t in sel)
            any_win |= ok
            print(f"   {X} {tgt}: {'WIN' if ok else 'no'} ({sum(t['lo'] > 0 and t['holm'] < 0.05 for t in sel)}/{len(sel)} tests pass)")
    print("   -> the ONE look is " + ("TRIGGERED" if any_win else "NOT triggered"))
    # exploratory: learned series alone and geometry on top of them, other contrasts
    print("\n=== Exploratory contrasts (not in the family; raw 95% CI)")
    for P, rows in prot.items():
        for tgt in ("dir", "vol"):
            T = table(rows, tgt)
            for x, y in (("A", "B"), ("C2a", "tb7" if tgt == "dir" else "har"), ("C2b", "tb7" if tgt == "dir" else "har"), ("B", "C1"), ("c3", "tb7" if tgt == "dir" else "har")):
                if x in T and y in T:
                    m, lo, hi, p, n = ci(T[x] - T[y]); print(f"   {P} {tgt} {x} - {y}: {m:+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()


def sensitivity():
    """SPEC protocol 3 (not in the verdict): K = 8 / 16 for A, B, C1 on the lab slices, direction, against tb7 of the main run."""
    main_rows = {r["slice"]: r for r in load("results_lab.jsonl", "slice")}
    print("\n=== Sensitivity: number of synthetic series K (lab slices, direction, mean AUC; diff against tb7 of the main run)")
    for k in (8, 12, 16):
        rows = load("results_lab.jsonl" if k == 12 else f"results_lab_k{k}.jsonl", "slice")
        for a in ("A", "B", "C1"):
            v = np.array([metric(r, a, "dir") for r in rows]); b = np.array([metric(main_rows[r["slice"]], "tb7", "dir") for r in rows])
            if len(v) > 2:
                m, lo, hi, _, n = ci(v); d = ci(v - b)
                print(f"   K={k:2d} {a:3s} n={n} {m:.4f} [{lo:.4f},{hi:.4f}]   vs tb7 {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")


if __name__ == "__main__":
    sensitivity()
