"""Paired comparison of variant B against variant A (default: base) over all climb chunks.
usage: python hc2_compare.py <variantB> [variantA=base]
Unit of inference = slice (per-slice mean of the paired (slice, seed) AUC differences); 95% t-interval over slices;
also the ADOPT rule fixed in make_hc2.py: CI entirely above 0 and mean diff >= +0.01."""
import glob, json, math, statistics as st, sys, collections

T = {n: v for n, v in zip(range(1, 61), [
    12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13, 2.12, 2.11, 2.10,
    2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05, 2.04, 2.04, 2.04, 2.04, 2.03, 2.03, 2.03,
    2.03, 2.02, 2.02, 2.02, 2.02, 2.02, 2.02, 2.01, 2.01, 2.01, 2.01, 2.01, 2.01, 2.01, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00])}


def load(variant, tag="c*"):
    d = {}
    for f in glob.glob(f"runs/tactical/screens/hc2_{variant}_{tag}/results*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); a = r.get("direction_auc") or {}
            if all(a.get(h) and a[h].get("auc") is not None for h in ("h0", "h1", "h2")):
                d[(r["data_end"], r["seed"])] = st.mean(a[h]["auc"] for h in ("h0", "h1", "h2"))
    return d


def per_slice(a, b):
    by = collections.defaultdict(list)
    for k in set(a) & set(b):
        by[k[0]].append(b[k] - a[k])
    return {s: st.mean(v) for s, v in by.items()}


if __name__ == "__main__":
    B = sys.argv[1]; A = sys.argv[2] if len(sys.argv) > 2 else "base"
    tag = "final" if "--final" in sys.argv else "c*"
    a, b = load(A, tag), load(B, tag)
    ps = per_slice(a, b); ms = list(ps.values())
    print(f"A={A} trials {len(a)}  mean {st.mean(a.values()):.4f} | B={B} trials {len(b)}  mean {st.mean(b.values()):.4f}")
    if len(ms) < 2:
        sys.exit("not enough paired slices")
    m, se = st.mean(ms), st.stdev(ms) / math.sqrt(len(ms)); t = T.get(len(ms) - 1, 2.0)
    lo, hi = m - t * se, m + t * se
    print(f"slices {len(ms)}  mean diff {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  SE {se:.4f}  slices up/down {sum(x>0 for x in ms)}/{sum(x<0 for x in ms)}")
    print("VERDICT:", "ADOPT-for-confirmation" if lo > 0 and m >= 0.01 else ("worse" if hi < 0 else "no effect / inconclusive"))
