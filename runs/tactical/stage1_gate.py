"""Stage-1 gate (owner: 120 trials per variant is too many). After chunk c0 only (8 slices x 3 seeds = 24 trials):
promote a variant to the full 40 slices unless it is already clearly not worth it. Rule fixed before any result:
PROMOTE if the upper end of the 95% t-interval over the 8 per-slice mean differences (vs base c0) exceeds +0.01
(a meaningful gain is not yet excluded); DROP otherwise. The final verdict still uses all 40 slices of promoted
variants (hc2_compare.py), so stage 1 only removes losers.
usage: python stage1_gate.py <variant>...   -> prints PROMOTE/DROP per variant"""
import math, statistics as st, sys
sys.path.insert(0, "runs/tactical")
from hc2_compare import T, load, per_slice
a = load("base", "c0")
for v in sys.argv[1:]:
    b = load(v, "c0"); ps = per_slice(a, b); ms = list(ps.values())
    if len(ms) < 6:
        print(v, "NOT ENOUGH SLICES", len(ms)); continue
    m, se = st.mean(ms), st.stdev(ms) / math.sqrt(len(ms)); hi = m + T.get(len(ms) - 1, 2.0) * se
    print(f"{v}: slices {len(ms)} mean diff {m:+.4f} upper95 {hi:+.4f} ->", "PROMOTE" if hi > 0.01 else "DROP")
