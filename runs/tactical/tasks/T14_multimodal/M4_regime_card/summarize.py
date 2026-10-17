"""compact table of the four evaluate results (A increments with t-CI, B relative Brier with day-block CI, controls, flags)"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); out = []
f = lambda x: "n/a" if x is None else f"{x:+.4f}"
for v in ("M4_img", "M4_cap", "M4_both", "M4b_img_gbm", "M4c_capnum"):
    r = json.load(open(os.path.join(H, v, "results.json")))
    out.append(f"== {v}  pass={r.get('pass', r.get('candidate_pass'))}")
    for lk, d in r["looks"].items():
        A = d["A"]; B = d["B"]; inc = A["increment"]
        s = lambda k: f"{f(inc[k]['mean'])}" + (f" [{f(inc[k]['ci'][0])},{f(inc[k]['ci'][1])}]" if "ci" in inc[k] else "")
        out.append(f" {lk} A: view {s('view')}  c1 {s('c1')}  c2a {s('c2a')}  c2b(M4_num) {s('c2b')}  flags {A['flags']['pass']} {[k for k,x in A['flags'].items() if not x]}  n_months {len(A['months'])}")
        R = B["rel_improvement"]; g = lambda k: f"{R[k]['mean']*100:+.2f}%" + (f" [{R[k]['ci'][0]*100:+.2f},{R[k]['ci'][1]*100:+.2f}]" if "ci" in R[k] else "")
        out.append(f" {lk} B: view {g('view')}  c1 {g('c1')}  c2a {g('c2a')}  c2b {g('c2b')}  flags {B['flags']['pass']} {[k for k,x in B['flags'].items() if not x]}  n {B['n']}")
open(os.path.join(H, "summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
