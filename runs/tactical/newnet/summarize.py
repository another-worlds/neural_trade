"""Print the summary table of results.jsonl: python summarize.py [tag]"""
import json, sys
tag = sys.argv[1] if len(sys.argv) > 1 else None
f = lambda s: f"{s[0]:+.4f} [{s[1]:+.4f},{s[2]:+.4f}]"
for l in open("results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if tag and r.get("tag") != tag: continue
    s = r["summary"]; ps = r["per_slice"]
    print(f"{r['arch']:6s} {r['span']:4s} seed {r['seed']} n={r['n_slices']} params {r['params']['total']} {r['seconds']:.0f}s | mean3 {s['auc3'][0]:.4f} | "
          f"d vs lin {f(s['d_auc3'])} | ll-const {f(s['ll_gap_vs_const'])} | d ll vs lin {f(s['d_ll_h1'])} | top10 bps {s['hon10_bps'][0]:+.2f} (null95 {s['hon10_null95'][0]:+.2f}) "
          f"| mag-filt bps {s['mon10_bps'][0]:+.2f} (lin {s['mon_lin10_bps'][0]:+.2f}, trailing-vol filter {s['bon10_bps'][0]:+.2f}) | vol rho {s['vol_rho'][0]:.3f} vs base {s['vol_rho_base'][0]:.3f} d {f(s['d_vol_rho'])}")
