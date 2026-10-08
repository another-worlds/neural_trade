"""No-price experiment (PRICE_HEAD none): A) 3-horizon no-price vs the base with price (cand_c2_base), all horizons;
B) 1-horizon no-price (ACTIVE_HORIZONS [1]) vs 3-horizon no-price, on h1. Paired by (slice, seed); per-slice means;
95% t-interval over the 6 slices. Price metrics are left out (no price head). Noise unit: the base's seed-0 vs seed-1 sd."""
import collections, glob, json, math, statistics as st, sys
sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical")
from hc2_compare import T

MET = [("direction", "auc", 1), ("direction", "log_loss", -1), ("direction", "brier", -1), ("direction", "hit_rate", 1),
       ("variance", "crpss", 1), ("variance", "coverage90", "cov"), ("variance", "corr_var_err2_spearman", 1)]


def load(name, hs):
    d = {}
    for f in glob.glob(f"runs/tactical/screens/{name}/results*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); hm = r.get("head_metrics")
            if not hm or (r.get("health") or {}).get("calib_failed"):
                continue
            row = {}
            for g, k, sg in MET:
                v = [hm[h][g].get(k) for h in hs if hm.get(h) and hm[h].get(g) and hm[h][g].get(k) is not None]
                if v:
                    m = st.mean(v); row[f"{g}.{k}"] = -abs(m - 0.9) if sg == "cov" else sg * m
            d[(r["data_end"][:16], r["seed"])] = row
    return d


def compare(a_name, b_name, hs, noise_name):
    a, b, nz = load(a_name, hs), load(b_name, hs), load(noise_name, hs)
    out = {}
    for g, k, _ in MET:
        key = f"{g}.{k}"
        by = collections.defaultdict(dict)
        for (s, sd), r in nz.items():
            if key in r: by[s][sd] = r[key]
        nd = [v[0] - v[1] for v in by.values() if 0 in v and 1 in v]
        sd = st.stdev(nd) if len(nd) > 1 else None
        per = collections.defaultdict(list)
        for kk in set(a) & set(b):
            if key in a[kk] and key in b[kk]:
                per[kk[0]].append(b[kk][key] - a[kk][key])
        ms = [st.mean(v) for v in per.values()]
        if len(ms) >= 2:
            m = st.mean(ms); se = st.stdev(ms) / math.sqrt(len(ms)); t = T.get(len(ms) - 1, 2.0)
            out[key] = {"diff": round(m, 4), "ci": [round(m - t * se, 4), round(m + t * se, 4)], "slices": len(ms),
                        "noise_units": round(m / sd, 2) if sd else None,
                        "verdict": "better" if m - t * se > 0 else ("worse" if m + t * se < 0 else "no difference")}
    return out


res = {"A_noprice3_vs_base_all_h": compare("cand_c2_base", "hc5_noprice3", ("h0", "h1", "h2"), "cand_c2_base"),
       "A_noprice3_vs_base_h1": compare("cand_c2_base", "hc5_noprice3", ("h1",), "cand_c2_base"),
       "B_noprice1_vs_noprice3_h1": compare("hc5_noprice3", "hc5_noprice1", ("h1",), "cand_c2_base"),
       "C_noprice1_vs_base_h1": compare("cand_c2_base", "hc5_noprice1", ("h1",), "cand_c2_base")}
for k, v in res.items():
    print("==", k)
    for m, r in v.items():
        print(f"   {m:34s} {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  {r['noise_units']} noise u.  {r['verdict']}")
json.dump(res, open("runs/tactical/probe/noprice_compare.json", "w"), indent=1)
