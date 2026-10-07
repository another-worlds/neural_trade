"""The tactical hill-climb's aggregated accuracy metric (owner, 2026-10-07: "3 groups equally, scaled by noise";
fixed before any hill-climb round uses it).

Groups and metrics (each the mean over h0-h2 of the screen row's head_metrics; sign so that higher = better):
  price      delta.corr, delta.skill_vs_zero
  direction  direction.auc, -direction.log_loss
  confidence variance.crpss, -|variance.coverage90 - 0.90|, variance.corr_var_err2_spearman
Noise unit of a metric: the sd of its paired difference between two runs of the SAME configuration on the same slice
with different seeds (the base's seed 0 vs seed 1), so a candidate's paired difference / that sd is its effect in
"seed-noise units".
Per slice: z_m = mean over the slice's seeds of (candidate - base) / sd_m; group score = mean of its metrics' z;
composite = mean of the 3 group scores. Verdict over slices (unit of inference = slice, the 6x2 design):
  BETTER  if the 95% t-interval of the composite over slices is above 0 AND no group's mean score is below -0.5
          AND (anti-collapse guard, added 2026-10-07 after bench H10, before any hill-climb round) the two resolution
          metrics - direction.auc and variance.corr_var_err2_spearman - each have a mean score >= -0.5: a net that stops
          predicting (constant delta, constant variance) gains on calibration-type metrics but loses resolution;
  WORSE   if the interval is below 0; otherwise NO DIFFERENCE.
usage: python hc4_metric.py <base_spec_name> <candidate_spec_name> [noise_spec_name]   (noise defaults to the base)
"""
import collections, glob, json, math, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hc2_compare import T

ROOT = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"price": [("delta", "corr", 1), ("delta", "skill_vs_zero", 1)],
          "direction": [("direction", "auc", 1), ("direction", "log_loss", -1)],
          "confidence": [("variance", "crpss", 1), ("variance", "coverage90", "cov"), ("variance", "corr_var_err2_spearman", 1)]}
H = ("h0", "h1", "h2")


def metric(row, g, k, sign):
    v = [row["head_metrics"][h][g].get(k) for h in H if row.get("head_metrics") and row["head_metrics"].get(h)]
    v = [x for x in v if x is not None]
    if not v:
        return None
    m = st.mean(v)
    return -abs(m - 0.90) if sign == "cov" else sign * m


def load(name):
    d = {}
    for f in glob.glob(os.path.join(ROOT, "screens", name, "results*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if (r.get("health") or {}).get("calib_failed"):
                continue  # the calibration pass failed (e.g. GPU out of memory): the trial trained uncalibrated, not as the variant
            if r.get("head_metrics"):
                d[(r["data_end"][:16], r["seed"])] = {f"{g}.{k}": metric(r, g, k, s) for grp in GROUPS.values() for g, k, s in grp}
    return d


def noise(base):
    """sd of the seed-0 minus seed-1 difference per metric, over the slices that have both."""
    by = collections.defaultdict(dict)
    for (s, seed), m in base.items():
        by[s][seed] = m
    out = {}
    for grp in GROUPS.values():
        for g, k, _ in grp:
            key = f"{g}.{k}"
            d = [v[0][key] - v[1][key] for v in by.values() if 0 in v and 1 in v and v[0][key] is not None and v[1][key] is not None]
            out[key] = st.stdev(d) if len(d) > 1 else None
    return out


def compare(base_name, cand_name, noise_name=None):
    base, cand = load(base_name), load(cand_name)
    sd = noise(load(noise_name) if noise_name else base)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    res_z = {"direction.auc": [], "variance.corr_var_err2_spearman": []}
    for k, m in cand.items():
        if k not in base:
            continue
        for gname, grp in GROUPS.items():
            zs = [(m[f"{g}.{kk}"] - base[k][f"{g}.{kk}"]) / sd[f"{g}.{kk}"] for g, kk, _ in grp
                  if m[f"{g}.{kk}"] is not None and base[k][f"{g}.{kk}"] is not None and sd[f"{g}.{kk}"]]
            if zs:
                per[k[0]][gname].append(st.mean(zs))
        for key in res_z:
            if m.get(key) is not None and base[k].get(key) is not None and sd.get(key):
                res_z[key].append((m[key] - base[k][key]) / sd[key])
    slices = {s: {g: st.mean(v) for g, v in d.items()} for s, d in per.items()}
    comp = [st.mean(v.values()) for v in slices.values() if len(v) == 3]
    res = {"slices": len(comp), "noise_sd": sd,
           "groups": {g: st.mean([v[g] for v in slices.values() if g in v]) for g in GROUPS if any(g in v for v in slices.values())}}
    if len(comp) >= 2:
        m = st.mean(comp); se = st.stdev(comp) / math.sqrt(len(comp)); t = T.get(len(comp) - 1, 2.0)
        res.update(mean=m, lo=m - t * se, hi=m + t * se)
        res["resolution"] = {k: st.mean(v) for k, v in res_z.items() if v}
        guard = all(v >= -0.5 for v in res["groups"].values()) and all(v >= -0.5 for v in res["resolution"].values())
        res["verdict"] = "BETTER" if res["lo"] > 0 and guard else ("WORSE" if res["hi"] < 0 else "NO DIFFERENCE")
    return res


if __name__ == "__main__":
    r = compare(*sys.argv[1:4])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "noise_sd"}, ensure_ascii=False))
    print("noise sd:", {k: round(v, 4) for k, v in r["noise_sd"].items() if v})
