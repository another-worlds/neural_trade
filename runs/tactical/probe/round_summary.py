"""Summary of a round: each variant vs its base on direction AUC, log loss, CRPSS, risk ranking (paired, CI over slices)
and, optionally, the honest ensemble tail (thresholds from the first half of each val block, tested on the second).
usage: python round_summary.py <base> <variant>... [--honest]"""
import json, subprocess, sys
src = open(r"D:\nt\nt_tactical\runs\tactical\probe\noprice_compare.py", encoding="utf-8").read().split("res = {")[0]
ns = {}; exec(src, ns)
args = [a for a in sys.argv[1:] if not a.startswith("--")]; honest = "--honest" in sys.argv
base, variants = args[0], args[1:]
out = {}
for v in variants:
    r = ns["compare"](base, v, ("h0", "h1", "h2"), base)
    row = {k: r.get(k) for k in ("direction.auc", "direction.log_loss", "variance.crpss", "variance.corr_var_err2_spearman")}
    if honest:
        subprocess.run([sys.executable, r"D:\nt\nt_tactical\runs\tactical\probe\honest_split.py", v, "1"], capture_output=True)
        try:
            h = json.load(open(f"runs/tactical/probe/honest_split_{v}_h1.json"))
            row["mean3_10_hit"] = h.get("mean3_10_hit"); row["mean3_10_bps"] = h.get("mean3_10_bps"); row["mean3_5_hit"] = h.get("mean3_5_hit")
        except Exception as e:
            row["honest_error"] = str(e)
    out[v] = row
    f = lambda d: "n/a" if not d else f"{d['diff']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] {d['verdict']}"
    line = f"{v:20s} AUC {f(row['direction.auc'])} | logloss {f(row['direction.log_loss'])} | CRPSS {f(row['variance.crpss'])}"
    if honest and row.get("mean3_10_hit"):
        line += f" | honest top10% {row['mean3_10_hit'][0]:.3f} {row['mean3_10_bps'][0]:+.2f}bps top5% {row['mean3_5_hit'][0]:.3f}"
    print(line, flush=True)
json.dump(out, open(f"runs/tactical/probe/round_summary_{base}.json", "w"), indent=1)
