"""tdconfirm analysis (SPEC.md). usage: python analyse_confirm.py -> tables_{conf,repl}.txt, confirm.json, curve_{conf,repl}.png"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_lib as C  # noqa: E402
import tdvar_lib as L  # noqa: E402

HCOL = {240: "#3987e5", 1440: "#d95926"}
MIN_GAIN = 2.0


def load(period):
    rows = []
    for p in sorted(glob.glob(f"{HERE}/results_{period}_*.jsonl")):
        rows += [json.loads(line) for line in open(p)]
    meta = []
    for p in sorted(glob.glob(f"{HERE}/meta_{period}_*.jsonl")):
        meta += [json.loads(line) for line in open(p)]
    return pd.DataFrame(rows), meta


def block_boot(v, block=3, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = len(v)
    nb = -(-m // block)
    out = [np.concatenate([v[(s + np.arange(block)) % m] for s in rng.integers(0, m, nb)])[:m].mean() for _ in range(n)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m, lo, hi = L.t_ci(v)
    bl, bh = block_boot(v)
    return dict(n=int(len(v)), mean=m, t_lo=lo, t_hi=hi, boot_lo=bl, boot_hi=bh, pos=int((v > 0).sum()))


def analyse(period):
    df, meta = load(period)
    q = df.pivot_table(index=["W", "H", "month"], columns="model", values="qlike")
    trunc = df.groupby(["W", "month"])["trunc"].max()
    res = dict(period=period, months=sorted(df.month.unique()), n_months=int(df.month.nunique()), cells={}, skipped=
               sum(len(m["skipped"]) for m in meta), sec_per_month=float(np.mean([m["sec"] for m in meta])),
               total_cpu_sec=float(np.sum([m["sec"] for m in meta])))
    lines = [f"== {period}: months {res['months'][0]} .. {res['months'][-1]} ({res['n_months']}); skipped cells {res['skipped']}"]
    neff = df.groupby("H").size()  # placeholder to keep pandas quiet
    n_eff = {(int(W), int(H)): float(np.mean([c["n_eff_test"] for m in meta for k, c in m["cells"].items()
                                              if k == f"{W}_{H}"])) for W in C.WINDOWS for H in C.HORIZONS}
    del neff
    for H in C.HORIZONS:
        lines.append(f"\n-- H = {H}: mean monthly QLIKE by window (all months of the period; 365 truncated months flagged below)")
        hdr = ["W", "nn_td", "nn_direct"] + list(C.BASELINES) + ["midas_sse", "best_baseline", "n_eff_test"]
        lines.append("  ".join(f"{h:>13}" for h in hdr))
        for W in C.WINDOWS:
            d = q.loc[(W, H)]
            best = min(C.BASELINES, key=lambda b: d[b].mean())
            vals = [d["nn_td"].mean(), d["nn_direct"].mean()] + [d[b].mean() for b in C.BASELINES] + [d["midas_sse"].mean()]
            lines.append("  ".join([f"{W:>13}"] + [f"{v:>13.4f}" for v in vals] + [f"{best:>13}", f"{n_eff[(W, H)]:>13.1f}"]))
    cond = {}
    for H in C.HORIZONS:
        lines.append(f"\n-- H = {H}: gain % of nn_td (mean of monthly gains, 95% t-interval | 95% block bootstrap | months gain>0)")
        for W in C.WINDOWS:
            d = q.loc[(W, H)]
            best = min(C.BASELINES, key=lambda b: d[b].mean())
            gb = 100 * (d[best] - d["nn_td"]) / d[best]
            orc = d[list(C.BASELINES)].min(axis=1)
            go = 100 * (orc - d["nn_td"]) / orc
            gd = 100 * (d["nn_direct"] - d["nn_td"]) / d["nn_direct"]
            cell = dict(best=best, gain_best=summ(gb), gain_oracle=summ(go), gain_direct=summ(gd),
                        per_baseline={b: summ(100 * (d[b] - d["nn_td"]) / d[b]) for b in C.BASELINES})
            res["cells"][f"{W}_{H}"] = cell
            cond[(W, H)] = (gb, gd)
            for lab, key in (("vs BEST baseline (" + best + ")", "gain_best"), ("vs per-month oracle", "gain_oracle"),
                             ("vs nn_direct", "gain_direct")):
                s = cell[key]
                lines.append(f"W={W:>4} {lab:<42} {s['mean']:+7.2f}  t[{s['t_lo']:+7.2f},{s['t_hi']:+7.2f}]  "
                             f"boot[{s['boot_lo']:+7.2f},{s['boot_hi']:+7.2f}]  {s['pos']}/{s['n']}")
            lines.append("        per baseline: " + "  ".join(f"{b} {cell['per_baseline'][b]['mean']:+.1f}" for b in C.BASELINES))
    # C1
    c1 = {}
    for H in C.HORIZONS:
        s = res["cells"][f"60_{H}"]["gain_best"]
        c1[H] = bool(s["mean"] >= MIN_GAIN and s["t_lo"] > 0 and s["boot_lo"] > 0)
    # C2: months where all four windows are complete
    # SPEC: a 365-day window is complete when t0 - 365 d is not before the first anchor (2017-01-08): 2018-02.. for the confirmation
    first_anchor = pd.Timestamp("2017-01-08")
    full = [m for m in res["months"] if pd.Period(m).start_time - pd.Timedelta(days=365) >= first_anchor
            and not any(bool(trunc.get((W, m), False)) for W in C.WINDOWS)]
    c2, rho, curve_full = {}, {}, {}
    for H in C.HORIZONS:
        means = []
        for W in C.WINDOWS:
            d = q.loc[(W, H)].loc[full]
            gd = 100 * (d["nn_direct"] - d["nn_td"]) / d["nn_direct"]
            means.append(summ(gd))
        curve_full[H] = means
        r = stats.spearmanr(np.log(C.WINDOWS), [m["mean"] for m in means])[0]
        rho[H], c2[H] = float(r), bool(r < 0)
    rho3 = {}
    for H in C.HORIZONS:
        g3 = [res["cells"][f"{W}_{H}"]["gain_direct"]["mean"] for W in (30, 60, 90)]
        rho3[H] = float(stats.spearmanr(np.log([30, 60, 90]), g3)[0])
    ctrl = {H: curve_full[H][-1] for H in C.HORIZONS}
    point_pos = all(res["cells"][f"60_{H}"]["gain_best"]["mean"] > 0 for H in C.HORIZONS)
    verdict = "CONFIRMED" if all(c1.values()) and all(c2.values()) else (
        "PARTIAL" if all(c2.values()) and point_pos else "NOT CONFIRMED")
    res.update(C1=c1, C2=c2, spearman_full_months=rho, spearman_30_60_90_all_months=rho3, full_months=full, n_full_months=len(full),
               control_365=ctrl, verdict=verdict)
    lines.append(f"\n-- C2 months (all four windows complete): {len(full)} ({full[0] if full else '-'} .. {full[-1] if full else '-'})")
    for H in C.HORIZONS:
        lines.append(f"H={H}: gain vs nn_direct by W {list(C.WINDOWS)}: " + "  ".join(f"{m['mean']:+.2f}" for m in curve_full[H])
                     + f"  Spearman(gain, log W) = {rho[H]:+.2f}  (30/60/90 on all months: {rho3[H]:+.2f})")
        c = ctrl[H]
        lines.append(f"      control W=365: {c['mean']:+.2f} [{c['t_lo']:+.2f},{c['t_hi']:+.2f}]")
    lines.append(f"\nC1 (W=60, gain_best >= {MIN_GAIN}% and both 95% intervals > 0): {c1}")
    lines.append(f"C2 (Spearman < 0): {c2}")
    lines.append(f"VERDICT ({period}): {verdict}")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, H in zip(axs, C.HORIZONS):
        for key, ls, lab in (("gain_direct", "-", "vs nn_direct"), ("gain_best", "--", "vs best baseline")):
            m = [res["cells"][f"{W}_{H}"][key] for W in C.WINDOWS]
            ax.errorbar(range(4), [x["mean"] for x in m], yerr=[[x["mean"] - x["t_lo"] for x in m], [x["t_hi"] - x["mean"] for x in m]],
                        color=HCOL[H], ls=ls, marker="o", capsize=3, label=lab)
        ax.set_yscale("symlog", linthresh=5)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(MIN_GAIN, color="gray", lw=0.8, ls=":")
        ax.set_xticks(range(4), [str(W) for W in C.WINDOWS])
        ax.set_xlabel("training window, days (365 = control)")
        ax.set_title(f"{period}: H = {H} min, nn_td QLIKE gain %, 95% t-interval")
        ax.legend()
    axs[0].set_ylabel("gain %")
    fig.tight_layout()
    fig.savefig(f"{HERE}/curve_{period}.png", dpi=110)
    open(f"{HERE}/tables_{period}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main():
    out = {}
    for p in ("conf", "repl"):
        if glob.glob(f"{HERE}/results_{p}_*.jsonl"):
            out[p] = analyse(p)
    json.dump(out, open(f"{HERE}/confirm.json", "w"), indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))


if __name__ == "__main__":
    main()
