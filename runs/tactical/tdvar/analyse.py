"""tdvar analysis (SPEC.md). usage: python analyse.py sel|look  -> <stage>_tables.txt, <stage>.json, curve_<stage>.png"""
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
import tdvar_lib as L  # noqa: E402

COMMON = ("2020-02", "2024-06")
MODELS = ("direct", "iterated", "td", "td_recent", "midas", "naive")
HCOL = {240: "#3987e5", 1440: "#d95926", 60: "#199e70"}


def load(kind, stage):
    rows = []
    for p in sorted(glob.glob(f"{HERE}/results_{kind}_{stage}_*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    return pd.DataFrame(rows)


def block_boot(v, block=3, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed); m = len(v); nb = -(-m // block); out = []
    for _ in range(n):
        st = rng.integers(0, m, nb)
        out.append(np.concatenate([v[(s + np.arange(block)) % m] for s in st])[:m].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def gains(df, model, base):
    """Per (W, H) per-month relative gain (%) of `model` over `base`."""
    p = df.pivot_table(index=["W", "H", "month"], columns="model", values="qlike")
    g = 100.0 * (p[base] - p[model]) / p[base]
    return g.rename("gain").reset_index()


def curve(g, sel_common=True):
    rows = []
    for (W, H), d in g.groupby(["W", "H"]):
        m, lo, hi = L.t_ci(d["gain"].values)
        bl, bh = block_boot(d["gain"].values)
        n = int(d["gain"].notna().sum())
        p = float(stats.ttest_1samp(d["gain"].dropna(), 0.0, alternative="greater").pvalue) if n > 2 else float("nan")
        rows.append(dict(W=W, H=H, n_months=n, gain=m, lo=lo, hi=hi, boot_lo=bl, boot_hi=bh, p_gt0=p,
                         frac_pos=float((d["gain"] > 0).mean())))
    return pd.DataFrame(rows).sort_values(["H", "W"])


def holm(ps, alpha=0.05):
    order = np.argsort(ps); out = [False] * len(ps)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k):
            out[i] = True
        else:
            break
    return out


def spearman_curve(c, H):
    d = c[c.H == H].sort_values("W")
    if len(d) < 3:
        return float("nan")
    return float(stats.spearmanr(np.log(d["W"]), d["gain"])[0])


def verdict(c, stage):
    r = {}
    ok1 = []
    for H in (240, 1440):
        d = c[(c.H == H) & (c.W == 60)].iloc[0]
        r[f"gain_60_{H}"] = float(d["gain"]); r[f"gain_1095_{H}"] = float(c[(c.H == H) & (c.W == 1095)].iloc[0]["gain"])
        r[f"rho_{H}"] = spearman_curve(c, H)
    ps = [float(c[(c.H == H) & (c.W == 60)].iloc[0]["p_gt0"]) for H in (240, 1440)]
    hz = holm(ps)
    r["holm_60"] = dict(zip(("240", "1440"), hz)); r["p_60"] = ps
    c1_point = all(r[f"gain_60_{H}"] >= 1.0 for H in (240, 1440))
    c1 = c1_point and (all(hz) if stage == "sel" else True)
    c2 = all(r[f"gain_1095_{H}"] <= 0.3 for H in (240, 1440))
    c3 = all(r[f"rho_{H}"] < 0 for H in (240, 1440))
    r.update(C1=bool(c1), C1_point=bool(c1_point), C2=bool(c2), C3=bool(c3))
    r["verdict"] = "SUPPORTED" if (c1 and c2 and c3) else ("REFUTED (gain does not fall)" if not c3 else "PARTIAL")
    return r


def fmt_curve(c, title):
    s = [title, f"{'H':>5} {'W':>5} {'n':>3} {'gain %':>8} {'t95 lo':>8} {'t95 hi':>8} {'boot lo':>8} {'boot hi':>8} {'p>0':>7} {'frac+':>6}"]
    for _, r in c.iterrows():
        s.append(f"{int(r.H):>5} {int(r.W):>5} {int(r.n_months):>3} {r.gain:>8.3f} {r.lo:>8.3f} {r.hi:>8.3f} {r.boot_lo:>8.3f} {r.boot_hi:>8.3f} {r.p_gt0:>7.3f} {r.frac_pos:>6.2f}")
    return "\n".join(s)


def level_table(df, title):
    p = df.groupby(["H", "W", "model"])[["qlike", "spearman"]].mean().reset_index()
    s = [title, f"{'H':>5} {'W':>5} " + " ".join(f"{m:>10}" for m in MODELS) + "   | spearman: " + " ".join(f"{m:>8}" for m in MODELS)]
    for (H, W), d in p.groupby(["H", "W"]):
        q = d.set_index("model")
        s.append(f"{H:>5} {W:>5} " + " ".join(f"{q.loc[m, 'qlike']:>10.4f}" if m in q.index else f"{'-':>10}" for m in MODELS) +
                 "   | " + " ".join(f"{q.loc[m, 'spearman']:>8.3f}" if m in q.index else f"{'-':>8}" for m in MODELS))
    return "\n".join(s)


def plot(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for ax, H in zip(axes, (240, 1440, 60)):
        for name, c, ls in curves:
            d = c[c.H == H].sort_values("W")
            if d.empty:
                continue
            ax.errorbar(d["W"], d["gain"], yerr=[d["gain"] - d["lo"], d["hi"] - d["gain"]], fmt="o" + ls, capsize=3, label=name)
        ax.axhline(0, color="#888", lw=0.8); ax.set_xscale("log"); ax.set_xticks(L.WINDOWS); ax.set_xticklabels(L.WINDOWS)
        if H in (240, 1440):
            ax.axhline(1.0, color="#bbb", ls=":", lw=0.8); ax.axhline(0.3, color="#bbb", ls=":", lw=0.8)
        ax.set_title(f"H = {H} min" + (" (secondary)" if H == 60 else ""), color=HCOL[H]); ax.set_xlabel("training window, days")
        ax.set_ylabel("QLIKE gain over direct HAR, % (95% t-interval)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    stage = sys.argv[1]
    df = load("lin", stage)
    if stage == "sel":
        common = df[(df.month >= COMMON[0]) & (df.month <= COMMON[1]) & (~df.trunc)]
        early = df[(df.month < COMMON[0])]
    else:
        common, early = df, None
    out = {"stage": stage, "n_rows": len(df), "months": sorted(df.month.unique())[:1] + sorted(df.month.unique())[-1:]}
    txt = []
    curves = []
    for model in ("td", "iterated", "td_recent", "midas", "naive"):
        g = gains(common, model, "direct"); c = curve(g)
        txt.append(fmt_curve(c, f"\n== gain over direct HAR, model {model}, {stage}, months {common.month.min()}..{common.month.max()} (complete windows only) =="))
        out[f"curve_{model}"] = c.to_dict("records")
        if model in ("td", "midas", "td_recent"):
            curves.append((model, c, "-" if model == "td" else "--"))
        if model == "td":
            out["verdict"] = verdict(c, stage)
    # TD against the strongest baseline per cell (min over iterated, midas of mean qlike)
    for base in ("midas", "iterated"):
        g = gains(common, "td", base); c = curve(g)
        txt.append(fmt_curve(c, f"\n== gain of td over {base} =="))
        out[f"curve_td_vs_{base}"] = c.to_dict("records")
    txt.append(level_table(common, "\n== mean QLIKE (and Spearman) by model =="))
    if early is not None and len(early):
        ge = gains(early, "td", "direct"); ce = curve(ge)
        txt.append(fmt_curve(ce, "\n== short-history months 2019-01..2020-01, td over direct (windows may be truncated; shown, not used) =="))
        out["curve_td_early"] = ce.to_dict("records")
    meta = [json.loads(l) for p in glob.glob(f"{HERE}/meta_lin_{stage}_*.jsonl") for l in open(p)]
    cell = [v for m in meta for v in m["cells"].values()]
    txt.append("\n== n_eff (mean per month) ==\n" + "\n".join(
        f"H={H} W={W}: n_train {np.mean([m['cells'][f'{W}_{H}']['n_train'] for m in meta if f'{W}_{H}' in m['cells']]):.0f}, "
        f"n_eff_train {np.mean([m['cells'][f'{W}_{H}']['n_eff_train'] for m in meta if f'{W}_{H}' in m['cells']]):.1f}, "
        f"n_eff_test {np.mean([m['cells'][f'{W}_{H}']['n_eff_test'] for m in meta if f'{W}_{H}' in m['cells']]):.1f}"
        for H in L.HORIZONS for W in L.WINDOWS))
    out["wall_sec_total"] = float(sum(m["sec"] for m in meta))
    nn = load("nn", stage)
    if len(nn):
        nc = nn[(nn.month >= COMMON[0]) & (~nn.trunc)] if stage == "sel" else nn
        for base, mod in (("nn_direct", "nn_td"), ("direct", "nn_direct"), ("direct", "nn_td")):
            lin = common[common.model.isin(["direct"])]
            both = pd.concat([nc, lin[lin.month.isin(nc.month.unique())]])
            both = both[both.model.isin([base, mod])]
            g = gains(both, mod, base); c = curve(g)
            txt.append(fmt_curve(c, f"\n== NEURAL: gain of {mod} over {base} =="))
            out[f"curve_{mod}_vs_{base}"] = c.to_dict("records")
            if (base, mod) == ("nn_direct", "nn_td"):
                curves.append(("nn_td vs nn_direct", c, ":"))
    v = out["verdict"]
    txt.insert(0, f"verdict ({stage}, headline td vs direct): {json.dumps(v, indent=1)}")
    open(f"{HERE}/{stage}_tables.txt".replace("sel_tables", "selection_tables").replace("look_tables", "look_tables"), "w").write("\n".join(txt))
    json.dump(out, open(f"{HERE}/{'selection' if stage == 'sel' else 'look'}.json", "w"), indent=1, default=float)
    plot(curves, f"{HERE}/curve_{stage}.png", f"tdvar {stage}: gain over direct HAR vs training window")
    print("\n".join(txt)[:6000])


if __name__ == "__main__":
    main()
