"""T03 run (SPEC.md): python run_t03.py -> results.json, tables.txt, cover_width_T{15,60,240}.png (log to stdout)."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t03_lib as L  # noqa: E402
T1 = L.T1

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

FS = ("trail", "HAR", "HAR+"); VERS = ("frozen", "cal"); METRICS = ("cover", "width_bps", "iscore_bps", "var_hit", "fz0")


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    tr_bars = T1.load_train(); fw_bars = T1.load_fwd()
    print("train bars", len(tr_bars), tr_bars.index.min(), tr_bars.index.max(), "| fwd bars", len(fw_bars), fw_bars.index.min(), fw_bars.index.max(), el(), flush=True)
    BT = L.make_data(tr_bars, L.FIRST_TRAIN); S1B = T1.build(tr_bars, 1441); BF = L.make_data(fw_bars, T1.WARM)
    print("anchors train", len(BT["D"].anch), "fwd", len(BF["D"].anch), el(), flush=True)
    fr = L.fit_frozen(BT, S1B)
    for H in L.HZ:
        print("T", H, "train rows", fr[H]["n_train"], "last label end", fr[H]["rows_last_label_end"], "frozen q", {f: {k: round(v, 3) for k, v in q.items()} for f, q in fr[H]["q"].items()}, flush=True)
    rows = []
    for H in L.HZ:
        for m in pd.PeriodIndex(L.SCORED, freq="M"):
            rows += L.score_month(fr, BF, H, m)
        print("scored T", H, el(), flush=True)
    df = pd.DataFrame(rows)

    lines = ["== T03: forward test of vol-based 90% return intervals (SPEC.md). Scored months 2025-11..2026-09 (11) ==",
             "train rows " + str({H: fr[H]["n_train"] for H in L.HZ}) + "; metrics are monthly means; width/interval score in bps of the T-minute log return"]
    res = {"train_rows": {str(H): fr[H]["n_train"] for H in L.HZ}, "frozen_q": {str(H): fr[H]["q"] for H in L.HZ}, "means": {}, "pooled": {}, "paired": {}, "criterion": {}}
    for H in L.HZ:
        lines.append(f"-- T={H}: mean over months (cover | width bps | interval score bps | VaR5 hit | FZ0)")
        for ver in VERS:
            for f in FS:
                d = df[(df.H == H) & (df.f == f) & (df.ver == ver)]
                mm = {k: float(d[k].mean()) for k in METRICS}
                pooled = float(d.n_in.sum() / d.n.sum())
                res["means"][f"{H}|{f}|{ver}"] = mm; res["pooled"][f"{H}|{f}|{ver}"] = pooled
                lines.append(f"   {ver:6s} {f:5s} cover {mm['cover']:.4f} (pooled {pooled:.4f}) | width {mm['width_bps']:.2f} | iscore {mm['iscore_bps']:.2f} | "
                             f"VaR hit {mm['var_hit']:.4f} | FZ0 {mm['fz0']:.4f}")
    # paired monthly differences (a minus b): mean, 95% t-CI, one-sided p for mean < 0
    pairs = [("HAR", "trail"), ("HAR+", "trail"), ("HAR+", "HAR")]
    lines.append("== paired monthly differences a - b (negative = a better for iscore/width/FZ0): mean [95% t-CI]  p(mean<0), months n=11 ==")
    for H in L.HZ:
        for ver in VERS:
            for a, b in pairs:
                for k in ("iscore_bps", "width_bps", "fz0", "cover", "var_hit"):
                    da = df[(df.H == H) & (df.f == a) & (df.ver == ver)].sort_values("month")[k].to_numpy()
                    db = df[(df.H == H) & (df.f == b) & (df.ver == ver)].sort_values("month")[k].to_numpy()
                    mu, lo, hi, p = L.ci(da - db)
                    res["paired"][f"{H}|{a}-{b}|{ver}|{k}"] = {"mean": mu, "ci": [lo, hi], "p_lt0": p, "months_a_lower": int((da < db).sum())}
                    if k in ("iscore_bps", "fz0", "width_bps"):
                        lines.append(f"   T={H:3d} {ver:6s} {a:5s}-{b:5s} {k:10s} {mu:+9.4f} [{lo:+9.4f},{hi:+9.4f}] p={p:.4f}  months a lower {int((da < db).sum())}/11")
    # pre-registered criterion: cal version, HAR vs trail
    ps = []; crit = {}
    for H in L.HZ:
        pr = res["paired"][f"{H}|HAR-trail|cal|iscore_bps"]; pc = res["pooled"][f"{H}|HAR|cal"]
        c1 = pr["ci"][1] < 0 and pr["mean"] < 0; c2 = 0.88 <= pc <= 0.92
        crit[H] = {"mean_diff_iscore_bps": pr["mean"], "ci": pr["ci"], "p_lt0": pr["p_lt0"], "c1_ci_below_0": bool(c1), "pooled_cover": pc, "c2_cover_in_band": bool(c2),
                   "success": bool(c1 and c2)}
        ps.append(pr["p_lt0"])
    adj = L.holm(ps)
    for H, a in zip(L.HZ, adj):
        crit[H]["holm_p"] = float(a); crit[H]["success_holm"] = bool(crit[H]["success"] and a < 0.05)
    res["criterion"] = {str(H): v for H, v in crit.items()}; res["task_pass"] = bool(crit[60]["success"])
    lines.append("== PRE-REGISTERED CRITERION (cal, HAR vs trail; iscore diff CI entirely below 0 and pooled HAR coverage in [0.88, 0.92]) ==")
    for H in L.HZ:
        lines.append(f"   T={H}: {crit[H]}")
    lines.append(f"   TASK PASS (T=60 success): {res['task_pass']}")
    lines.append("== per-month rows (H, month, ver, forecaster: n, dropped, cover, width, iscore, VaR hit, FZ0) ==")
    for _, r in df.sort_values(["H", "ver", "f", "month"]).iterrows():
        lines.append(f"{r.H} {r.month} {r.ver:6s} {r.f:5s} n={r.n} drop={r.n_dropped} ncal={r.n_cal} cov={r.cover:.4f} w={r.width_bps:.2f} is={r.iscore_bps:.2f} var={r.var_hit:.4f} fz0={r.fz0:.4f}")
    res["runtime_s"] = time.time() - t0
    json.dump({"summary": res, "rows": json.loads(df.to_json(orient="records"))}, open(f"{HERE}/results.json", "w"), indent=1)
    open(f"{HERE}/tables.txt", "w").write("\n".join(lines))
    print("\n".join(lines[:80]), flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neural_trade.visualization import theme as TH
    cols = {"trail": TH.NEUTRAL, "HAR": TH.SERIES[0], "HAR+": TH.SERIES[1]}
    for H in L.HZ:
        fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=TH.PAPER, sharex=True)
        for j, ver in enumerate(VERS):
            for i, (k, yl) in enumerate((("cover", "coverage (target 0.90)"), ("width_bps", "mean width, bps"))):
                a_ = ax[i, j]; a_.set_facecolor(TH.SURFACE); a_.grid(color=TH.GRID, lw=.6)
                for sp in a_.spines.values():
                    sp.set_color(TH.AXIS)
                a_.tick_params(colors=TH.MUTED)
                for f in FS:
                    d = df[(df.H == H) & (df.f == f) & (df.ver == ver)].sort_values("month")
                    a_.plot(d.month, d[k], color=cols[f], marker="o", lw=1.4, ms=4, ls={"trail": "--", "HAR": "-", "HAR+": ":"}[f], label=f)
                if k == "cover":
                    a_.axhline(0.90, color=TH.INK, lw=.8, ls=":"); a_.axhspan(0.88, 0.92, color=TH.INK, alpha=.07)
                a_.set_ylabel(yl, color=TH.INK); a_.set_title(f"{ver}", color=TH.INK, fontsize=10)
                if i == 1:
                    a_.tick_params(axis="x", rotation=60)
        ax[0, 0].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK)
        pr = res["paired"][f"{H}|HAR-trail|cal|iscore_bps"]
        fig.suptitle(f"T03 T={H} min: 90% interval coverage and width by month (cal diff HAR-trail interval score {pr['mean']:+.2f} bps [{pr['ci'][0]:+.2f},{pr['ci'][1]:+.2f}])",
                     color=TH.INK, fontsize=11)
        fig.tight_layout(); fig.savefig(f"{HERE}/cover_width_T{H}.png", dpi=110, facecolor=TH.PAPER); plt.close(fig)
    print("done", el())


if __name__ == "__main__":
    main()
