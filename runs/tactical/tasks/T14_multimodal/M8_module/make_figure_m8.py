"""The M8 figure (SPEC_M8 (d)): (top) ablation bars, A increment and B relative Brier improvement per look, full model, each key group removed, M8_lin;
(bottom) a real-day strip: Binance 2026-03, the lower-median day by activity (rule in the SPEC), V3 against V3 + the frozen module (M7c walk-forward view)
forecasts of the next-hour change in log RV, against the realised change. Run: D:/nt/nt_mm_env/python.exe make_figure_m8.py -> fig_m8.png, fig_m8.json, fig_m8_day.txt"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402
import evaluate as EV  # noqa: E402

th = EV.theme_colors()
R = json.load(open(os.path.join(HERE, "results_m8.json"))); FR = json.load(open(os.path.join(HERE, "frozen_P.json")))
MONTH = pd.Period("2026-03", "M")


def pick_day():
    base = AN.load_base("fwd"); view = VC.read_view(FR["module"], "fwd"); M, cols = VC.align_view(view, base.ta)
    valid = base.ok & np.isfinite(M).all(1)
    rows = EV.run_A(base, valid, [MONTH], AN.LOOKS["look2"]["train_from"], {"V3": None, "view": M}, keep_preds=True)[0]
    te = rows["te"]; ta = base.ta[te]; day = np.asarray(ta.normalize()); lv0 = base.A["LV0"][te]; y = base.A["T"]["A"][te]
    stats = []
    for d in np.unique(day):
        m = day == d
        if m.sum() >= 20:
            stats.append((pd.Timestamp(d), float(np.mean(lv0[m])), np.where(m)[0]))
    stats.sort(key=lambda s: s[1]); k = (len(stats) - 1) // 2; d, act, idx = stats[k]
    out = {"day": d, "activity": act, "n_days": len(stats), "rank_index": k, "hours": np.asarray(ta[idx].hour), "y": y[idx], "v3": rows["preds"]["V3"][idx], "mod": rows["preds"]["view"][idx],
           "month_skill": rows["skill"], "activity_all": [(str(s[0].date()), round(s[1], 3)) for s in stats]}
    return out


def main():
    day = pick_day()
    names = [("M7_gbm (full)", None), ("without stretch", "stretch"), ("without volpct", "volpct"), ("without hourwk", "hourwk"), ("without sr", "sr"), ("without lrv", "lrv"), ("M8_lin", "lin")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"look1": th.OTHER_SERIES[0], "look2": th.OTHER_SERIES[1]}
    fig = plt.figure(figsize=(14, 9.2), facecolor=th.PAPER); gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    pl = {"data": [], "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 900, "barmode": "group",
                                 "title": {"text": "T14 M8: which key group carries the M7 gain, and a real March-2026 day"}, "annotations": []}}
    doms = [([0.0, 0.46], [0.56, 0.94]), ([0.54, 1.0], [0.56, 0.94]), ([0.0, 1.0], [0.0, 0.42])]
    for j, (tg, pct) in enumerate((("A", False), ("B", True))):
        ax = axs[j]; ax.set_facecolor(th.SURFACE); sfx = "" if j == 0 else "2"
        def get(g, lk):
            if g is None:
                h = R["ablation"][tg]["full"][lk]
            elif g == "lin":
                h = R["linear"][tg][lk]
            else:
                h = R["ablation"][tg]["groups"][g][lk]
            s = 100 if pct else 1
            return h["mean"] * s, (h["mean"] - h["lo"]) * s, (h["hi"] - h["mean"]) * s
        x = np.arange(len(names))
        for i, lk in enumerate(("look1", "look2")):
            v = np.array([get(g, lk) for _, g in names]); w = 0.38
            ax.bar(x + (i - 0.5) * w, v[:, 0], w, yerr=[v[:, 1], v[:, 2]], color=col[lk], ecolor=th.INK_2, capsize=2, label=f"{lk} (harness 95% CI)")
            pl["data"].append({"type": "bar", "name": lk, "x": [n for n, _ in names], "y": [float(a) for a in v[:, 0]], "marker": {"color": col[lk]},
                               "error_y": {"type": "data", "symmetric": False, "array": [float(a) for a in v[:, 2]], "arrayminus": [float(a) for a in v[:, 1]]},
                               "xaxis": "x" + sfx, "yaxis": "y" + sfx, "legendgroup": lk, "showlegend": bool(j == 0)})
        thr = (M8.CRIT_B * 100) if pct else M8.CRIT_A
        ax.axhline(0, color=th.NEUTRAL, lw=0.8); ax.axhline(thr, color=th.GOOD, lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" (full)", "\n(full)").replace("without ", "w/o\n") for n, _ in names], fontsize=7.5, color=th.MUTED)
        ttl = "A: skill increment over V3 (monthly mean)" if tg == "A" else "B: relative Brier improvement over H3 (%)"
        ax.set_title(ttl, color=th.INK, fontsize=9.5); ax.tick_params(colors=th.MUTED, labelsize=8); ax.grid(color=th.GRID, lw=0.5, axis="y")
        for sp in ax.spines.values():
            sp.set_color(th.AXIS)
        if j == 0:
            ax.legend(fontsize=7.5, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2)
        pl["layout"]["xaxis" + sfx] = {"domain": doms[j][0], "anchor": "y" + sfx, "gridcolor": th.GRID}
        pl["layout"]["yaxis" + sfx] = {"domain": doms[j][1], "anchor": "x" + sfx, "gridcolor": th.GRID, "zerolinecolor": th.AXIS, "title": {"text": ttl}}
        pl["layout"].setdefault("shapes", []).append({"type": "line", "xref": "paper", "yref": "y" + sfx, "x0": doms[j][0][0], "x1": doms[j][0][1], "y0": thr, "y1": thr,
                                                      "line": {"color": th.GOOD, "dash": "dash", "width": 1}})
    ax = axs[2]; ax.set_facecolor(th.SURFACE); h = day["hours"]
    mse = lambda p: float(np.mean((day["y"] - p) ** 2))
    ax.plot(h, day["y"], marker="o", ms=5, lw=1.2, color=th.INK_2, label="realised change in log RV (next hour)")
    ax.plot(h, day["v3"], marker="s", ms=4, lw=1.6, color=th.OTHER_SERIES[1], label=f"V3 forecast (day MSE {mse(day['v3']):.3f})")
    ax.plot(h, day["mod"], marker="^", ms=4, lw=1.6, color=th.OTHER_SERIES[0], label=f"V3 + module forecast (day MSE {mse(day['mod']):.3f})")
    ax.axhline(0, color=th.NEUTRAL, lw=0.8)
    d = str(day["day"].date()); ttl = (f"Binance {d} (UTC), the lower-median day of 2026-03 by activity (rank {day['rank_index'] + 1} of {day['n_days']}): "
                                       f"forecast of the change in log RV over the next hour, by anchor hour. One day, descriptive only.")
    ax.set_title(ttl, color=th.INK, fontsize=9); ax.set_xlabel("anchor hour (UTC)", color=th.INK_2, fontsize=8); ax.set_xticks(range(0, 24, 2))
    ax.tick_params(colors=th.MUTED, labelsize=8); ax.grid(color=th.GRID, lw=0.5); ax.legend(fontsize=7.5, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, loc="upper right")
    for sp in ax.spines.values():
        sp.set_color(th.AXIS)
    for nm, y, c, sym in (("realised", day["y"], th.INK_2, "circle"), (f"V3 (day MSE {mse(day['v3']):.3f})", day["v3"], th.OTHER_SERIES[1], "square"),
                          (f"V3 + module (day MSE {mse(day['mod']):.3f})", day["mod"], th.OTHER_SERIES[0], "triangle-up")):
        pl["data"].append({"type": "scatter", "mode": "lines+markers", "name": nm, "x": [int(a) for a in h], "y": [float(a) for a in y],
                           "line": {"color": c}, "marker": {"symbol": sym}, "xaxis": "x3", "yaxis": "y3"})
    pl["layout"]["xaxis3"] = {"domain": doms[2][0], "anchor": "y3", "title": {"text": "anchor hour (UTC)"}, "gridcolor": th.GRID}
    pl["layout"]["yaxis3"] = {"domain": doms[2][1], "anchor": "x3", "gridcolor": th.GRID, "zerolinecolor": th.AXIS, "title": {"text": "change in log RV, next hour"}}
    pl["layout"]["annotations"].append({"text": ttl, "x": 0.5, "y": 0.45, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 11, "color": th.INK_2}})
    fig.suptitle("T14 M8: which key group carries the M7 gain (dashed green = pre-registered threshold), and a real day", color=th.INK, fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_m8.png"), dpi=130, facecolor=th.PAPER); plt.close(fig)
    json.dump(pl, open(os.path.join(HERE, "fig_m8.json"), "w"))
    open(os.path.join(HERE, "fig_m8_day.txt"), "w").write(
        f"day {d} (UTC), module {FR['module']} walk-forward view, March refit\nrule: ascending mean A_LV0 among days with >= 20 valid rows, index (n-1)//2; n_days {day['n_days']}, rank {day['rank_index'] + 1}, activity {day['activity']:.3f}\n"
        f"day MSE: V3 {mse(day['v3']):.4f}, V3+module {mse(day['mod']):.4f}\nMarch 2026 monthly skill vs AR: {day['month_skill']}\n"
        f"activity by day: {day['activity_all']}\nhour  realised  V3  V3+module\n" + "\n".join(f"{int(a):4d} {b:+.3f} {c:+.3f} {e:+.3f}" for a, b, c, e in zip(h, day["y"], day["v3"], day["mod"])) + "\n")
    print(open(os.path.join(HERE, "fig_m8_day.txt")).read()[:900])


if __name__ == "__main__":
    main()
