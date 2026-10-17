"""The M9 figure (SPEC_M9 Figure): (top) the estimated 24-hour profiles, weekday vs weekend, of the A target and of P(hit) (B) from the final fit of the profile module
(fitted means of the final ridge / logistic on the pool of label-known Binance anchors to 2026-09-30, realised means as dots, the base without the module dashed),
marked at the US open; (bottom) recovery bars: gain of M7 and the M9 views per target and look, harness CIs, R printed.
Run: D:/nt/nt_mm_env/python.exe make_figure_m9.py [profile_file]  -> fig_m9.png, fig_m9.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m9_lib as M9  # noqa: E402  (puts core on sys.path)
import evaluate as EV  # noqa: E402

th = EV.theme_colors()
PROFILE = sys.argv[1] if len(sys.argv) > 1 else "profile_M9h6_prof_str.json"
P = json.load(open(os.path.join(HERE, PROFILE))); R = json.load(open(os.path.join(HERE, "results_m9.json")))
BARS = ["M7_gbm", "M7c_gbm_resid", "M8_lin", "M9_prof", "M9_prof_str", "M9_prof_str_vp", "M9_gbm7", P["view"]]
BARS = list(dict.fromkeys(BARS))
OPEN_H = (13, 14)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hrs = np.arange(24); cwd, cwe = th.OTHER_SERIES[0], th.OTHER_SERIES[1]
    fig = plt.figure(figsize=(14, 10.5), facecolor=th.PAPER); gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    pl = {"data": [], "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 1000, "barmode": "group",
                                 "title": {"text": f"T14 M9: estimated 24-hour profile of the module {P['view']} (post hoc candidate) and the recovery of M7"}, "shapes": [], "annotations": []}}
    doms = [([0.0, 0.46], [0.56, 0.94]), ([0.54, 1.0], [0.56, 0.94]), ([0.0, 0.46], [0.0, 0.42]), ([0.54, 1.0], [0.0, 0.42])]
    ylab = {"A": "change in log RV, next hour", "B": "P(max abs move >= 50 bps in 60 min)"}
    for j, tg in enumerate(("A", "B")):
        ax = axs[j]; ax.set_facecolor(th.SURFACE); sfx = "" if j == 0 else str(j + 1)
        for grp, c in (("weekday", cwd), ("weekend", cwe)):
            f = np.array(P[f"{tg}_view"][grp], float); b = np.array(P[f"{tg}_base"][grp], float); r = np.array(P[f"{tg}_real"][grp], float)
            ax.plot(hrs, f, color=c, lw=1.8, label=f"{grp}: base + module (fitted)"); ax.plot(hrs, b, color=c, lw=1.0, ls="--", label=f"{grp}: base only (V3 / H3)")
            ax.plot(hrs, r, ls="none", marker="o", ms=3.5, color=c, alpha=0.7, label=f"{grp}: realised mean")
            for nm, y, dash, mode in ((f"{grp} base+module", f, None, "lines"), (f"{grp} base only", b, "dash", "lines"), (f"{grp} realised", r, None, "markers")):
                pl["data"].append({"type": "scatter", "mode": mode, "name": nm, "x": [int(h) for h in hrs], "y": [None if np.isnan(v) else float(v) for v in y],
                                   "line": {"color": c, **({"dash": dash} if dash else {})}, "marker": {"color": c, "size": 5}, "xaxis": "x" + sfx, "yaxis": "y" + sfx,
                                   "legendgroup": f"{tg}{grp}", "showlegend": bool(j == 0)})
        for h in OPEN_H:
            ax.axvline(h, color=th.NEUTRAL, lw=0.9, ls=":")
            pl["layout"]["shapes"].append({"type": "line", "xref": "x" + sfx, "yref": "paper", "x0": h, "x1": h, "y0": doms[j][1][0], "y1": doms[j][1][1], "line": {"color": th.NEUTRAL, "dash": "dot", "width": 1}})
        ymax = ax.get_ylim()[1]; ax.text(13.5, ymax, "US open\n13:30 / 14:30 UTC", ha="center", va="top", fontsize=7.5, color=th.INK_2)
        pl["layout"]["annotations"].append({"text": "US open (13:30 summer / 14:30 winter UTC)", "x": 13.5, "xref": "x" + sfx, "yref": "paper", "y": doms[j][1][1], "showarrow": False, "font": {"size": 10, "color": th.INK_2}})
        ttl = f"{tg}: " + ("next-hour change in log RV" if tg == "A" else "barrier-hit probability") + " by anchor hour (UTC)"
        ax.set_title(ttl, color=th.INK, fontsize=9.5); ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("anchor hour (UTC)", color=th.INK_2, fontsize=8); ax.set_ylabel(ylab[tg], color=th.INK_2, fontsize=8)
        ax.tick_params(colors=th.MUTED, labelsize=8); ax.grid(color=th.GRID, lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(th.AXIS)
        if j == 0:
            ax.legend(fontsize=6.5, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, ncol=2, loc="lower left")
        pl["layout"]["xaxis" + sfx] = {"domain": doms[j][0], "anchor": "y" + sfx, "gridcolor": th.GRID, "title": {"text": "anchor hour (UTC)"}}
        pl["layout"]["yaxis" + sfx] = {"domain": doms[j][1], "anchor": "x" + sfx, "gridcolor": th.GRID, "title": {"text": ylab[tg]}}
    col = {"look1": th.OTHER_SERIES[0], "look2": th.OTHER_SERIES[1]}
    for j, tg in enumerate(("A", "B")):
        ax = axs[2 + j]; ax.set_facecolor(th.SURFACE); sfx = str(3 + j); s = 100 if tg == "B" else 1
        x = np.arange(len(BARS)); w = 0.38
        for i, lk in enumerate(("look1", "look2")):
            v = np.array([[R["views"][b][tg][lk]["mean"], R["views"][b][tg][lk]["mean"] - R["views"][b][tg][lk]["lo"], R["views"][b][tg][lk]["hi"] - R["views"][b][tg][lk]["mean"]] for b in BARS]) * s
            ax.bar(x + (i - 0.5) * w, v[:, 0], w, yerr=[v[:, 1], v[:, 2]], color=col[lk], ecolor=th.INK_2, capsize=2, label=f"{lk} (harness 95% CI)")
            pl["data"].append({"type": "bar", "name": lk, "x": BARS, "y": [float(a) for a in v[:, 0]], "marker": {"color": col[lk]},
                               "error_y": {"type": "data", "symmetric": False, "array": [float(a) for a in v[:, 2]], "arrayminus": [float(a) for a in v[:, 1]]},
                               "xaxis": "x" + sfx, "yaxis": "y" + sfx, "legendgroup": lk, "showlegend": False})
        thr = 1.0 if tg == "B" else 0.005
        ax.axhline(0, color=th.NEUTRAL, lw=0.8); ax.axhline(thr, color=th.GOOD, lw=0.8, ls="--")
        top = ax.get_ylim()[1]
        for k, b in enumerate(BARS):
            ax.text(k, top, f"R={R['views'][b][tg]['R']:.2f}", ha="center", va="bottom", fontsize=7, color=th.INK_2)
        ax.set_xticks(x); ax.set_xticklabels([b.replace("_", "\n", 1) for b in BARS], fontsize=6.5, color=th.MUTED)
        t = "A: skill increment over V3" if tg == "A" else "B: relative Brier improvement over H3 (%)"
        ax.set_title(t + "  (R = pooled gain / M7_gbm's)", color=th.INK, fontsize=9, pad=14); ax.tick_params(colors=th.MUTED, labelsize=8); ax.grid(color=th.GRID, lw=0.5, axis="y")
        for sp in ax.spines.values():
            sp.set_color(th.AXIS)
        if j == 0:
            ax.legend(fontsize=7.5, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, loc="upper left")
        pl["layout"]["xaxis" + sfx] = {"domain": doms[2 + j][0], "anchor": "y" + sfx, "gridcolor": th.GRID}
        pl["layout"]["yaxis" + sfx] = {"domain": doms[2 + j][1], "anchor": "x" + sfx, "gridcolor": th.GRID, "title": {"text": t}}
        pl["layout"]["shapes"].append({"type": "line", "xref": "paper", "yref": "y" + sfx, "x0": doms[2 + j][0][0], "x1": doms[2 + j][0][1], "y0": thr, "y1": thr, "line": {"color": th.GOOD, "dash": "dash", "width": 1}})
        pl["layout"]["annotations"].append({"text": "R = " + ", ".join(f"{b}: {R['views'][b][tg]['R']:.2f}" for b in BARS), "x": sum(doms[2 + j][0]) / 2, "y": doms[2 + j][1][1] + 0.03, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 9, "color": th.INK_2}})
    fig.suptitle(f"T14 M9: profile of {P['view']} (in-sample fitted means on the pool to 2026-09-30, post hoc candidate) and recovery of M7 "
                 f"(dashed green = pre-registered threshold)", color=th.INK, fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_m9.png"), dpi=130, facecolor=th.PAPER); plt.close(fig)
    json.dump(pl, open(os.path.join(HERE, "fig_m9.json"), "w"))
    print("wrote fig_m9.png / fig_m9.json")


if __name__ == "__main__":
    main()
