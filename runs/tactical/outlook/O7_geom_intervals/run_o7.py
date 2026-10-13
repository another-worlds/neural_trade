"""O7 runner (SPEC.md). usage: run_o7.py [run|tables|plot]. Resumable through results.jsonl (key span, month, H, model)."""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o7_lib as P  # noqa: E402
O3 = P.O3; T = P.T
RES = os.path.join(HERE, "results.jsonl"); LOG = os.path.join(HERE, "run.log")


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nice():
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def free_gb():
    import psutil; return psutil.virtual_memory().available / 2 ** 30


def done_keys():
    if not os.path.exists(RES):
        return set()
    return {(r["span"], r["month"], r["H"], r["model"]) for r in map(json.loads, open(RES, encoding="utf-8"))}


def write(rows):
    with open(RES, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def rows_df():
    return pd.DataFrame([json.loads(x) for x in open(RES, encoding="utf-8")])


def run():
    nice(); t_all = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    F = P.load_all(); done = done_keys()
    assert pd.Timestamp(F["ta"].max()) <= T.CUTOFF
    log("frame", len(F["ta"]), pd.Timestamp(F["ta"][0]), pd.Timestamp(F["ta"][-1]), f"{time.time() - t_all:.0f}s", f"free {free_gb():.0f} GB")
    for span, months in P.SPANS.items():
        for mo in months:
            t0 = time.time(); f = P.prev_month(mo); new = []
            for H in P.HORIZONS:
                if all((span, mo, H, nm) in done for nm in P.MODELS):
                    continue
                rows = P.month_rows(F, H, f); tr, cal, te = rows
                m0 = P.month_bounds(f)[0]; n0 = P.month_bounds(mo)[0]
                assert len(tr) and len(cal) and len(te)
                assert pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=H) <= m0 and pd.Timestamp(F["ta"][cal].max()) + pd.Timedelta(minutes=H) <= n0
                for nm, d in P.score_fit_month(F, H, f, rows=rows).items():
                    d.update({"span": span, "month": mo, "fit_month": f}); new.append(d)
            write(new); log(span, mo, f"{time.time() - t0:.0f}s", f"free {free_gb():.0f} GB")
    log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime.json"), "w"))


def fmt(d, nd=2):
    return "n/a" if d is None else f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}] better {d['better']}/{d['n']} (min {d['min']:+.{nd}f}, max {d['max']:+.{nd}f})"


SPAN_LABEL = {"judge": "JUDGEMENT 2019-02..2020-12 (23 scored months)", "spent1": "DESCRIPTIVE 2021-02..2024-06", "spent2": "DESCRIPTIVE 2024-08..2025-07 (2025-07 ends at the cut 07-20)"}


def tables():
    df = rows_df(); out = []; Q = out.append
    Q(f"O7 tables. rows {len(df)}; interval score = Gneiting-Raftery alpha 0.1 in bps (lower is better); differences a - b negative = a better; 'better' counts months with a - b < 0")
    summ = {}
    for span, months in P.SPANS.items():
        d = df[df.span == span]; ms = sorted(d.month.unique())
        for H in P.HORIZONS:
            Q(f"\n===== {SPAN_LABEL[span]}  T={H}: pooled coverage | mean width bps | mean interval score bps  (months {len(ms)}) =====")
            for nm in P.MODELS:
                x = d[(d.H == H) & (d.model == nm)]
                Q(f"  {nm:4s} cover {x.n_in.sum() / x.n.sum():.4f} | width {x.width_bps.mean():8.2f} | iscore {x.iscore_bps.mean():8.2f}   (monthly cover min {x.cover.min():.3f} max {x.cover.max():.3f})")
            for a, b in (("L1", "R0"), ("L1", "HAR")):
                Q(f"  {a}-{b} interval score diff {fmt(P.paired(d, a, b, 'iscore_bps', ms, H))}")
                Q(f"  {a}-{b} width diff          {fmt(P.paired(d, a, b, 'width_bps', ms, H))}")
        # Holm
        for a, b, hs in (("L1", "R0", (15, 240)), ("L1", "HAR", (15, 60, 240))):
            prs = [P.paired(d, a, b, "iscore_bps", ms, H) for H in hs]; adj = P.holm([x["p"] for x in prs])
            Q(f"  [{span}] {a}-{b} interval-score diff, Holm over T={list(hs)}: " + "; ".join(f"T={H} p={x['p']:.4f} Holm {ad:.4f}" for H, x, ad in zip(hs, prs, adj)))
            summ[f"{span}|{a}-{b}"] = {str(H): {**x, "holm": ad} for H, x, ad in zip(hs, prs, adj)}
    jd = df[df.span == "judge"]; mo = P.J_SCORED
    pr = P.paired(jd, "L1", "R0", "iscore_bps", mo, 60); cv = P.pooled_cover(jd, "L1", mo, 60); c = P.criterion(pr, cv)
    Q("\n===== PRE-REGISTERED CRITERION (T = 60, L1 - R0, judgement span 2019-02..2020-12) =====")
    Q(f"interval-score diff {fmt(pr)}")
    Q(f"  mean<0 and 95% t-CI entirely below 0 and better >= {c['need_better']}: {c['c1_score_below_0_ci_below_0_months']} (mean {pr['mean']:+.3f}, upper {pr['hi']:+.3f}, better {pr['better']}/{pr['n']})")
    Q(f"  pooled L1 coverage {cv:.4f} in [0.88, 0.92]: {c['c2_pooled_cover_in_band']}")
    verdict = "pass" if c["pass"] else "fail"
    Q("VERDICT: " + verdict)
    Q("\n===== per-month, T=60 (judgement): cover L1/R0/HAR | iscore L1/R0/HAR | L1-R0 iscore =====")
    d = jd[jd.H == 60].set_index(["month", "model"])
    for m in mo:
        Q(f"  {m}  cover {d.loc[(m, 'L1'), 'cover']:.3f}/{d.loc[(m, 'R0'), 'cover']:.3f}/{d.loc[(m, 'HAR'), 'cover']:.3f} | iscore "
          f"{d.loc[(m, 'L1'), 'iscore_bps']:.1f}/{d.loc[(m, 'R0'), 'iscore_bps']:.1f}/{d.loc[(m, 'HAR'), 'iscore_bps']:.1f} | {d.loc[(m, 'L1'), 'iscore_bps'] - d.loc[(m, 'R0'), 'iscore_bps']:+.2f}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"verdict": verdict, "criterion": c, "paired_T60_L1_R0": pr, "summary": summ}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


def plot():
    """One figure per T (judgement span): monthly coverage, width and interval score of L1, R0, HAR; plus the plotly JSON of the same panels.
    Models take the non-horizon slots of the theme (horizon colours are not used); all solid; band 0.88-0.92 on the coverage panel."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from neural_trade.visualization import theme as TH
    df = rows_df(); df = df[df.span == "judge"]
    cols = {"L1": TH.OTHER_SERIES[0], "R0": TH.OTHER_SERIES[1], "HAR": TH.OTHER_SERIES[3]}; mk = {"L1": "o", "R0": "s", "HAR": "^"}
    panels = (("cover", "coverage (target 0.90, band 0.88-0.92)"), ("width_bps", "mean width, bps"), ("iscore_bps", "interval score, bps (lower is better)"))
    for H in P.HORIZONS:
        pr = P.paired(df, "L1", "R0", "iscore_bps", P.J_SCORED, H); ph = P.paired(df, "L1", "HAR", "iscore_bps", P.J_SCORED, H)
        title = (f"O7 T={H} min, judgement 2019-02..2020-12: 90% interval by month. Interval score L1-R0 {pr['mean']:+.2f} bps [{pr['lo']:+.2f},{pr['hi']:+.2f}] "
                 f"better {pr['better']}/{pr['n']}; L1-HAR {ph['mean']:+.2f} [{ph['lo']:+.2f},{ph['hi']:+.2f}] better {ph['better']}/{ph['n']}")
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), facecolor=TH.PAPER, sharex=True)
        pf = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[p[1] for p in panels])
        for i, (k, yl) in enumerate(panels):
            a_ = ax[i]; a_.set_facecolor(TH.SURFACE); a_.grid(color=TH.GRID, lw=.6)
            for sp in a_.spines.values():
                sp.set_color(TH.AXIS)
            a_.tick_params(colors=TH.MUTED)
            for nm in P.MODELS:
                d = df[(df.H == H) & (df.model == nm)].sort_values("month")
                a_.plot(d.month, d[k], color=cols[nm], marker=mk[nm], lw=1.6, ms=4.5, label=nm)
                pf.add_trace(go.Scatter(x=d.month, y=d[k], mode="lines+markers", name=nm, legendgroup=nm, showlegend=(i == 0), line=dict(color=cols[nm])), row=i + 1, col=1)
            if k == "cover":
                a_.axhline(0.90, color=TH.NEUTRAL, lw=.8, ls=":"); a_.axhspan(*P.COVER_BAND, color=TH.NEUTRAL, alpha=.12)
                pf.add_hrect(y0=P.COVER_BAND[0], y1=P.COVER_BAND[1], fillcolor=TH.NEUTRAL, opacity=0.15, line_width=0, row=i + 1, col=1)
            a_.set_ylabel(yl, color=TH.INK, fontsize=9)
        a_.tick_params(axis="x", rotation=60); ax[0].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK, ncol=3)
        fig.suptitle(title, color=TH.INK, fontsize=10); fig.tight_layout(); fig.savefig(f"{HERE}/cover_width_T{H}.png", dpi=110, facecolor=TH.PAPER); plt.close(fig)
        TH.apply(pf, title=f"O7 T={H} min, judgement span: 90% interval by month", subtitle=title.split(": 90% interval by month. ")[1], height=900)
        pf.write_json(f"{HERE}/cover_width_T{H}.json")
    print("plotted")


if __name__ == "__main__":
    {"run": run, "tables": tables, "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
