"""T08 figures in the style of runs/tactical/report/tactical_report.html (theme.py); plotly JSON + PNG (headless Edge, as T07's make_figs).
Solid lines; no horizon colours for strategies (theme.OTHER_SERIES slots). Reads D:/nt/nt_t08_cache/{look,fwd}_scores.npz, selection.json, selection_full.json
and results.json. usage: python make_figs.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import t08_lib as L
import t07_lib as T
sys.path.insert(0, L.T07)
import make_figs as MF  # T07's helpers: theme loading, thin/save (PNG via Edge)

TH = MF.TH
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); MF.FIG = FIG
S_ = TH.OTHER_SERIES
MCOL = {"M1": S_[0], "M2": S_[1], "M3": S_[2], "M4": S_[3], "M5": S_[4], "M6": S_[6]}
FCOL = [S_[0], S_[1]]
BH_COL = TH.INK_2
save = MF.save


def load(tag):
    z = np.load(f"{L.CACHE}/{tag}_scores.npz"); return {k: z[k] for k in z.files}


def stage(tag):
    d = load(tag); hour_t = pd.DatetimeIndex(d["hour_t"]); Hr = T.hsum(d["r"]); C = L.percentiles(d["S"], L.month_labels(hour_t))
    return d, hour_t, Hr, C


# ------------------------------------------------------------------ 1. hit by decile
def fig_deciles(sel_dec, sel_cdec, fwd_dec, fwd_cdec):
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, vertical_spacing=0.14, horizontal_spacing=0.05,
                        subplot_titles=("Members, selection 2019-01..2024-06", "Members, forward 2025-11..2026-09 (frozen)",
                                        "Consensus rules, selection", "Consensus rules, forward"))
    x = list(range(1, 11))
    for col, (dm, dc) in enumerate(((sel_dec, sel_cdec), (fwd_dec, fwd_cdec)), start=1):
        n_all = np.mean([[r["n"] for r in rows] for rows in dm.values()])
        half = 1.96 * np.sqrt(0.25 / (n_all / L.HOLD))                                      # n_eff = N / 4 bars (overlapping 4-hour targets)
        for row in (1, 2):
            fig.add_trace(go.Scatter(x=x + x[::-1], y=[0.5 + half] * 10 + [0.5 - half] * 10, fill="toself", fillcolor=TH.rgba(TH.NEUTRAL, 0.2), line=dict(width=0),
                                     name="chance band (95%, n/4 effective)", showlegend=(col == 1 and row == 1), hoverinfo="skip"), row, col)
        for m, rows in dm.items():
            fig.add_trace(go.Scatter(x=x, y=[r["hit"] for r in rows], mode="lines+markers", line=dict(color=MCOL[m], width=2), marker=dict(size=6),
                                     name=m, legendgroup=m, showlegend=(col == 1), customdata=[r["n"] for r in rows],
                                     hovertemplate=m + " decile %{x}: hit %{y:.3f}, n %{customdata}<extra></extra>"), 1, col)
        for k, rows in dc.items():
            if k == "K3":
                continue
            fig.add_trace(go.Scatter(x=x, y=[r["hit"] for r in rows], mode="lines+markers", line=dict(color=TH.INK if k == "K1" else S_[5], width=2), marker=dict(size=6),
                                     name=f"{k} consensus", legendgroup=k, showlegend=(col == 1), customdata=[r["n"] for r in rows],
                                     hovertemplate=k + " decile %{x}: hit %{y:.3f}, n %{customdata}<extra></extra>"), 2, col)
        for row in (1, 2):
            fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1), row=row, col=col)
    fig.update_xaxes(title_text="confidence decile (10 = most confident)", dtick=1, row=2); fig.update_yaxes(title_text="hit rate (gross, 4-hour)", tickformat=".0%", col=1)
    TH.apply(fig, title="T08: does confidence sort the trades?", subtitle="Hit rate of the member's sign against the next 4-hour open-to-open return, by percentile-confidence decile "
             "(rank within the previous month). K3 (all six agree) is not drawn: it has about 150 decisions in all.", height=960)
    fig.update_layout(margin=dict(t=125, b=90), legend=dict(y=1.08))
    return fig


# ------------------------------------------------------------------ 2. equity of the finalists
def null_paths(ent, side, Hr, cost, h0, h1, draws=300, seed=0):
    from scipy import sparse
    nh = len(Hr); n = len(ent); cst = np.zeros(nh); h = cost * 0.5e-4
    np.add.at(cst, ent, h); np.add.at(cst, np.minimum(ent + L.HOLD, nh - 1), h)
    rows = (ent[:, None] + np.arange(L.HOLD)[None, :]).ravel()
    M = sparse.csr_matrix((Hr[rows], (rows, np.repeat(np.arange(n), L.HOLD))), shape=(nh, n))
    fl = np.where(np.random.default_rng(seed).random((draws, n)) < 0.5, 1.0, -1.0)
    rh = (M @ fl.T - cst[:, None])[h0:h1]
    return np.vstack([np.ones((1, draws)), np.cumprod(1 + rh, axis=0)]).T


def dd_of(eq):
    return 1.0 - eq / np.maximum.accumulate(eq)


def fig_equity(tag, names, start, stop, title):
    d, hour_t, Hr, C = stage(tag)
    h0 = int(np.searchsorted(hour_t.values, np.datetime64(start))); h1 = int(np.searchsorted(hour_t.values, np.datetime64(stop)))
    ht = hour_t[h0:h1]; x = [ht[0] - pd.Timedelta(hours=1)] + list(ht)
    bh = np.concatenate([[1.0], np.cumprod(1 + Hr[h0:h1])]); stb = T.equity_stats(Hr[h0:h1])
    fig = make_subplots(rows=2, cols=len(names), shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06, horizontal_spacing=0.06,
                        subplot_titles=[f"{n}" for n in names] + ["Drawdown"] * len(names))
    sub = []
    for j, nm in enumerate(names, start=1):
        ent, side = L.simulate(L.signal_for(nm, d["S"], C)); rh, _ = L.hourly_returns(ent, side, Hr, L.VERDICT_COST); rh = rh[h0:h1]
        eq = np.concatenate([[1.0], np.cumprod(1 + rh)]); st = T.equity_stats(rh)
        nul = null_paths(ent, side, Hr, L.VERDICT_COST, h0, h1); lo, med, hi = np.quantile(nul, [0.05, 0.5, 0.95], axis=0)
        fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, j)
        fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor=TH.rgba(TH.NEUTRAL, 0.22), line=dict(width=0), name="random-side null 5-95% (300 draws)", showlegend=(j == 1)), 1, j)
        fig.add_trace(go.Scatter(x=x, y=med, line=dict(color=TH.MUTED, width=1), name="null median", showlegend=(j == 1)), 1, j)
        fig.add_trace(go.Scatter(x=x, y=bh, line=dict(color=BH_COL, width=2), name="buy-and-hold (no cost)", showlegend=(j == 1)), 1, j)
        fig.add_trace(go.Scatter(x=x, y=eq, line=dict(color=FCOL[j - 1], width=2.5), name=f"{nm} (2 bps)", showlegend=True), 1, j)
        fig.add_trace(go.Scatter(x=x, y=-dd_of(bh), line=dict(color=BH_COL, width=1.5), showlegend=False), 2, j)
        fig.add_trace(go.Scatter(x=x, y=-dd_of(eq), line=dict(color=FCOL[j - 1], width=2), fill="tozeroy", fillcolor=TH.rgba(FCOL[j - 1], 0.2), showlegend=False), 2, j)
        k = (ent >= h0) & (ent < h1)
        sub.append(f"{nm}: return {st['total_return']:+.1%}, max DD {st['max_dd']:.1%}, {int(k.sum())} trades")
    fig.update_yaxes(title_text="equity (start 1.0)", col=1, row=1); fig.update_yaxes(tickformat=".0%", title_text="drawdown", col=1, row=2)
    TH.apply(fig, title=title, subtitle="  |  ".join(sub) + f"  |  buy-and-hold: {stb['total_return']:+.1%}, max DD {stb['max_dd']:.1%}", height=820)
    fig.update_layout(margin=dict(t=125), legend=dict(y=1.1))
    return fig


# ------------------------------------------------------------------ 3. trades on price, one forward month
def trade_counts(names):
    d, hour_t, Hr, C = stage("fwd"); fh0 = int(np.searchsorted(hour_t.values, np.datetime64("2025-11-01")))
    mon = L.month_labels(hour_t); cnt = {}
    for nm in names:
        ent, side = L.simulate(L.signal_for(nm, d["S"], C)); ent = ent[ent >= fh0]
        for m in mon[ent]:
            cnt[m] = cnt.get(m, 0) + 1
    return cnt


def fig_trades(names, month):
    d, hour_t, Hr, C = stage("fwd"); mon = L.month_labels(hour_t); idx = np.flatnonzero(mon == month)
    h0, h1 = idx[0], idx[-1] + 1; ht = hour_t[h0:h1]; hc = d["hc"][h0:h1]
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=names)
    sub = []
    for j, nm in enumerate(names, start=1):
        ent, side = L.simulate(L.signal_for(nm, d["S"], C)); k = (ent >= h0) & (ent < h1); e, s = ent[k], side[k]
        nb = L.trade_net_bps(e, s, Hr, L.VERDICT_COST)
        fig.add_trace(go.Scatter(x=ht, y=hc, line=dict(color=BH_COL, width=1.4), name="BTC/USDT hourly close", showlegend=(j == 1)), j, 1)
        for i in range(len(e)):
            seg = slice(e[i] - h0, min(e[i] + L.HOLD, h1 - 1) - h0 + 1)
            fig.add_trace(go.Scatter(x=ht[seg], y=hc[seg], line=dict(color=FCOL[j - 1], width=4), opacity=0.55, showlegend=False, hoverinfo="skip"), j, 1)
        ex = np.minimum(e + L.HOLD, len(d["hc"]) - 1)
        ent_px = d["ho"][e]; ex_px = d["ho"][ex]
        for sd, sym, lab in ((1, "triangle-up", "long entry"), (-1, "triangle-down", "short entry")):
            m = s == sd
            fig.add_trace(go.Scatter(x=hour_t[e[m]], y=ent_px[m], mode="markers", marker=dict(symbol=sym, size=11, color=FCOL[j - 1], line=dict(color=TH.INK, width=1)),
                                     name=lab, legendgroup=lab, showlegend=(j == 1), customdata=nb[m],
                                     hovertemplate="%{x}<br>" + lab + " at %{y:,.0f}<br>net %{customdata:+.1f} bps<extra></extra>"), j, 1)
        fig.add_trace(go.Scatter(x=hour_t[ex], y=ex_px, mode="markers", marker=dict(symbol="x", size=8, color=TH.INK), name="exit (4 h later)", legendgroup="exit",
                                 showlegend=(j == 1), hoverinfo="skip"), j, 1)
        sub.append(f"{nm}: {len(e)} trades, mean net {nb.mean() if len(nb) else float('nan'):+.1f} bps, {int((nb > 0).sum())} winners at 2 bps")
    fig.update_yaxes(title_text="BTC/USDT")
    TH.apply(fig, title=f"T08: the finalists' forward trades, {month}", subtitle="  |  ".join(sub) + "  |  month with the most trades of the two finalists (chosen by count, not by result)", height=820)
    fig.update_layout(margin=dict(t=125), legend=dict(y=1.08))
    return fig


def main():
    sel = json.load(open(f"{HERE}/selection.json")); names = [sel["finalist_A"], sel["finalist_B"]]
    sf = json.load(open(f"{HERE}/selection_full.json")); res = json.load(open(f"{HERE}/results.json"))
    save(fig_deciles(sf["deciles_member"], sf["deciles_consensus"], res["forward"]["deciles_member"], res["forward"]["deciles_consensus"]), "hit_by_decile", h=960)
    save(fig_equity("fwd", names, "2025-11-01", "2026-10-01", "T08: finalists against buy-and-hold and the null, forward 2025-11 .. 2026-09"), "equity_fwd", h=820)
    save(fig_equity("look", names, "2024-07-01", "2025-07-21", "T08: finalists against buy-and-hold and the null, Look 1 2024-07 .. 2025-07-20"), "equity_look", h=820)
    cnt = trade_counts(names); month = sorted(cnt, key=lambda m: (-cnt[m], m))[0]
    print("forward trades per month (both finalists):", dict(sorted(cnt.items())), "-> month", month)
    json.dump({"counts": dict(sorted(cnt.items())), "month": month}, open(f"{HERE}/figs/trades_month.json", "w"))
    save(fig_trades(names, month), "trades_fwd_month", h=820)


if __name__ == "__main__":
    main()
