"""Builds runs/tactical/deck/tactical_deck.html: ONE self-contained slide deck (16:9, Russian) of the tactical session 2026-10-06..13.
Reuses the report's visual system (theme.py colours and fonts, the report's plotly figures from build_report.py / fut_sections.py /
geom_figs.py, captured as figure objects and re-sized for slides) and its result files. Nothing is recomputed: every number is read
from committed result files (runs/tactical/webdash/out/*.json, hc2/, fwdtest/, tasks/T0x/...) or from the cut price files the report
builders already read (cutoff 2025-07-20 in common/cutoff.py; the forward file only through vol_windows.json).

usage: python runs/tactical/deck/build_deck.py     (CPU, about 1 minute)  -> tactical_deck.html
PNGs of every slide: python runs/tactical/deck/shoot_deck.py"""
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
import build_report as BR  # noqa: E402  (module level only loads data; build() is not called)
from rcommon import (T, TAC, PINK, GOLD, TAN, VIOLET, INK, INK2, MUTED, GRID, SURFACE, PAPER, AXIS, ru)  # noqa: E402

NEUT = T.NEUTRAL
CAP = {}


def _cap(fig, key, height=None, title=None, subtitle=None, lift=0):
    CAP[key] = fig
    return ""


BR.fig_html = _cap          # the report's section functions now hand their figure objects to CAP instead of rendering html
d_bars = BR.load_bars()
WS = BR.pick_windows(d_bars)
BR.sec6()
BR.sec4(WS)
FSF = lambda fig, sec, key, title, h, lift=0, subtitle=None: _cap(fig, key)  # noqa: E731
BR.FS.sec9(FSF, lambda *a: None)
BR.FS.sec10(FSF, lambda *a: None)


def jf(rel):
    return json.load(open(TAC / rel, encoding="utf-8"))


def pct(x, nd=1):
    return ru(x * 100, nd) + "%"


MON = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def mlab(m):
    return f"{MON[int(m[5:7]) - 1]} {m[2:4]}"


# ============================================================================================ figure plumbing
def fig_div(fig, key, w, h, legend=True, tfs=15, top=None, lift=0):
    """Theme + a fixed slide size; returns the plotly div (plotly itself is inlined once in the page)."""
    T.apply(fig, legend_top=legend)
    fig.update_layout(width=w, height=h, autosize=False, font=dict(size=15, color=INK2), paper_bgcolor=PAPER, plot_bgcolor=PAPER,
                      margin=dict(l=64, r=24, t=top if top is not None else ((74 + lift) if legend else 24), b=52), title_text=None,
                      legend=dict(font=dict(size=14), orientation="h", yanchor="bottom", y=1.0 + lift / max(h - (top if top is not None else ((74 + lift) if legend else 24)) - 52, 100), xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)"))
    for a in fig.layout.annotations or ():
        if a.font is not None and a.font.size is not None and a.font.size <= 12:
            a.update(font=dict(color=INK2, size=tfs))
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"f_{key}",
                       config={"displaylogo": False, "displayModeBar": False, "responsive": False, "scrollZoom": False})


# ============================================================================================ own figures
decay = BR.decay
regimes = BR.regimes


def ymean(key, y0, y1):
    return BR.year_mean(decay, key, y0, y1)[0]


def fig_decay_two():
    """Monthly AUC (close and next-open entry, chance band, regime steps) over the top-10% hit rate."""
    mths = [r["month"] for r in decay]
    xs = [f"{m}-01" for m in mths]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.66, 0.34],
                        subplot_titles=("AUC направления по месяцам (обучение 60 дней, стрелка на 15 минут)", "доля верных в топ-10% уверенных сигналов (вход по следующему открытию)"))
    band = [1.96 / np.sqrt(3.0 * max(BR.raw_n.get(m, 400) * 10 * 10 / 15.0, 100)) for m in mths]
    fig.add_trace(go.Scatter(x=xs, y=[0.5 + b for b in band], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[0.5 - b for b in band], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(137,135,129,0.25)",
                             name="шум: 95% интервал при чистой случайности", customdata=band, hovertemplate="шум ±%{customdata:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_close"] for r in decay], mode="lines+markers", line=dict(color=PINK, width=1.5), marker=dict(size=5),
                             name="вход по закрытию (H51)", hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra>закрытие</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_nextopen"] for r in decay], mode="lines+markers", line=dict(color=VIOLET, width=2.5), marker=dict(size=6, symbol="square"),
                             name="вход по следующему открытию (H59, честный)", hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra>открытие</extra>"), row=1, col=1)
    for i, sg in enumerate(regimes):
        x0 = f"{sg['from']}-01"
        x1 = (pd_period(sg["to"]) + 1).start_time.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=[x0, x1], y=[sg["mean"]] * 2, mode="lines", line=dict(color=GOLD, width=5), name="ступень (среднее режима, H72)", legendgroup="reg",
                                 showlegend=(i == 0), hovertemplate=f"{sg['from']} .. {sg['to']}: среднее AUC {sg['mean']:.4f}, {sg['n']} мес.<extra></extra>"), row=1, col=1)
        fig.add_annotation(x=x0, y=sg["mean"], text=ru(sg["mean"], 3), showarrow=False, yshift=16, xanchor="left", font=dict(size=14, color=GOLD), bgcolor="rgba(18,18,17,0.8)", row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["hit10_nextopen"] for r in decay], mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=5, symbol="square"),
                             name="доля верных, топ-10%", hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra>топ-10%</extra>"), row=2, col=1)
    fig.add_hline(y=0.6, line=dict(color=T.CRITICAL, width=2, dash="dash"), annotation_text="цель владельца 60%", annotation_position="top right", row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_yaxes(title_text="hit", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return fig


def pd_period(s):
    import pandas as pd
    return pd.Period(s, "M")


def fig_bounce():
    yrs = list(range(2017, 2026))
    c = [ymean("auc_close", y, y) for y in yrs]
    o = [ymean("auc_nextopen", y, y) for y in yrs]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.09,
                        subplot_titles=("AUC направления по годам: две цены входа", "разница (закрытие − открытие): отскок bid/ask"))
    fig.add_trace(go.Bar(x=yrs, y=c, name="вход по закрытию сигнального бара (H51)", marker=dict(color=PINK, line=dict(color=SURFACE, width=1.5)),
                         hovertemplate="%{x}: AUC %{y:.3f}<extra>закрытие</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=yrs, y=o, name="вход по следующему открытию (H59)", marker=dict(color=VIOLET, line=dict(color=SURFACE, width=1.5), pattern_shape="/"),
                         hovertemplate="%{x}: AUC %{y:.3f}<extra>открытие</extra>"), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1.5, dash="dash"), row=1, col=1)
    gap = [a - b for a, b in zip(c, o)]
    fig.add_trace(go.Bar(x=yrs, y=gap, name="разница", showlegend=False, marker=dict(color=GOLD, line=dict(color=SURFACE, width=1.5)),
                         text=[ru(g, 3) for g in gap], textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.3f}<extra>разница</extra>"), row=1, col=2)
    fig.update_yaxes(range=[0.48, 0.64], title_text="AUC", row=1, col=1)
    fig.update_yaxes(range=[0, max(gap) * 1.25], title_text="Δ AUC", row=1, col=2)
    fig.update_xaxes(dtick=1, row=1, col=1)
    fig.update_xaxes(dtick=1, row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.2)
    return fig


def fig_vol_vs_dir():
    mths = [r["month"] for r in decay]
    xs = [f"{m}-01" for m in mths]
    dv = {r["month"]: r for r in jf("lab/decay_vol.json")}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_nextopen"] for r in decay], mode="lines", line=dict(color=VIOLET, width=1), opacity=0.5, name="AUC направления, по месяцам (честный вход)",
                             hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra>направление</extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[r["har_spearman"] for r in decay], mode="lines", line=dict(color=TAN, width=1.5), opacity=0.6, name="Спирмен HAR-прогноза волатильности, по месяцам",
                             hovertemplate="%{x|%Y-%m}: Спирмен %{y:.3f}<extra>HAR</extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=xs, y=[dv[m]["rv60"] if m in dv else None for m in mths], mode="lines", line=dict(color=NEUT, width=1.2), name="наивная база: RV за 60 мин (Спирмен)",
                             hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra>RV60</extra>"), secondary_y=True)
    for y in range(2017, 2026):
        a, h = ymean("auc_nextopen", y, y), ymean("har_spearman", y, y)
        fig.add_trace(go.Scatter(x=[f"{y}-01-01", f"{y + 1}-01-01"], y=[a, a], mode="lines", line=dict(color=VIOLET, width=5), name="среднее за год: направление", legendgroup="a",
                                 showlegend=(y == 2017), hovertemplate=f"{y}: AUC {a:.3f}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=[f"{y}-01-01", f"{y + 1}-01-01"], y=[h, h], mode="lines", line=dict(color=TAN, width=5), name="среднее за год: волатильность", legendgroup="h",
                                 showlegend=(y == 2017), hovertemplate=f"{y}: Спирмен {h:.3f}<extra></extra>"), secondary_y=True)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"), secondary_y=False)
    fig.update_yaxes(title_text="AUC направления", range=[0.40, 0.66], secondary_y=False, title_font=dict(color=VIOLET), tickfont=dict(color=VIOLET))
    fig.update_yaxes(title_text="Спирмен прогноза волатильности", range=[-0.10, 0.66], secondary_y=True, title_font=dict(color=TAN), tickfont=dict(color=TAN), showgrid=False)
    fig.update_layout(hovermode="x unified")
    return fig


def fig_verdicts():
    vs = list(BR.verdicts)
    t5 = jf("tasks/T05_interval_models/resultsA.json")
    m0m1 = t5["M0_vs_M1"]
    vs.append({"h": "H87", "name": "HAR лучше EWMA на 2017-18 (интервалы)", "unit": "разница интервальной оценки, б.п. (16 месяцев)", "effect": -m0m1["mean_diff_bps"], "threshold": 0.0, "ratio": 1.0, "verdict": "прошла"})
    vm = t5["verdict"]["M4_vs_M0"]
    vs.append({"h": "H87", "name": "Модель лучше HAR (квантильная регрессия)", "unit": "разница интервальной оценки, б.п. (ниже - лучше)", "effect": -vm["mean_diff_bps"], "threshold": 0.0, "ratio": -1.0, "verdict": "не прошла"})
    ok = [v["verdict"].startswith("прошла") for v in vs]
    order = sorted(range(len(vs)), key=lambda i: int(vs[i]["h"][1:]))
    fig = go.Figure()
    names = [f"{vs[i]['h']} · {vs[i]['name']}" for i in order]
    for k, i in enumerate(order):
        col, sym = (T.GOOD, "diamond") if ok[i] else (T.CRITICAL, "x")
        r = max(min(vs[i]["ratio"], 2.0), -2.0)
        fig.add_trace(go.Bar(y=[names[k]], x=[r], orientation="h", marker=dict(color=col, opacity=0.55, line=dict(color=col, width=1)), showlegend=False,
                             hovertemplate=f"{vs[i]['unit']}<br>эффект {vs[i]['effect']:+.4g}, порог {vs[i]['threshold']:.4g}<br>вердикт: {vs[i]['verdict']}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[names[k]], x=[r], mode="markers", marker=dict(symbol=sym, size=12, color=col, line=dict(color=INK, width=1)), showlegend=False, hoverinfo="skip"))
    for nm, col, sym in (("порог пройден", T.GOOD, "diamond"), ("не пройден", T.CRITICAL, "x")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(symbol=sym, size=12, color=col)))
    fig.add_vline(x=1, line=dict(color=GOLD, width=2, dash="dash"))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12.5))
    fig.update_xaxes(title_text="эффект / порог, записанный до результата (порог 0: показан знак ±1; обрезано до ±2)", range=[-2.1, 2.1])
    fig.update_layout(bargap=0.3)
    return fig, sum(ok), len(vs), [vs[i]["h"] for i in order if ok[i]]


def fig_tried():
    """What was tried for direction: regression vs networks (AUC with CI) and the geometry forest (gain over the base)."""
    ms = BR.models
    fig = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.54], horizontal_spacing=0.34,
                        subplot_titles=("AUC направления: регрессия против сетей (H35, H65)", "прирост AUC от геометрии и синтетики (H24-H49, H80), 95% ДИ"))
    for i, m in enumerate(ms):
        net = m["step"] not in ("L0", "L1", "L3", "linear_1y")
        col, sym = (PINK, "square") if net else (VIOLET, "circle")
        fig.add_trace(go.Scatter(x=[m["auc"]], y=[m["model"]], mode="markers", showlegend=False, marker=dict(symbol=sym, size=11, color=col, line=dict(color=INK, width=1)),
                                 error_x=dict(type="data", symmetric=False, array=[m["hi"] - m["auc"]], arrayminus=[m["auc"] - m["lo"]], color=col, thickness=2, width=5),
                                 hovertemplate=f"{m['model']}: {m['auc']:.4f} [{m['lo']:.4f}; {m['hi']:.4f}]<extra></extra>"), row=1, col=1)
    for nm, col, sym in (("регрессия (круг)", VIOLET, "circle"), ("сеть (квадрат)", PINK, "square")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(symbol=sym, size=11, color=col)), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="AUC (блоки 11,5 дня; 1-3 года)", row=1, col=1)
    rows = list(BR.geom)
    ys = [f"{r['h']} · {r['name']}" for r in rows]
    for i, r in enumerate(rows):
        if r["lo"] is None:
            fig.add_trace(go.Scatter(x=[r["effect"]], y=[ys[i]], mode="markers", showlegend=False, marker=dict(symbol="circle-open", size=11, color=INK2, line=dict(color=INK2, width=2)),
                                     hovertemplate=f"{ys[i]}<br>Δ AUC {r['effect']:.4f} (интервал не сохранён)<extra></extra>"), row=1, col=2)
        else:
            col, sym = (T.GOOD, "diamond") if r["lo"] > 0 else ((T.CRITICAL, "x") if r["hi"] < 0 else (NEUT, "circle"))
            fig.add_trace(go.Scatter(x=[r["effect"]], y=[ys[i]], mode="markers", showlegend=False, marker=dict(symbol=sym, size=11, color=col, line=dict(color=INK, width=1)),
                                     error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["effect"]], arrayminus=[r["effect"] - r["lo"]], color=col, thickness=2, width=5),
                                     hovertemplate=f"{ys[i]}<br>Δ AUC {r['effect']:.4f} [{r['lo']:.4f}; {r['hi']:.4f}]<extra></extra>"), row=1, col=2)
    for nm, col, sym in (("ДИ покрывает 0", NEUT, "circle"), ("значимо хуже базы", T.CRITICAL, "x"), ("ДИ не сохранён", INK2, "circle-open")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(symbol=sym, size=11, color=col)), row=1, col=2)
    fig.add_vline(x=0, line=dict(color=NEUT, width=1.5, dash="dash"), row=1, col=2)
    fig.add_vline(x=0.005, line=dict(color=GOLD, width=1.5, dash="dot"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), row=1, col=2)
    fig.update_xaxes(title_text="Δ AUC к базе (0,005 - минимальный эффект)", row=1, col=2)
    return fig


def fig_t03_months():
    t3 = jf("tasks/T03_vol_intervals/results.json")
    import pandas as pd
    rw = pd.DataFrame(t3["rows"])
    rw = rw[(rw.ver == "cal") & (rw.H == 60) & (rw.f.isin(["HAR", "trail"]))]
    a = rw[rw.f == "HAR"].set_index("month").iscore_bps
    b = rw[rw.f == "trail"].set_index("month").iscore_bps
    diff = (a - b).sort_index()
    c60 = t3["summary"]["criterion"]["60"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[mlab(m) for m in diff.index], y=diff.values, marker=dict(color=[VIOLET if v < 0 else PINK for v in diff.values], line=dict(color=SURFACE, width=1.5)),
                         text=[f"{v:.0f}" for v in diff.values], textposition="outside", cliponaxis=False, showlegend=False,
                         hovertemplate="%{x}: HAR − скользящая RV = %{y:.1f} б.п.<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.add_hline(y=c60["mean_diff_iscore_bps"], line=dict(color=GOLD, width=2, dash="dash"))
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=1.07, showarrow=False, font=dict(size=15, color=INK2),
                       text=f"T = 60, по месяцам; золотая линия - среднее {ru(c60['mean_diff_iscore_bps'], 1)}")
    fig.update_yaxes(title_text="интервальная оценка, б.п. (ниже - лучше HAR)")
    return fig, int((diff < 0).sum()), len(diff)


def fig_t05():
    t5 = jf("tasks/T05_interval_models/resultsA.json")
    import pandas as pd
    rw = pd.DataFrame(t5["rows"])
    p = rw.pivot(index="month", columns="model", values="iscore_bps")
    diff = (p["M1"] - p["M0"])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.09,
                        subplot_titles=("EWMA 0,94 − HAR по месяцам, б.п. (выше нуля - HAR лучше)", "разница интервальной оценки и 95% ДИ (16 месяцев)"))
    fig.add_trace(go.Bar(x=[mlab(m) for m in diff.index], y=diff.values, showlegend=False, marker=dict(color=[VIOLET if v > 0 else PINK for v in diff.values], line=dict(color=SURFACE, width=1.5)),
                         hovertemplate="%{x}: %{y:.1f} б.п.<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    m01 = t5["M0_vs_M1"]
    v = t5["verdict"]
    items = [("EWMA − HAR", -m01["mean_diff_bps"], -m01["ci"][1], -m01["ci"][0], T.GOOD, "diamond"),
             ("квантильная регрессия − HAR", v["M4_vs_M0"]["mean_diff_bps"], v["M4_vs_M0"]["ci"][0], v["M4_vs_M0"]["ci"][1], NEUT, "circle"),
             ("квантильная регрессия − EWMA", v["M4_vs_M1"]["mean_diff_bps"], v["M4_vs_M1"]["ci"][0], v["M4_vs_M1"]["ci"][1], NEUT, "circle")]
    for nm, e, lo, hi, col, sym in items:
        fig.add_trace(go.Scatter(x=[e], y=[nm], mode="markers", showlegend=False, marker=dict(symbol=sym, size=13, color=col, line=dict(color=INK, width=1)),
                                 error_x=dict(type="data", symmetric=False, array=[hi - e], arrayminus=[e - lo], color=col, thickness=2.5, width=7),
                                 hovertemplate=f"{nm}: {e:+.1f} [{lo:+.1f}; {hi:+.1f}] б.п.<extra></extra>"), row=1, col=2)
    fig.add_vline(x=0, line=dict(color=NEUT, width=1.5, dash="dash"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="б.п. (положительно: первая модель хуже)", row=1, col=2)
    fig.update_yaxes(title_text="б.п.", row=1, col=1)
    return fig, int((diff > 0).sum()), len(diff)


def fig_spark():
    yrs = list(range(2017, 2026))
    a = [ymean("auc_nextopen", y, y) for y in yrs]
    h = [ymean("har_spearman", y, y) for y in yrs]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=yrs, y=a, mode="lines+markers+text", line=dict(color=VIOLET, width=5), marker=dict(size=11, symbol="square"), name="направление: AUC",
                             text=[ru(v, 3) if y in (2017, 2025) else "" for v, y in zip(a, yrs)], textposition="top center", textfont=dict(size=18, color=VIOLET)), secondary_y=False)
    fig.add_trace(go.Scatter(x=yrs, y=h, mode="lines+markers+text", line=dict(color=TAN, width=5), marker=dict(size=11, symbol="circle"), name="волатильность: Спирмен HAR",
                             text=[ru(v, 2) if y in (2018, 2025) else "" for v, y in zip(h, yrs)], textposition="top center", textfont=dict(size=18, color=TAN)), secondary_y=True)
    fig.update_yaxes(visible=False, range=[0.44, 0.60], secondary_y=False)
    fig.update_yaxes(visible=False, range=[0.15, 0.50], secondary_y=True)
    fig.update_xaxes(dtick=1, tickfont=dict(size=16))
    return fig


# ============================================================================================ numbers
fwd = jf("fwdtest/fwdtest_result.json")["arms"]["frozen"]
L2 = fwd["levels"]["2"]
lb = jf("hc2/lockbox_result.json")
t1 = jf("tasks/T01_vol_forward/results.json")
S3 = t1["summary"]["F"]["A"]["S3"]
t3 = jf("tasks/T03_vol_intervals/results.json")
c60 = t3["summary"]["criterion"]["60"]
t4 = jf("tasks/T04_vol_calendar/resultsA.json")
t2 = jf("tasks/T02_vol_exposure/results.json")
t5 = jf("tasks/T05_interval_models/resultsA.json")
auc_c19, auc_c25 = ymean("auc_close", 2019, 2019), ymean("auc_close", 2025, 2025)
auc_o19, auc_o25 = ymean("auc_nextopen", 2018, 2020), ymean("auc_nextopen", 2025, 2025)
har17, har25 = ymean("har_spearman", 2017, 2017), ymean("har_spearman", 2025, 2025)
har18 = ymean("har_spearman", 2018, 2018)

# ============================================================================================ slides
SL = []   # dicts: kicker, h, lines, vis (list of html), ev, mode ('side' | 'wide' | 'title'), stat


def slide(kick, h, lines, vis, ev, mode="side", stat=None, vis_class=""):
    SL.append(dict(kick=kick, h=h, lines=lines, vis=vis, ev=ev, mode=mode, stat=stat, vc=vis_class))


# 1 title
slide("Тактическая сессия · 2026-10-06 … 13", "Направление цены исчерпано, волатильность работает", [], [fig_div(fig_spark(), "spark", 1500, 380, legend=True, top=60)],
      "H5-H87 · данные до 2025-07-20 (подбор) и Binance 2025-10…2026-10 (проверка вперёд) · tactical_report.html", mode="title")

# 2 goal and outcome
vfig, npass, nall, passed_h = fig_verdicts()
slide("Цель и итог · H67-H87", f"Цель владельца не достигнута; из {nall} порогов прошло {npass}, и все - про волатильность",
      ["Цель: <b>hit &gt; 60%</b> при просадке <b>&lt; 5%</b>: ни одна стратегия.",
       "Честная часовая: hit 57%, просадка 9-10%; &lt; 5% - только плечом ~0,4x.",
       "Зелёное: " + ", ".join(sorted(set(passed_h), key=lambda s: int(s[1:]))) + "."],
      [fig_div(vfig, "verdicts", 1040, 720, legend=True)],
      "Доска вердиктов: эффект / порог, записанный до результата (LOG.md H67-H87; H81 - сейф, H82 - вперёд)",
      stat=("0", "стратегий в цели<br>hit &gt; 60% и DD &lt; 5%"))

# 3 decay
rg = regimes
slide("Затухание направления · H51 / H59 / H72", f"AUC направления упал ступенями: {ru(rg[1]['mean'], 2)} → {ru(rg[4]['mean'], 3)} → {ru(rg[5]['mean'], 3)}",
      [f"Годовое среднее при входе по закрытию: {ru(auc_c19, 2)} (2019) → {ru(auc_c25, 2)} (2025).",
       f"Режимы H72: 0,574 (2018-03) → 0,545 (2021-03) → 0,521 (с 2024-09).",
       "Топ-10% сигналов: hit около 60% в 2018-20, около 53% в 2025."],
      [fig_div(fig_decay_two(), "decay", 1060, 730, tfs=14, lift=30)],
      "H51 (вход по закрытию), H59 (по открытию), H72: 5 разрывов против тренда, ΔBIC +41,8, sup-Chow p = 0,004; месячный walk-forward 2017-04…2025-07; серая полоса - шум 95%")

# 4 bounce
slide("Отскок bid/ask · H59 / H60", "Половина старого перевеса была отскоком между bid и ask при входе по закрытию",
      [f"AUC 2018-20: {ru(ymean('auc_close', 2018, 2020), 3)} по закрытию против {ru(ymean('auc_nextopen', 2018, 2020), 3)} по следующему открытию.",
       "H48 (барьер на минутах): +0,54 → +0,30 б.п. при честном входе; интервал пересекает 0 - не проходит.",
       "Недавние годы почти не меняются: перевес там и так мал."],
      [fig_div(fig_bounce(), "bounce", 1480, 640, tfs=15, lift=30)],
      "H59: тот же walk-forward, вход по открытию следующего бара; H60: H48 пересчитан, meta +0,30 б.п. (null95 +0,25), месячный интервал [−0,20; +0,79]", mode="wide")

# 5 tried
slide("Всё, что пробовали для направления · H23-H49, H80", "Сети, геометрия, JEPA, физика и синтетика не обогнали простую регрессию",
      ["Регрессия на 7 учебниковых индикаторах: AUC 0,570; GRU, PatchTST, 3 года данных: 0,567-0,572.",
       "Геометрия (14 вариантов): ни один интервал целиком выше нуля; N1, N2, N6 значимо хуже.",
       "JEPA (H55), физика (H75-77), синтетика (H80): направление не растёт; польза - только на оси дисперсии."],
      [fig_div(fig_tried(), "tried", 1480, 650, tfs=14, lift=30)],
      "H35 / H65 (модели), H24-H49 (геометрия, к сети или регрессии), H80 (синтетические ряды); 24 лабораторных среза, dev-фолды, одно зерно", mode="wide")

# 6 geometry on real BTC
slide("Геометрия на реальном BTC · H24 / H43 / H47", "Геометрия на реальном BTC красива, но направления не добавляет",
      ["Окно BTC/USDT 2024-12-05, 60 минут: каналы EMA20, RSI14, MACD, Stoch и их наклоны.",
       "В сети геометрия улучшила калибровку, но не AUC: +0,014 [−0,002; +0,030].",
       "В регрессии - ни одного прироста (H45, H47)."],
      [fig_div(CAP["h24_2"], "h24", 1080, 730, legend=True, top=100, tfs=13)],
      "H24-H47; окно 2024-12-05 22:00-22:59 UTC (минутные свечи, фиолетовый - рост, жёлтый - падение); справа заштрихована зона сжатия")

# 7 forward test
slide("Проверка вперёд · H82", f"H62 на невиданных данных Binance провалился: hit {pct(L2['stats']['hit'])}, итог {ru(L2['stats']['cum_return_pct'], 1, True)}%",
      [f"Замороженный кандидат, 2025-11…2026-09, 437 сделок, 2 б.п.; на подборе было hit 59,3%.",
       f"Просадка {ru(L2['stats']['max_drawdown_pct'], 1)}% при пределе 15%; {pct(L2['null']['share_null_cum_ge_actual'], 0)} случайных сторон не хуже.",
       f"Месяцев с hit &gt; 60%: {L2['months_hit_gt_60']} из {L2['n_months']}."],
      [fig_div(CAP["h82_months"], "h82m", 700, 650, tfs=14, lift=34), fig_div(CAP["h82_equity"], "h82e", 780, 650, tfs=14, top=130)],
      "H82: Binance spot BTCUSDT 1m 2025-10-01…2026-10-12; критерии пройдены 0 из " + str(len(fwd["criteria"])) + "; серая полоса - случайная сторона (оценка 5-95%)",
      mode="wide", vis_class="two")

# 8 hill-climb
W_, B_ = lb["winner"], lb["baseline"]
slide("Hill-climb v2 · H81", f"Победитель 1 944 конфигураций переобучился: Calmar {ru(W_['dev']['calmar'], 1)} → {ru(W_['lockbox']['calmar'], 2)} на сейфе",
      [f"На сейфе (12 месяцев 2019-24): {ru(W_['lockbox']['mean_net_bps'], 1)} б.п. за сделку, просадка {ru(W_['lockbox']['max_dd_pct'], 0)}%.",
       "Базовая H62 «хороша» на сейфе, но сама выбрана на этих месяцах: это не доказательство.",
       "Урок: 1 944 конфигурации на ~5 годах - подгонка шума."],
      [fig_div(CAP["h81_dev_lock"], "h81", 1480, 650, tfs=15, lift=30)],
      "H81: SPEC 3d561ba, сейф - 12 месяцев, отложенных до подбора; штриховка - сейф; зелёные линии - цель владельца", mode="wide")

# 9 the turn
slide("Поворот · H53", f"Предсказуемость волатильности выросла {ru(har18, 2)} → {ru(har25, 2)}, пока направление угасало",
      [f"Спирмен HAR-прогноза: {ru(har18, 2)} (2018) → {ru(har25, 2)} (2025); в 2017 было {ru(har17, 2)}.",
       f"AUC направления (вход по открытию): {ru(auc_o19, 3)} (2018-20) → {ru(auc_o25, 3)} (2025).",
       "Размер движения предсказуем, его знак - нет (H54)."],
      [fig_div(fig_vol_vs_dir(), "turn", 1480, 660, tfs=15, lift=30)],
      "H53 (decay_vol.json), H51/H59; месячный walk-forward 2017-04…2025-07, обучение на 60 днях; HAR-RV + время суток", mode="wide")

# 10 T01
slide("T01 · H83 / H86", "Изменение волатильности на час предсказуемо вперёд: 12 из 12 месяцев",
      [f"Сверх HAR-уровней: +0,03; повторилось на 2017-18 (+0,019 [+0,010; +0,027], 13 из 16).",
       f"Скилл к AR {ru(S3['skill_ar'], 3, True)}; на взгляде H72 было +0,163: на 13% меньше, не обвал.",
       "Это измерение предсказуемости, не торговый перевес."],
      [fig_div(CAP["t01_skill"], "t01", 1480, 650, tfs=15, lift=34)],
      "T01 (SPEC 34b40f4, до результатов), Binance 2025-10…2026-09, модели заморожены на 365 днях до 2025-07-20; qa-deep PASS; слабая база AR - оговорка", mode="wide")

# 11 T03
fig_m, nbest, nm_ = fig_t03_months()
slide("T03 · H85", f"Интервалы по HAR уже при покрытии 90%: лучше скользящей RV в {nbest} из {nm_} месяцев",
      [f"Интервальная оценка T = 60: {ru(c60['mean_diff_iscore_bps'], 1)} б.п. [{ru(c60['ci'][0], 1)}; {ru(c60['ci'][1], 1)}], покрытие {ru(c60['pooled_cover'], 3)}.",
       "Прошло на T = 15, 60 и 240 (Холм); интервал около 7% уже.",
       "Справа - реальный день BTC: красные кресты - выход за интервал."],
      [fig_div(CAP["t03_day1"], "t03d", 920, 660, tfs=13, lift=70), fig_div(fig_m, "t03m", 540, 660, legend=False, top=60)],
      "T03 (SPEC ada88f5), Binance 2025-10…2026-09; день выбран по квантилю волатильности, не по качеству; QA PASS: −9,48 [−11,25; −7,70]", mode="wide", vis_class="two2")

# 12 T05
fig5, nh, nn = fig_t05()
slide("T05 · H87", f"HAR сильнее EWMA на нетронутых 2017-18: {nh} из {nn} месяцев; лучше HAR ничего нет",
      ["Квантильная регрессия, полу- и скачок-вариации, адаптивный ACI: не обгоняют HAR.",
       f"Выбранная на подборе модель на 2017-18: +{ru(t5['verdict']['M4_vs_M0']['mean_diff_bps'], 1)} б.п. [{ru(t5['verdict']['M4_vs_M0']['ci'][0], 1)}; +{ru(t5['verdict']['M4_vs_M0']['ci'][1], 1)}] - хуже или равна.",
       "ACI держит покрытие точно (sd 0,0014), цена - около 3,5% оценки."],
      [fig_div(fig5, "t05", 1480, 650, tfs=15, top=60)],
      "T05 (SPEC 7a02d55), look 2017-09…2018-12 (16 месяцев), T = 60, 90%-интервалы; заморожено для 2026-10… (sha256 e9c38d61…)", mode="wide")

# 13 what does not work
look2 = t2["look2"]["2"]
fig_t04 = CAP["t04_months"]
_an = list(fig_t04.layout.annotations)
_an[0].text = "S3C − S3 (календарь): " + ru(t4["criterion_A"]["mean_diff_skill_ar"], 3, True) + ", 7 из 16"
_an[1].text = "S3 сверх HAR-уровней: +0,019, 13 из 16"
fig_t04.layout.annotations = tuple(_an[:2])
slide("Что не работает · H84 / H86", "Вол-тайминг экспозиции и 15-минутный календарь не прошли",
      [f"H84: просадка {ru(look2['strategy']['max_dd'] * 100, 1)}% против предела 40,6%; DD &lt; 5% - только 7-15% капитала в BTC.",
       f"H86: календарь на 2017-18: S3C − S3 {ru(t4['criterion_A']['mean_diff_skill_ar'], 3, True)}, 7 из 16 - артефакт форварда.",
       "Прогноз волатильности говорит о размере движения, не о моменте просадки."],
      [fig_div(CAP["t02_equity"], "t02", 790, 660, tfs=14, lift=70), fig_div(fig_t04, "t04", 690, 660, legend=False, tfs=14, top=60)],
      "T02 (SPEC 34b40f4): выбор на 2019-24, форвардный год, 2 б.п., серое - часы вне BTC; T04-A (SPEC 44441a5): 2017-09…2018-12, серая полоса - 95% ДИ среднего", mode="wide", vis_class="two3")

# 14 next
slide("Дальше", "Волатильность идёт в MVP; остальное - вторая сессия и замороженные тесты", [], [], "TACTICAL.md, LOG.md (For the MVP lead), OUTLOOK_positive.md; даты проспективных проверок - от 2026-10", mode="next")

# ============================================================================================ html
CSS = """
:root{--bg:#121211;--sf:#1a1a19;--ink:#f2f1ec;--ink2:#c3c2b7;--mut:#898781;--grid:#2c2c2a;--axis:#383835;--pink:#d55181;--gold:#c98500;--good:#0ca30c;--bad:#d03b3b}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#0b0b0a;overflow:hidden;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink2)}
#deck{position:absolute;left:50%;top:50%;width:1600px;height:900px;transform-origin:0 0}
.slide{position:absolute;left:0;top:0;width:1600px;height:900px;background:var(--bg);padding:34px 52px 0;overflow:hidden;opacity:0;visibility:hidden;pointer-events:none}
.slide.on{opacity:1;visibility:visible;pointer-events:auto}
.kick{font-size:17px;color:var(--gold);letter-spacing:.4px;text-transform:uppercase;font-weight:600}
h1{margin:6px 0 0;font-size:35px;line-height:1.14;color:var(--ink);font-weight:650;letter-spacing:-.3px;max-width:1480px}
.body{display:flex;gap:30px;margin-top:16px}
.side{width:372px;flex:none}.side .stat{background:var(--sf);border:1px solid var(--axis);border-radius:10px;padding:14px 18px;margin-bottom:16px}
.stat .big{font-size:64px;line-height:1;color:var(--ink);font-weight:700}.stat .lab{font-size:17px;margin-top:6px;color:var(--ink2)}
.lines{margin:0;padding:0;list-style:none}.lines li{font-size:20px;line-height:1.32;margin:0 0 12px;padding-left:16px;border-left:3px solid var(--gold);color:var(--ink2)}
.lines li b{color:var(--ink)}
.vis{flex:1;min-width:0}.plot{background:var(--bg)}
.wide .lines{display:flex;gap:22px;margin:12px 0 4px}.wide .lines li{flex:1;font-size:18px;margin:0}
.wide .body{display:block;margin-top:6px}
.vis.two,.vis.two2,.vis.two3{display:flex;gap:0}
.ev{position:absolute;left:52px;right:52px;bottom:12px;font-size:14px;color:var(--mut);border-top:1px solid var(--grid);padding-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.num{position:absolute;right:52px;top:38px;font-size:16px;color:var(--mut)}
.title h1{font-size:62px;margin-top:70px;max-width:1300px;line-height:1.08}.title .kick{font-size:22px;margin-top:50px}
.title .vis{position:absolute;left:52px;bottom:48px;opacity:.9}
.title .sub{font-size:26px;color:var(--ink2);margin-top:18px}
.legend2{position:absolute;left:60px;bottom:420px;font-size:19px}.legend2 span{margin-right:30px}
.next{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:26px}
.card{background:var(--sf);border:1px solid var(--axis);border-radius:12px;padding:22px 24px;height:520px}
.card h2{margin:0 0 6px;font-size:28px;color:var(--ink)}.card .tag{display:inline-block;font-size:15px;border:1px solid;border-radius:12px;padding:1px 10px;margin-bottom:14px}
.card ul{margin:0;padding:0;list-style:none}.card li{font-size:19px;line-height:1.35;margin:0 0 13px;padding-left:14px;border-left:3px solid var(--axis)}
.card li b{color:var(--ink)}.card code{background:#262624;padding:1px 6px;border-radius:4px;font-size:16px;color:var(--ink)}
.tl{margin-top:20px;border-top:2px solid var(--axis);position:relative}.tl .pt{font-size:16px;margin-top:10px}
#prog{position:fixed;left:0;top:0;height:5px;background:var(--gold);width:0;z-index:20;transition:width .2s}
#ctr{position:fixed;right:14px;bottom:10px;font-size:14px;color:var(--mut);z-index:20;background:rgba(18,18,17,.7);padding:2px 8px;border-radius:8px}
body.ov{overflow:auto}body.ov #deck{position:static;transform:none!important;width:auto;height:auto;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:18px}
body.ov .slide{position:relative;opacity:1;visibility:visible;pointer-events:none;zoom:.2;border:3px solid var(--axis)}
body.ov .slide.on{border-color:var(--gold)}
body.ov #ctr,body.ov #prog{display:none}
@page{size:1600px 900px;margin:0}
@media print{html,body{overflow:visible;height:auto;background:#121211}#deck{position:static;transform:none!important;width:1600px;height:auto}
.slide{position:relative;opacity:1;visibility:visible;page-break-after:always;break-after:page;-webkit-print-color-adjust:exact;print-color-adjust:exact}#prog,#ctr{display:none}.slide:last-of-type{page-break-after:auto;break-after:auto}}
"""

JS = """
const sl=[...document.querySelectorAll('.slide')],deck=document.getElementById('deck');let cur=0;
function fit(){const s=Math.min(innerWidth/1600,innerHeight/900);deck.style.transform=`scale(${s}) translate(-50%,-50%)`;}
function go(n,noHash){cur=Math.max(0,Math.min(sl.length-1,n));sl.forEach((e,i)=>e.classList.toggle('on',i===cur));
 document.getElementById('prog').style.width=((cur+1)/sl.length*100)+'%';document.getElementById('ctr').textContent=(cur+1)+' / '+sl.length;
 if(!noHash)history.replaceState(null,'','#'+(cur+1));if(document.body.classList.contains('ov')&&sl[cur].scrollIntoView)sl[cur].scrollIntoView({block:'center'});}
function ov(on){document.body.classList.toggle('ov',on===undefined?!document.body.classList.contains('ov'):on);fit();}
addEventListener('resize',fit);
addEventListener('keydown',e=>{const k=e.key;
 if(k==='ArrowRight'||k==='PageDown'||k===' '||k==='ArrowDown'){e.preventDefault();go(cur+1);}
 else if(k==='ArrowLeft'||k==='PageUp'||k==='ArrowUp'){e.preventDefault();go(cur-1);}
 else if(k==='Home')go(0);else if(k==='End')go(sl.length-1);
 else if(k==='o'||k==='O'||k==='щ'||k==='Щ')ov();else if(k==='Escape')ov(false);});
sl.forEach((e,i)=>e.addEventListener('click',ev=>{if(document.body.classList.contains('ov')){ov(false);go(i);return;}
 if(ev.target.closest('.js-plotly-plot'))return; if(ev.clientX<innerWidth*0.25)go(cur-1);else go(cur+1);}));
document.body.addEventListener('click',e=>{if(document.body.classList.contains('ov')){const s=e.target.closest('.slide');if(s){ov(false);go(sl.indexOf(s));}}},true);
const m=location.hash.match(/#(\\d+)/);go(m?+m[1]-1:0,true);fit();if(/ov/.test(location.search))ov(true);
"""


def slide_html(i, s):
    n = len(SL)
    if s["mode"] == "title":
        return (f'<section class="slide title"><div class="kick">{s["kick"]}</div><h1>{s["h"]}</h1>'
                f'<div class="sub">Волатильность предсказуема и растёт, направление угасло: что проверили и что держится на новых данных</div>'
                f'<div class="vis">{"".join(s["vis"])}</div>'
                f'<div class="ev">{s["ev"]}</div></section>')
    if s["mode"] == "next":
        return next_html(i, s)
    lines = "".join(f"<li>{x}</li>" for x in s["lines"])
    stat = f'<div class="stat"><div class="big">{s["stat"][0]}</div><div class="lab">{s["stat"][1]}</div></div>' if s["stat"] else ""
    vis = f'<div class="vis {s["vc"]}">{"".join(f"<div class=plot>{v}</div>" for v in s["vis"])}</div>'
    if s["mode"] == "wide":
        return (f'<section class="slide wide"><div class="num">{i + 1} / {n}</div><div class="kick">{s["kick"]}</div><h1>{s["h"]}</h1>'
                f'<ul class="lines">{lines}</ul><div class="body">{vis}</div><div class="ev">{s["ev"]}</div></section>')
    return (f'<section class="slide"><div class="num">{i + 1} / {n}</div><div class="kick">{s["kick"]}</div><h1>{s["h"]}</h1>'
            f'<div class="body"><div class="side">{stat}<ul class="lines">{lines}</ul></div>{vis}</div><div class="ev">{s["ev"]}</div></section>')


def next_html(i, s):
    cards = [
        ("В MVP", "ветка nt-tactical-harscale", T.GOOD, [
            "PR: <b>необязательный масштаб HAR</b> для conformal-интервалов (кандидат D-008).",
            f"Основание: H85, {nbest} из {nm_} месяцев, интервал около 7% уже при покрытии 0,90.",
            "По умолчанию ничего не меняется: включается опцией; решает ведущий MVP и QA.",
            "Не в PR: вол-тайминг (H84), календарь (H86)."]),
        ("Параллельная сессия", "обзор: OUTLOOK_positive.md", GOLD, [
            "<b>Геометрия</b> индикаторов: непройденные варианты U1-U11.",
            "<b>Физика</b>: термы на оси дисперсии (H75-H77), где был малый плюс.",
            "<b>Регрессия</b> как честная база; <b>синтетические индикаторы</b> с HAR-признаками (честное сравнение с HAR).",
            "Направление - только если появится новый источник информации (D-050: пока нет)."]),
        ("Проспективные тесты", "заморожены, смотрим через месяцы", PINK, [
            "<b>T04-B</b>: календарь на новых месяцах с 2026-10; <code>score_B.py</code> - один раз, при 3 полных месяцах.",
            "<b>T05-B</b>: HAR против моделей, заморожено (sha256 e9c38d61…).",
            "Binance-файл докачивать помесячно от 2026-10; критерии - как в SPEC, не менять.",
            "Открыто для владельца: чистое дообучение H62 на Binance (не рекомендуется после H82)."])]
    cs = "".join(f'<div class="card"><h2>{t}</h2><span class="tag" style="color:{c};border-color:{c}">{tag}</span><ul>{"".join(f"<li>{x}</li>" for x in li)}</ul></div>' for t, tag, c, li in cards)
    return (f'<section class="slide"><div class="num">{i + 1} / {len(SL)}</div><div class="kick">{s["kick"]}</div><h1>{s["h"]}</h1>'
            f'<div class="next">{cs}</div><div class="tl"><div style="display:flex;justify-content:space-between" class="pt"><span>2026-10-13 · итоги сессии</span><span>2026-10… · докачка Binance помесячно</span><span>не раньше 2027-01 · T04-B (3 полных месяца), T05-B</span></div></div><div class="ev">{s["ev"]}</div></section>')


def main():
    body = "".join(slide_html(i, s) for i, s in enumerate(SL))
    page = ("<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Тактическая сессия: слайды</title><style>" + CSS + "</style><script>" + po.get_plotlyjs() + "</script></head><body>"
            "<div id='prog'></div><div id='ctr'></div><div id='deck'>" + body + "</div><script>" + JS + "</script></body></html>")
    out = HERE / "tactical_deck.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} {out.stat().st_size / 1e6:.1f} MB, {len(SL)} slides")


if __name__ == "__main__":
    main()
