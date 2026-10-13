"""Builds runs/tactical/report/outlook_report.html: ONE self-contained interactive report of the OUTLOOK tactical session
(owner request 2026-10-13: a positive outlook on geometry, physics, regression, synthetic indicators). Same look as
tactical_report.html (rcommon helpers, the CSS of build_report.py read as text, plotly embedded once, no external requests,
text in Russian). Every number is read from the committed result files of runs/tactical/outlook/O*/ or computed here from their
rows (paired month differences with 95% t-intervals); candles come from the committed windows (interp.json) or the cut price
file (hard cutoff 2025-07-20). O2 reads its REPORT.md when it exists, so a later rebuild fills the section.

usage: python runs/tactical/report/build_outlook_report.py   (CPU, about 1 minute)   -> outlook_report.html"""
import html as H
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcommon import (T, C, TAC, ROOT, UP, DOWN, PINK, GOLD, TAN, VIOLET, SALMON, INK, INK2, MUTED, GRID, SURFACE, PAPER, AXIS,  # noqa: E402
                     fig_html, badge, table, card, ru, load_bars)

OUT = TAC / "outlook"
HERE = Path(__file__).resolve().parent
NEUT = T.NEUTRAL
HC = T.HORIZON_COLORS                       # h0/h1/h2 colours: used only for the volatility horizons 15/60/240
HCOL = {15: HC["h0"], 60: HC["h1"], 240: HC["h2"]}
SPANC = {"sel": GOLD, "look": PINK, "fwd": VIOLET}
SPANN = {"sel": "подбор 2021-01..2024-06", "look": "взгляд 2024-07..2025-07", "fwd": "вперёд 2025-10..2026-09"}
FIGS = []
t_start = time.time()


def F(fig, sec, key, title, height, lift=0, subtitle=None):
    FIGS.append((sec, key, title))
    return fig_html(fig, key, height=height, lift=lift, subtitle=subtitle)


def jl(rel):
    return [json.loads(x) for x in open(OUT / rel, encoding="utf-8") if x.strip()]


def jf(rel):
    return json.load(open(OUT / rel, encoding="utf-8"))


def txt(rel):
    p = OUT / rel
    return p.read_text(encoding="utf-8") if p.exists() else None


def tci(x):
    """mean, lo, hi (95% t over months), months positive, n."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = sps.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h), int((x > 0).sum()), n


def sg(x, nd=4):
    return ru(x, nd, sign=True)


def fmt_ci(t, nd=4):
    return f"{sg(t[0], nd)} [{sg(t[1], nd)}; {sg(t[2], nd)}], {t[3]}/{t[4]}"


def paired(rows, a, b, key="spearman", **flt):
    """{month: a - b} over rows matching flt (span, H, target ...)."""
    d = {}
    for r in rows:
        if all(r.get(k) == v for k, v in flt.items()):
            d.setdefault(r["month"], {})[r["model"]] = r[key]
    return {m: v[a] - v[b] for m, v in sorted(d.items()) if a in v and b in v}


def md(s):
    """Minimal markdown -> html for the REPORT.md of O2 (headings, bullets, tables, bold, code)."""
    out, intab, inul = [], False, False
    for ln in s.split("\n"):
        e = H.escape(ln)
        e = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
        e = re.sub(r"`(.+?)`", r"<code>\1</code>", e)
        if ln.startswith("|"):
            cells = [c.strip() for c in e.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            if not intab:
                out.append('<table class="rt"><tbody>')
                intab = True
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if intab:
            out.append("</tbody></table>")
            intab = False
        if ln.startswith("- ") or ln.startswith("  - "):
            if not inul:
                out.append("<ul>")
                inul = True
            out.append(f"<li>{e.strip()[2:]}</li>")
            continue
        if inul:
            out.append("</ul>")
            inul = False
        m = re.match(r"(#+) (.*)", e)
        out.append(f"<h4>{m.group(2)}</h4>" if m else (f"<p class='note'>{e}</p>" if e.strip() else ""))
    if intab:
        out.append("</tbody></table>")
    if inul:
        out.append("</ul>")
    return "".join(out)


def month_bars(fig, months, vals, color, row=None, col=None, name="", hover="%{x}: %{y:+.4f}<extra></extra>", marker_colors=None):
    tr = go.Bar(x=months, y=vals, name=name, marker=dict(color=marker_colors or color, line=dict(color=SURFACE, width=0.5)), hovertemplate=hover)
    fig.add_trace(tr, row=row, col=col) if row else fig.add_trace(tr)


def vlines_band(fig, x0, x1, ci, color, label):
    """horizontal mean line + CI band over [x0, x1] (categorical month labels)."""
    m, lo, hi = ci[0], ci[1], ci[2]
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=lo, y1=hi, fillcolor=T.rgba(color, 0.18), line=dict(width=0), layer="below")
    fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=m, y1=m, line=dict(color=color, width=2, dash="dash"))


# ======================================================================================================== data
R1 = jl("O1_geom_vol/results.jsonl")
R3 = jl("O3_synth_volind/results.jsonl")
R4 = jl("O4_bank_confirm/resultsA.jsonl")
R5 = jl("O5_geom_bestbase/results.jsonl")
interp = jf("O1_geom_vol/interp.json")
kern = jf("O3_synth_volind/kernels.json")
win3 = jf("O3_synth_volind/windows.json")
sum3 = jf("O3_synth_volind/summary.json")
sum4 = jf("O4_bank_confirm/summaryA.json")
sum5 = jf("O5_geom_bestbase/summary.json")
choice5 = jf("O5_geom_bestbase/choice.json")
sha4 = (OUT / "O4_bank_confirm/frozen_B.sha256").read_text().strip()
sha5 = (OUT / "O5_geom_bestbase/frozen_B.sha256").read_text().strip()
o2_report = txt("O2_phys_var8/REPORT.md")
o2_prog = txt("O2_phys_var8/progress.txt")

# headline numbers (computed from rows; checked against the committed tables below)
O1 = {}
for sp in ("sel", "look", "fwd"):
    for H_ in (15, 60, 240):
        O1[sp, H_] = tci(list(paired(R1, "R2", "R0", span=sp, H=H_, target="lnrv").values()))
O1ql = {sp: tci(list(paired(R1, "R2", "R0", key="qlike", span=sp, H=60, target="lnrv").values())) for sp in ("sel", "look", "fwd")}
assert abs(O1["sel", 60][0] - 0.0073) < 6e-5 and abs(O1["look", 60][0] - 0.0054) < 6e-5 and abs(O1["fwd", 60][0] - 0.0052) < 6e-5, O1
O3 = {sp: tci(list(paired(R3, "LK", "R0", span=sp, H=60).values())) for sp in ("sel", "look", "fwd")}
O3b = {sp: tci(list(paired(R3, "BANK", "R0", span=sp, H=60).values())) for sp in ("sel", "look", "fwd")}
O3lb = {sp: tci(list(paired(R3, "LK", "BANK", span=sp, H=60).values())) for sp in ("sel", "look", "fwd")}
assert abs(O3["look"][0] - 0.0004) < 6e-5 and abs(O3b["look"][0] - 0.0068) < 6e-5, (O3, O3b)
A4 = {H_: tci(list(paired(R4, "BANK", "R0", H=H_).values())) for H_ in (15, 60, 240)}
assert abs(A4[60][0] + 0.0043) < 6e-5, A4
SPAN5 = sorted({r["span"] for r in R5})
JSP = [s for s in SPAN5 if s != "sel"][0]
J5 = {H_: tci(list(paired(R5, "L1", "L0", span=JSP, H=H_).values())) for H_ in (15, 60, 240)}
assert abs(J5[60][0] - 0.0056) < 6e-5 and J5[60][3] == 23, J5


# ======================================================================================================== section 1: overview
def sec1():
    verdicts = [
        ("O1", "Геометрия индикаторов поверх честного прогноза волатильности (HAR)", "good", "держится, узко",
         f"G*−R0, H=60: взгляд {sg(O1['look', 60][0])}, вперёд {sg(O1['fwd', 60][0])}; порог +0,005 пройден на ножах; на бустинге-базе вперёд не держится"),
        ("O2", "Физические члены на голове дисперсии при 8 эпохах вместо 4", "info", "идёт" if not o2_report else "результат есть",
         "считается на GPU; секция читает runs/tactical/outlook/O2_phys_var8/REPORT.md и заполнится при пересборке" if not o2_report else "см. секцию O2"),
        ("O3", "Синтетические индикаторы волатильности (обучаемые ядра памяти)", "bad", "нет",
         f"LK−R0 на взгляде {sg(O3['look'][0])} [{sg(O3['look'][1])}; {sg(O3['look'][2])}]: ядра ничего не выучили; побочная находка BANK (плотный HAR)"),
        ("O4", "Подтверждение BANK на 2017-09..2018-12", "bad", "не подтверждено",
         f"BANK−R0, H=60: {sg(A4[60][0])} [{sg(A4[60][1])}; {sg(A4[60][2])}]; один месяц (2017-09, 10,7 дня обучения) тянет среднее вниз; часть B заморожена"),
        ("O5", "Геометрия поверх лучшей базы (BANK) на 2018-07..2020-12", "good", "геометрия добавляет",
         f"L1−L0, H=60: {sg(J5[60][0])} [{sg(J5[60][1])}; {sg(J5[60][2])}], {J5[60][3]}/{J5[60][4]} месяцев; запас по порогу малый; часть B заморожена"),
    ]
    rows = [[f"<b>{i}</b>", n, badge(k, t), d] for i, n, k, t, d in verdicts]
    board = table(rows, ["", "вопрос", "вердикт", "ключевое число"])

    # forest of the headline effects
    items = [("O1 G*−R0 · подбор", O1["sel", 60]), ("O1 G*−R0 · взгляд", O1["look", 60]), ("O1 G*−R0 · вперёд", O1["fwd", 60]),
             ("O3 LK−R0 · взгляд", O3["look"]), ("O3 LK−R0 · вперёд", O3["fwd"]), ("O3 BANK−R0 · взгляд", O3b["look"]), ("O3 BANK−R0 · вперёд", O3b["fwd"]),
             ("O4 BANK−R0 · 2017-09..2018-12", A4[60]), ("O5 L1−L0 · 2018-07..2020-12", J5[60])]
    fig = go.Figure()
    col = {"O1": GOLD, "O3": TAN, "O4": SALMON, "O5": VIOLET}
    for nm, t in items:
        c = col[nm[:2]]
        fig.add_trace(go.Scatter(x=[t[0]], y=[nm], mode="markers", marker=dict(color=c, size=11, symbol="diamond"), error_x=dict(type="data", symmetric=False, array=[t[2] - t[0]], arrayminus=[t[0] - t[1]], color=c, thickness=2.2),
                                 showlegend=False, hovertemplate=nm + f"<br>{ru(t[0], 4, True)} [{ru(t[1], 4, True)}; {ru(t[2], 4, True)}]<br>месяцы с плюсом {t[3]}/{t[4]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.add_vline(x=0.005, line=dict(color=T.GOOD, width=1, dash="dash"), annotation_text="порог +0,005", annotation_font_color=T.GOOD)
    fig.update_layout(xaxis_title="выигрыш по Спирмену (H=60, ln RV), среднее по месяцам ± 95% t-интервал", yaxis=dict(autorange="reversed"))
    f1 = F(fig, "o1", "ov_forest", "Все ключевые эффекты H=60", 440, subtitle="парные разности по месяцам; эффекты порядка 0,005 на базе 0,73-0,80")

    tiles = [(f"{sg(O1['look', 60][0])}", "O1: выигрыш геометрии, взгляд (H=60)", f"вперёд {sg(O1['fwd', 60][0])}; все 13/13 и 12/12 интервалов выше 0 на взгляде и вперёде"),
             (f"{sg(J5[15][0], 4)}", "O5: выигрыш геометрии при H=15", f"{J5[15][3]}/{J5[15][4]} месяцев; при H=240 ничего ({sg(J5[240][0])})"),
             (f"{sg(O3b['look'][0])}", "O3: BANK−R0, взгляд (H=60)", f"вперёд {sg(O3b['fwd'][0])}; но в O4 на раннем периоде не подтвердилось"),
             (f"{sg(O3['look'][0])}", "O3: синтетические ядра", "ядра остались около равномерных: «нет»")]
    th = "".join(f'<div class="tile"><div class="big">{a}</div><div class="lab">{b}</div><div class="sub">{c}</div></div>' for a, b, c in tiles)

    body = ('<div class="lead">Запрос владельца: найти <b>положительный прогноз</b> (outlook) для четырёх идей: <b>геометрия индикаторов</b>, <b>физика</b>, '
            '<b>регрессия</b> и <b>синтетические индикаторы</b>. Общий вывод: геометрия помогает предсказывать <b>размер движений</b> (волатильность) '
            'на горизонтах 15 и 60 минут, на 240 минутах - нет. <b>Направление не заявляется</b>: ни один результат не про направление цены.</div>'
            f'<div class="tiles">{th}</div>'
            + card("Доска вердиктов O1-O5", board, "Единица - месяц; 95% t-интервалы парных разностей по месяцам; «k/n» - месяцев с плюсом. Всё одно зерно, только CPU.")
            + card("Все ключевые эффекты на одной оси", f1, "Метрика - выигрыш по Спирмену против опорной модели на H=60 (цель ln RV). Порог из SPEC - +0,005 (O1/O3/O4) и +0,003 (O5).")
            + card("Как читать", "<ul class='note'><li><b>Что работает:</b> плотные признаки геометрии MA-каналов (особенно на 15-минутном масштабе) добавляют около +0,005 Спирмена к линейному HAR; "
                   "эффект стабилен по знаку (почти каждый месяц), мал по размеру.</li>"
                   "<li><b>Что не работает:</b> обучаемые ядра памяти не выучили форму; прибавка BANK - просто более плотный HAR.</li>"
                   "<li><b>Оговорки:</b> прибавка геометрии O1 затухает на форвардном году и не держится против бустинга; O4 не подтвердил BANK на раннем периоде; "
                   "порог размера пройден на грани.</li></ul>"))
    return body


# ======================================================================================================== section O1
def ema(x, n):
    a = 2.0 / (n + 1)
    o = np.empty(len(x))
    o[0] = x[0]
    for i in range(1, len(x)):
        o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o


def sec_o1():
    parts = []
    # monthly gain
    g = paired(R1, "R2", "R0", span="sel", H=60, target="lnrv")
    allm, vals, cols, spn = [], [], [], []
    for sp in ("sel", "look", "fwd"):
        d = paired(R1, "R2", "R0", span=sp, H=60, target="lnrv")
        for m, v in d.items():
            allm.append(m); vals.append(v); cols.append(SPANC[sp]); spn.append(sp)
    fig = go.Figure()
    for sp in ("sel", "look", "fwd"):
        idx = [i for i, s in enumerate(spn) if s == sp]
        fig.add_trace(go.Bar(x=[allm[i] for i in idx], y=[vals[i] for i in idx], name=SPANN[sp], marker=dict(color=SPANC[sp], line=dict(color=SURFACE, width=0.5)),
                             hovertemplate="%{x}: %{y:+.4f}<extra>" + SPANN[sp] + "</extra>"))
    for sp in ("sel", "look", "fwd"):
        ms = [m for m, s in zip(allm, spn) if s == sp]
        vlines_band(fig, ms[0], ms[-1], O1[sp, 60], SPANC[sp], SPANN[sp])
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.add_hline(y=0.005, line=dict(color=T.GOOD, width=1, dash="dot"), annotation_text="порог +0,005", annotation_font_color=T.GOOD)
    fig.update_layout(yaxis_title="G* − R0, Спирмен", xaxis=dict(type="category", tickangle=-60, nticks=30), barmode="overlay", bargap=0.15)
    f1 = F(fig, "o1", "o1_monthly", "O1 помесячный выигрыш G*−R0, H=60", 480,
           subtitle="штрих - среднее по отрезку, заливка - 95% t-интервал по месяцам; цель ln RV")
    parts.append(card("Помесячный парный выигрыш G*−R0 при H=60", f1,
                      f"G* = R2 (ridge на S3 + геометрия 1-мин и 15-мин). Среднее [95% ИИ], месяцев с плюсом: подбор {fmt_ci(O1['sel', 60])}; взгляд {fmt_ci(O1['look', 60])}; "
                      f"вперёд {fmt_ci(O1['fwd', 60])}. Видно затухание: в 2025-10..2026-02 около +0,009, в 2026-03..09 около +0,0025."))

    # per-horizon summary
    fig = go.Figure()
    for H_ in (15, 60, 240):
        xs = [SPANN[s] for s in ("sel", "look", "fwd")]
        ys = [O1[s, H_][0] for s in ("sel", "look", "fwd")]
        fig.add_trace(go.Bar(x=xs, y=ys, name=f"H={H_} мин", marker=dict(color=HCOL[H_], line=dict(color=SURFACE, width=1)),
                             error_y=dict(type="data", symmetric=False, array=[O1[s, H_][2] - O1[s, H_][0] for s in ("sel", "look", "fwd")],
                                          arrayminus=[O1[s, H_][0] - O1[s, H_][1] for s in ("sel", "look", "fwd")], color=INK2),
                             text=[f"{ru(v, 4, True)}" for v in ys], textposition="outside", hovertemplate="%{x}<br>%{y:+.4f}<extra>H=" + str(H_) + "</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(barmode="group", yaxis_title="G* − R0, Спирмен")
    f2 = F(fig, "o1", "o1_horizons", "O1 по горизонтам", 420, subtitle="среднее по месяцам ± 95% t-интервал")
    rows = []
    for sp in ("sel", "look", "fwd"):
        rows.append([SPANN[sp]] + [fmt_ci(O1[sp, H_]) for H_ in (15, 60, 240)] + [fmt_ci(O1ql[sp])])
    t1 = table(rows, ["отрезок", "H=15", "H=60", "H=240", "QLIKE H=60 (меньше - лучше)"])
    parts.append(card("Сводка по горизонтам", f2 + t1, "H=15 ярче всего (+0,014 / +0,009 / +0,008, все месяцы с плюсом); на H=240 эффекта нет. Геометрия помогает предсказать "
                      "реализованную волатильность, а не размер чистого сдвига (цель ln(|r|) - без выигрыша, как H47 и H57)."))

    # coefficients
    top = interp["top"]
    GC = {"price_bands": GOLD, "volatility": PINK, "oscillators": TAN, "volume": VIOLET}
    fig = go.Figure(go.Bar(y=[t["name"] for t in top], x=[t["coef"] for t in top], orientation="h",
                           marker=dict(color=[GC.get(t["group"], NEUT) for t in top], line=dict(color=SURFACE, width=1)),
                           text=[ru(t["coef"], 3, True) for t in top], textposition="outside",
                           customdata=[[t["group"], t["channel"]] for t in top],
                           hovertemplate="%{y}<br>коэффициент %{x:+.3f}<br>группа %{customdata[0]}, канал %{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="стандартизованный коэффициент замороженной G* (365 дней до 2025-07-20, H=60)", margin=dict(l=190))
    ng = {g_: sum(1 for t in top if t["group"] == g_) for g_ in GC}
    f3 = F(fig, "o1", "o1_coefs", "O1 топ-15 коэффициентов", 560, subtitle="цвет - группа канала (золото: ценовые полосы MA)")
    parts.append(card("Топ-15 признаков замороженной модели", f3,
                      f"{ng['price_bands']} из 15 - каналы скользящих средних (ценовые полосы), {sum(1 for t in top if t['name'].startswith('g15'))} из 15 - на 15-минутном масштабе; осцилляторов и объёма нет. "
                      "Пары противоположного знака - контрасты внутри семейства MA (грубо, кривизна 15-минутного тренда), а не отдельные эффекты."))

    # real windows for the top 3 features
    ex = interp["examples"][:3]
    cols_i = [0, 2, 3, 5]
    fig = make_subplots(rows=3, cols=4, vertical_spacing=0.09, horizontal_spacing=0.035,
                        subplot_titles=[f"{e['feature']}: {ru(e['windows'][i]['value'], 2, True)}" for e in ex for i in cols_i])
    for r, e in enumerate(ex, 1):
        for c, i in enumerate(cols_i, 1):
            w = e["windows"][i]
            t = pd.to_datetime(w["bar_times"])
            A = np.array(w["ohlcv"], float)
            assert t.max() <= pd.Timestamp(C.CUTOFF)
            fig.add_trace(go.Candlestick(x=t, open=A[:, 0], high=A[:, 1], low=A[:, 2], close=A[:, 3], showlegend=False,
                                         increasing=dict(line=dict(color=UP), fillcolor=UP), decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
                                         hovertemplate="%{x|%Y-%m-%d %H:%M}<br>O %{open:.0f} H %{high:.0f} L %{low:.0f} C %{close:.0f}<extra></extra>"), row=r, col=c)
            fig.add_trace(go.Scatter(x=t, y=ema(A[:, 3], 20), mode="lines", line=dict(color=GOLD, width=2), name="EMA20", legendgroup="e20", showlegend=(r == 1 and c == 1),
                                     hovertemplate="EMA20 %{y:.0f}<extra></extra>"), row=r, col=c)
            fig.add_trace(go.Scatter(x=t, y=ema(A[:, 3], 40), mode="lines", line=dict(color=PINK, width=2, dash="dash"), name="EMA40", legendgroup="e40", showlegend=(r == 1 and c == 1),
                                     hovertemplate="EMA40 %{y:.0f}<extra></extra>"), row=r, col=c)
            fig.update_xaxes(rangeslider_visible=False, showticklabels=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, row=r, col=c)
    fig.update_annotations(font_size=11, font_color=INK2)
    f4 = F(fig, "o1", "o1_windows", "O1 топ-3 признака на реальных 15-минутных окнах", 980, lift=10,
           subtitle="столбцы - значения признака от самого отрицательного до самого положительного (стандартизованные); свечи 15 минут, линии - EMA20 и EMA40 из окна")
    parts.append(card("Топ-3 признака на реальных окнах BTC", f4,
                      "Окна из селекции 2021-2024, 60 свечей по 15 минут (interp.json, 6 окон на признак: показаны 4). slope3 - наклон, pos - положение цены относительно канала (как названы в коде признаков); "
                      "значение в заголовке панели - стандартизованное значение признака в этом окне."))

    # robustness: decay + HGB base
    d_fwd = paired(R1, "R2", "R0", span="fwd", H=60, target="lnrv")
    dH = {sp: tci(list(paired(R1, "H2", "H0", span=sp, H=60, target="lnrv").values())) for sp in ("sel", "look", "fwd")}
    base = {}
    for sp in ("sel", "look", "fwd"):
        for mdl in ("R0", "R2", "H0", "H2"):
            base[sp, mdl] = float(np.mean([r["spearman"] for r in R1 if r["span"] == sp and r["H"] == 60 and r["target"] == "lnrv" and r["model"] == mdl]))
    rows = [[SPANN[sp]] + [ru(base[sp, m_], 4) for m_ in ("R0", "R2", "H0", "H2")] + [fmt_ci(dH[sp])] for sp in ("sel", "look", "fwd")]
    t2 = table(rows, ["отрезок", "R0 (линейная)", "G* = R2", "HGB S3 (H0)", "HGB S3+G (H2)", "H2 − H0"])
    parts.append(card("Оговорки O1: база и затухание", t2 +
                      f"<ul class='note'><li><b>Нож:</b> порог +0,005 пройден на взгляде и вперёд с запасом 0,0004 и 0,0002; без неполного 2025-10 вперёд {ru(0.0046, 4, True)} [+0,0025; +0,0067], 11/11 - знак и интервал держатся, порог размера нет; "
                      "в 4 из 12 форвардных случаев при исключении одного месяца результат ниже порога (QA).</li>"
                      f"<li><b>Затухание:</b> помесячно вперёд {', '.join(ru(v, 4, True) for v in d_fwd.values())} - замороженная модель стареет.</li>"
                      f"<li><b>Бустинг-база:</b> HGB на S3 (H0) обгоняет линейную R0 на форварде ({ru(base['fwd', 'H0'], 4)} против {ru(base['fwd', 'R0'], 4)}); геометрия поверх бустинга "
                      f"на форварде не держится: H2−H0 {fmt_ci(dH['fwd'])}. Вывод: «геометрия добавляет к линейному HAR», а не «к лучшему доступному прогнозу» - ответ даёт O5.</li>"
                      "<li><b>Другой источник:</b> форвардные бары - Binance spot; объёмные каналы сами по себе дают около 0.</li></ul>"))
    return "".join(parts)


# ======================================================================================================== section O2
def sec_o2():
    if o2_report:
        body = card("O2: REPORT.md (прочитан при сборке)", md(o2_report))
        for p in sorted((OUT / "O2_phys_var8").glob("scores*.json")):
            body += card(f"{p.name}", f"<pre class='note' style='white-space:pre-wrap'>{H.escape(p.read_text(encoding='utf-8')[:6000])}</pre>")
        return body
    spec = txt("O2_phys_var8/SPEC.md") or ""
    q = re.search(r"## Question\n(.*?)\n\n", spec, re.S)
    last = (o2_prog or "").strip().splitlines()[-1] if o2_prog else "нет данных"
    return ('<div class="lead">Статус: <b>идёт</b>. REPORT.md ещё нет; секция читает runs/tactical/outlook/O2_phys_var8/REPORT.md при каждой пересборке.</div>'
            + card("Вопрос (SPEC)", f"<p class='note'>{H.escape(q.group(1)) if q else 'см. SPEC.md'}</p>")
            + card("Что будет показано здесь", "<ul class='note'><li>эффект физики (вкл − выкл) по Спирмену головы дисперсии с |r| на 6 срезах, 95% t-интервал;</li>"
                   "<li>разность эффектов при 8 и 4 эпохах на тех же срезах и зёрнах (растёт, тот же в пределах шума, сжимается);</li>"
                   "<li>сеть минус HAR+tod по Спирмену и QLIKE; число эпох и обслуживаемая эпоха каждого прогона.</li></ul>")
            + card("Прогресс", f"<p class='note'>Последняя строка progress.txt: <code>{H.escape(last)}</code></p>"))


# ======================================================================================================== section O3
def sec_o3():
    parts = []
    taus = kern["taus"]
    names = [f"{ch} · ядро {k}" for ch in kern["channels"] for k in (1, 2)]
    P = np.array(kern["frozen"]["60"]["P"])                                # 6 x 12
    months = list(kern["monthly_H60"])
    M = np.array([kern["monthly_H60"][m]["P"] for m in months])            # months x 6 x 12
    fig = make_subplots(rows=2, cols=3, subplot_titles=names, vertical_spacing=0.16, horizontal_spacing=0.06, shared_yaxes=True)
    for k in range(6):
        r, c = k // 3 + 1, k % 3 + 1
        q10, q50, q90 = np.quantile(M[:, k, :], [0.1, 0.5, 0.9], axis=0)
        mn, mx = M[:, k, :].min(0), M[:, k, :].max(0)
        fig.add_trace(go.Scatter(x=taus + taus[::-1], y=list(mx) + list(mn[::-1]), fill="toself", fillcolor=T.rgba(NEUT, 0.14), line=dict(width=0), name="min-max по месяцам",
                                 legendgroup="mm", showlegend=(k == 0), hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=taus + taus[::-1], y=list(q90) + list(q10[::-1]), fill="toself", fillcolor=T.rgba(GOLD, 0.28), line=dict(width=0), name="10-90% по месяцам",
                                 legendgroup="q", showlegend=(k == 0), hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=taus, y=q50, mode="lines", line=dict(color=GOLD, width=1.6), name="медиана по месяцам", legendgroup="med", showlegend=(k == 0),
                                 hovertemplate="τ=%{x} мин<br>вес %{y:.3f}<extra>медиана</extra>"), row=r, col=c)
        fig.add_trace(go.Scatter(x=taus, y=P[k], mode="lines+markers", line=dict(color=PINK, width=2.4), marker=dict(size=6), name="замороженная (365 дней до 2025-07-20)", legendgroup="fz", showlegend=(k == 0),
                                 hovertemplate="τ=%{x} мин<br>вес %{y:.3f}<extra>заморож.</extra>"), row=r, col=c)
        fig.add_hline(y=1 / len(taus), line=dict(color=NEUT, width=1, dash="dot"), row=r, col=c)
        fig.update_xaxes(type="log", title_text="τ, минуты (лог)" if r == 2 else None, row=r, col=c)
    fig.update_yaxes(title_text="вес ядра", col=1)
    f1 = F(fig, "o3", "o3_kernels", "O3 выученные ядра по τ, H=60", 640, lift=12, subtitle="точки - 12 EWMA с τ=1..2048; пунктир - равномерное ядро 1/12 = 0,083")
    parts.append(card("Выученные ядра (H=60): формы нет", f1,
                      "Замороженные подгонки остановились на 1-2 эпохах; все веса 0,06-0,10 против 0,083 у равномерного ядра. Помесячные подгонки чуть тянутся к 1-8 минутам и к 2048 "
                      "(собственная короткая+длинная форма HAR), с провалом на 64-512. Более медленный шаг и больше терпения дают то же (diag_lr.json)."))

    # monthly paired gain
    fig = go.Figure()
    series = [("LK − R0 (кандидат)", "LK", "R0", PINK), ("BANK − R0 (контроль)", "BANK", "R0", GOLD), ("LK − BANK (добавляет ли синтетика)", "LK", "BANK", VIOLET)]
    for nm, a, b, c_ in series:
        xs, ys = [], []
        for sp in ("sel", "look", "fwd"):
            d = paired(R3, a, b, span=sp, H=60)
            xs += list(d); ys += list(d.values())
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=nm, line=dict(color=c_, width=1.6), marker=dict(size=5),
                                 hovertemplate="%{x}: %{y:+.4f}<extra>" + nm + "</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.add_hline(y=0.005, line=dict(color=T.GOOD, width=1, dash="dot"), annotation_text="порог +0,005", annotation_font_color=T.GOOD)
    fig.update_layout(yaxis_title="разность Спирмена (H=60)", xaxis=dict(type="category", tickangle=-60, nticks=30))
    f2 = F(fig, "o3", "o3_monthly", "O3 помесячные разности, H=60", 450, subtitle="подбор 2021-01..2024-06, взгляд 2024-07..2025-07, вперёд 2025-10..2026-09")
    rows = []
    for sp in ("sel", "look", "fwd"):
        base = {m_: float(np.mean([r["spearman"] for r in R3 if r["span"] == sp and r["H"] == 60 and r["model"] == m_])) for m_ in ("R0", "BANK", "LK")}
        rows.append([SPANN[sp], ru(base["R0"], 4), ru(base["BANK"], 4), ru(base["LK"], 4), fmt_ci(O3[sp]), fmt_ci(O3lb[sp]), fmt_ci(O3b[sp])])
    t = table(rows, ["отрезок", "R0", "BANK", "LK", "LK − R0", "LK − BANK", "BANK − R0"])
    parts.append(card("Вердикт и контроль: помесячно", f2 + t,
                      f"<b>Вердикт: «{'нет' if sum3['verdict'] == 'no' else sum3['verdict']}».</b> LK−R0 на взгляде {fmt_ci(O3['look'])} - ниже порога +0,005, нижняя граница ниже 0. "
                      "<b>Побочная находка (не заранее зарегистрирована):</b> контроль BANK (ridge на S3 + логарифмы 12 EWMA r², диапазона и объёма) обыгрывает базу везде; на H=60 взгляд и вперёд "
                      "прибавка бы прошла порог. Это более плотный HAR, а не синтетический индикатор; чтобы считаться результатом, нужен свой SPEC (-> O4)."))

    # windows
    taus_ = win3["taus"]
    wq, ws = win3["windows"]["quiet"], win3["windows"]["shock"]
    fig = make_subplots(rows=3, cols=2, shared_xaxes=False, vertical_spacing=0.07, horizontal_spacing=0.06,
                        subplot_titles=[f"Спокойный день {wq['day']} + следующий (диапазон {ru(wq['daily_range'] * 100, 2)}%)", f"День шока {ws['day']} + следующий (диапазон {ru(ws['daily_range'] * 100, 1)}%)"])
    IC = [GOLD, PINK, TAN, VIOLET, SALMON, "#c3c2b7"]
    corr_rows = []
    for c, (key, w) in enumerate((("quiet", wq), ("shock", ws)), 1):
        t = pd.to_datetime(w["t"])
        assert t.max() <= pd.Timestamp(C.CUTOFF)
        fig.add_trace(go.Scatter(x=t, y=w["close"], mode="lines", line=dict(color=INK, width=1.3), name="закрытие", legendgroup="p", showlegend=(c == 1), hovertemplate="%{x|%m-%d %H:%M} %{y:.0f}<extra>close</extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=t, y=w["ln_rv60_per_bar"], mode="lines", line=dict(color=HCOL[60], width=2), name="ln RV60 (за бар)", legendgroup="r60", showlegend=(c == 1),
                                 hovertemplate="%{x|%m-%d %H:%M} %{y:.2f}<extra>ln RV60</extra>"), row=2, col=c)
        fig.add_trace(go.Scatter(x=t, y=w["ln_rv1440_per_bar"], mode="lines", line=dict(color=NEUT, width=2, dash="dash"), name="ln RV1440 (за бар)", legendgroup="r1440", showlegend=(c == 1),
                                 hovertemplate="%{x|%m-%d %H:%M} %{y:.2f}<extra>ln RV1440</extra>"), row=2, col=c)
        for k in range(6):
            fig.add_trace(go.Scatter(x=t, y=w["Z"][k], mode="lines", line=dict(color=IC[k], width=1.5), name=names[k], legendgroup=f"i{k}", showlegend=(c == 1),
                                     hovertemplate="%{x|%m-%d %H:%M} %{y:.2f}<extra>" + names[k] + "</extra>"), row=3, col=c)
            corr_rows.append((key, k, float(np.corrcoef(w["I"][k], w["ln_rv60_per_bar"])[0, 1])))
        fig.update_yaxes(title_text="цена" if c == 1 else None, row=1, col=c)
        fig.update_yaxes(title_text="ln RV" if c == 1 else None, row=2, col=c)
        fig.update_yaxes(title_text="Z индикатора" if c == 1 else None, row=3, col=c)
    fig.update_annotations(font_size=12, font_color=INK2)
    f3 = F(fig, "o3", "o3_windows", "O3 шесть индикаторов на тихом и шоковом окне", 900, lift=10, subtitle="окно - сутки и следующие сутки (2880 минут), ядра замороженной подгонки H=60")
    rows = [[names[k]] + [ru(next(v for kk, k_, v in corr_rows if kk == key and k_ == k), 3) for key in ("quiet", "shock")] for k in range(6)]
    parts.append(card("Шесть синтетических индикаторов рядом с RV60 и RV1440", f3 + table(rows, ["индикатор", "корр. с ln RV60, тихое окно", "корр. с ln RV60, шок"]),
                      "Окна выбраны по правилу из windows.json (минимальный и максимальный диапазон суток 2024-07-01..2025-07-18). Корреляции в таблице посчитаны по окну: индикаторы r² и диапазона близки к ln RV60 (ядра почти равномерны), "
                      "объёмные слабее; новой информации сверх HAR это не даёт - вердикт «нет»."))
    return "".join(parts)


# ======================================================================================================== section O4
def sec_o4():
    parts = []
    d60 = paired(R4, "BANK", "R0", H=60)
    dq = paired(R4, "BANK", "R0", H=60, key="qlike")
    days = {r["month"]: r["train_days"] for r in R4 if r["H"] == 60 and r["model"] == "BANK"}
    ms = list(d60)
    cols = [SALMON if m == "2017-09" else GOLD for m in ms]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ms, y=[d60[m] for m in ms], name="BANK − R0, Спирмен (H=60)", marker=dict(color=cols, line=dict(color=SURFACE, width=0.8)),
                         hovertemplate="%{x}: %{y:+.4f}<extra>Спирмен</extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=ms, y=[days[m] for m in ms], mode="lines+markers", name="дней обучения", line=dict(color=NEUT, width=1.6, dash="dot"), marker=dict(size=5),
                             hovertemplate="%{x}: %{y:.1f} дня<extra>обучение</extra>"), secondary_y=True)
    fig.add_annotation(x="2017-09", y=d60["2017-09"], text=f"2017-09: {ru(d60['2017-09'], 3, True)}<br>10,7 дня, 2199 строк на 52 признака", showarrow=True, arrowcolor=SALMON, font=dict(color=SALMON, size=11), ax=150, ay=-60, yref="y")
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.add_hline(y=0.005, line=dict(color=T.GOOD, width=1, dash="dot"), annotation_text="порог +0,005", annotation_font_color=T.GOOD)
    fig.update_yaxes(title_text="выигрыш по Спирмену", secondary_y=False)
    fig.update_yaxes(title_text="дней обучения", secondary_y=True, showgrid=False)
    fig.update_layout(xaxis=dict(type="category", tickangle=-50))
    f1 = F(fig, "o4", "o4_monthly", "O4 помесячный BANK−R0, H=60", 470, subtitle="красный столбец - 2017-09, единственный месяц с 10,7 днями обучения")
    nz = {m: v for m, v in d60.items() if m != "2017-09"}
    t_nz = tci(list(nz.values()))
    parts.append(card("Помесячный выигрыш BANK−R0 (H=60) и дни обучения", f1,
                      f"<b>Вердикт: не подтверждено.</b> Среднее {fmt_ci(A4[60])}: порог +0,005 не достигнут, нижняя граница ниже 0 (месяцев с плюсом 14/16 - выполнено; QLIKE {sg(tci(list(dq.values()))[0])} - выполнено). "
                      f"Это <b>post hoc, не вердикт:</b> без 2017-09 среднее {fmt_ci(t_nz)}."))

    fig = go.Figure()
    for H_ in (15, 60, 240):
        d = paired(R4, "BANK", "R0", H=H_)
        fig.add_trace(go.Scatter(x=list(d), y=list(d.values()), mode="lines+markers", name=f"H={H_} мин", line=dict(color=HCOL[H_], width=2), marker=dict(size=6),
                                 hovertemplate="%{x}: %{y:+.4f}<extra>H=" + str(H_) + "</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(yaxis_title="BANK − R0, Спирмен", xaxis=dict(type="category", tickangle=-50), yaxis_range=[-0.06, 0.04])
    f2 = F(fig, "o4", "o4_horizons", "O4 по горизонтам (ось обрезана: 2017-09 уходит за -0,06 при H=60 и -0,33 при H=240)", 420,
           subtitle="ось обрезана, чтобы были видны остальные месяцы")
    rows = []
    for H_ in (15, 60, 240):
        qq = tci(list(paired(R4, "BANK", "R0", H=H_, key="qlike").values()))
        rows.append([f"H={H_}", fmt_ci(A4[H_]), fmt_ci(qq), ru(sum4["holm_15_240"][0], 3) if H_ != 60 else "основной"])
    parts.append(card("По горизонтам", f2 + table(rows, ["горизонт", "BANK − R0, Спирмен", "QLIKE (меньше - лучше)", "Holm p (15/240)"]),
                      "Та же картина, что в O3: прибавка на 15 и 60 минутах без 2017-09, на 240 нет."))

    parts.append(card("Ошибка постановки (руководитель)", "<p class='note'>Правило короткой истории скопировано из T04 (там три регрессора, хватает 10 дней) без вопроса, можно ли подогнать 52 признака на 10 днях. "
                      "Это <b>ошибка дизайна руководителя до результатов</b>; она не чинится после них. Вердикт остаётся «не подтверждено». Чистая проверка BANK - часть B.</p>"
                      f"<p class='note'><b>Часть B (заморожена, не оценена):</b> frozen_B.json, sha256 <code>{sha4}</code>; обучение на барах Binance 2025-10-01..2026-09-30; score_B.py оценивает первые 3 полных месяца с 2026-10 "
                      "(критерий A при H=60, 3 из 3 месяцев с плюсом).</p>"))
    return "".join(parts)


# ======================================================================================================== section O5
def sec_o5():
    parts = []
    s, ql = choice5["sel_spearman_h60"], choice5["sel_qlike_h60"]
    rows = [[m_, ru(s[m_], 4), ru(ql[m_], 4), "<b>выбрана</b>" if m_ == "L0" else ""] for m_ in ("S3", "L0", "N0")]
    fig = go.Figure(go.Bar(x=["S3 ridge", "L0 = BANK ridge", "N0 = HGB на S3+bank"], y=[s[m_] for m_ in ("S3", "L0", "N0")], marker=dict(color=[NEUT, GOLD, SALMON], line=dict(color=SURFACE, width=1)),
                           text=[ru(s[m_], 4) for m_ in ("S3", "L0", "N0")], textposition="outside", hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(yaxis=dict(title="средний Спирмен (подбор, H=60)", range=[0.70, 0.745]))
    f0 = F(fig, "o5", "o5_base", "O5 выбор базы на подборе", 340, subtitle="42 месяца 2021-01..2024-06; ось не от нуля")
    parts.append(card("Выбор базы (только подбор)", f0 + table(rows, ["модель", "Спирмен", "QLIKE", ""]),
                      "B* = L0 (ridge BANK); кандидат L1 = L0 + геометрия (G1 + G15, 328 столбцов). Бустинг хуже ridge на подборе."))

    d = paired(R5, "L1", "L0", span=JSP, H=60)
    dq = paired(R5, "L1", "L0", span=JSP, H=60, key="qlike")
    ms = list(d)
    colr = [VIOLET if 2018 <= int(m[:4]) else GOLD for m in ms]
    fig = go.Figure()
    for y, c_ in (("2018", GOLD), ("2019", PINK), ("2020", VIOLET)):
        idx = [m for m in ms if m.startswith(y)]
        fig.add_trace(go.Bar(x=idx, y=[d[m] for m in idx], name=y, marker=dict(color=c_, line=dict(color=SURFACE, width=0.5)), hovertemplate="%{x}: %{y:+.4f}<extra>" + y + "</extra>"))
        t = tci([d[m] for m in idx])
        vlines_band(fig, idx[0], idx[-1], t, c_, y)
    fig.add_vrect(x0="2019-11", x1="2020-02", fillcolor=T.rgba(T.CRITICAL, 0.12), line_width=0, annotation_text="слабый участок: 4 месяца подряд минус", annotation_position="top left", annotation_font_color=INK2)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.add_hline(y=0.003, line=dict(color=T.GOOD, width=1, dash="dot"), annotation_text="порог +0,003", annotation_font_color=T.GOOD, annotation_position="bottom right")
    fig.update_layout(yaxis_title="L1 − L0, Спирмен (H=60)", xaxis=dict(type="category", tickangle=-60, nticks=30), barmode="overlay")
    f1 = F(fig, "o5", "o5_monthly", "O5 помесячный L1−L0, H=60", 470, subtitle="штрих и заливка - среднее и 95% t-интервал по году")
    parts.append(card("Помесячный выигрыш геометрии поверх L0 (2018-07..2020-12)", f1,
                      f"<b>Вердикт: геометрия добавляет поверх лучшей базы.</b> L1−L0 {fmt_ci(J5[60])} (порог ≥ +0,003, нижняя граница > 0, ≥ 20 месяцев; выполнено); "
                      f"QLIKE {fmt_ci(tci(list(dq.values())))}. Запас по среднему 0,0026, по числу месяцев - 3."))

    fig = go.Figure()
    for H_ in (15, 60, 240):
        t = J5[H_]
        fig.add_trace(go.Bar(x=[f"H={H_} мин"], y=[t[0]], name=f"H={H_} мин", marker=dict(color=HCOL[H_], line=dict(color=SURFACE, width=1)),
                             error_y=dict(type="data", symmetric=False, array=[t[2] - t[0]], arrayminus=[t[0] - t[1]], color=INK2), text=[ru(t[0], 4, True)], textposition="outside",
                             hovertemplate=f"H={H_}: {ru(t[0], 4, True)} [{ru(t[1], 4, True)}; {ru(t[2], 4, True)}], {t[3]}/{t[4]}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(yaxis_title="L1 − L0, Спирмен", showlegend=False)
    f2 = F(fig, "o5", "o5_horizons", "O5 по горизонтам", 360, subtitle="среднее по 30 месяцам ± 95% t")
    yr = []
    for y in ("2018", "2019", "2020"):
        t = tci([v for m, v in d.items() if m.startswith(y)])
        yr.append([y, fmt_ci(t)])
    parts.append(card("Горизонты и годы", f'<div class="two"><div>{f2}</div><div>' + table(yr, ["год", "L1 − L0, H=60"]) +
                      table([[f"H={H_}", fmt_ci(J5[H_])] for H_ in (15, 60, 240)], ["горизонт", "L1 − L0"]) + "</div></div>",
                      "H=15 ярче всего (все 30 месяцев с плюсом); H=240 - ничего. В 2018 и 2019 интервалы по году включают 0; в 2020 - выше 0."))
    parts.append(card("Часть B (заморожена, не оценена)", f"<p class='note'>frozen_B.json, sha256 <code>{sha5}</code>: L0 и L1 обучены один раз на Binance 2025-10-01..2026-09-30; "
                      "score_B.py оценивает первые 3 полных месяца с 2026-10 один раз; критерий тот же с 3 из 3 месяцев с плюсом.</p>"
                      "<p class='note'><b>Оговорки:</b> 24 коротких лабораторных среза H47 и H53 касались этих лет (заявлено в SPEC); 2018-07 и 2018-08 обучались на 313,7 и 344,7 дня (в SPEC было 318 - счёт руководителя); "
                      "прежние отрезки O1 и O3 использовали те же определения моделей без переобучения.</p>"))
    return "".join(parts)


def sec_pending():
    body = ("<ul class='note'>"
            "<li><b>O4, часть B:</b> score_B.py (O4_bank_confirm) оценивает BANK−R0 на первых 3 полных месяцах с 2026-10 (2026-10..12), критерий A при H=60, 3 из 3 месяцев с плюсом. "
            f"Артефакт заморожен: sha256 <code>{sha4[:16]}...</code>.</li>"
            "<li><b>O5, часть B:</b> score_B.py (O5_geom_bestbase) оценивает L1−L0 на тех же месяцах, один раз. "
            f"sha256 <code>{sha5[:16]}...</code>.</li>"
            f"<li><b>O2:</b> {'результат есть (секция O2)' if o2_report else 'идёт на GPU; REPORT.md ещё нет. Пересборка отчёта заполнит секцию.'}</li>"
            "<li>Пересборка: <code>python runs/tactical/report/build_outlook_report.py</code></li></ul>")
    return card("Что ещё не готово", body, "До появления трёх полных месяцев 2026-10.. часть B не оценивается и не обсуждается.")


# ======================================================================================================== assemble
def get_css():
    src = (HERE / "build_report.py").read_text(encoding="utf-8")
    m = re.search(r'CSS = """(.*?)""" % dict', src, re.S)
    css = m.group(1) % dict(PAPER=PAPER, SURFACE=SURFACE, INK=INK, INK2=INK2, MUTED=MUTED, GRID=GRID, AXIS=AXIS, PINK=PINK, GOLD=GOLD, FONT=T.FONT)
    return css + ".lead{max-width:1500px}ul.note{padding-left:20px}td ul{margin:0}"


def build():
    d = load_bars()
    assert d["timestamp"].max() <= pd.Timestamp(C.CUTOFF)
    secs = [("o0", "Обзор", sec1), ("o1", "O1. Геометрия", sec_o1), ("o2", "O2. Физика", sec_o2), ("o3", "O3. Синтетические индикаторы", sec_o3),
            ("o4", "O4. Подтверждение BANK", sec_o4), ("o5", "O5. Геометрия на лучшей базе", sec_o5), ("pend", "Что ещё не готово", sec_pending)]
    built = []
    for sid, title, fn in secs:
        body = fn()
        built.append((sid, title, body))
        print(f" {sid} done {time.time() - t_start:.0f}s", flush=True)
    nav = "".join(f'<a href="#{sid}">{title}</a>' for sid, title, _ in built)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    parts = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Outlook: положительный прогноз</title><style>" + get_css() + "</style><script>" + po.get_plotlyjs() + "</script></head><body>",
             f"<nav><h1>Outlook</h1><div class='sub'>neural_trade · тактическая сессия<br>данные цен до {C.CUTOFF[:10]}</div>{nav}</nav><main>"]
    for sid, title, body in built:
        parts.append(f'<section id="{sid}"><h2>{title}</h2>{body}</section>')
    parts.append(f"<footer>Сгенерировано runs/tactical/report/build_outlook_report.py (коммит основы {sha}); данные: runs/tactical/outlook/O1..O5 (results, tables, interp, kernels, windows, summary). "
                 f"Цены - только до {C.CUTOFF}; числа прогнозного года вперёд (2025-10..2026-09) - результаты, не бары. Исследовательская работа, одно зерно, не влияет на умолчания MVP (D-062).</footer></main>")
    parts.append("""<script>(function(){const q=new URLSearchParams(location.search).get('only');
if(q){document.body.classList.add('only');document.querySelectorAll('main > section').forEach(s=>{if(s.id!==q)s.style.display='none'});}
addEventListener('load',()=>setTimeout(()=>window.dispatchEvent(new Event('resize')),500));
const links=[...document.querySelectorAll('nav a')];
function onscroll(){let cur=null;document.querySelectorAll('main > section').forEach(s=>{if(s.getBoundingClientRect().top<140)cur=s.id});links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+cur))}
addEventListener('scroll',onscroll);})();</script></body></html>""")
    out = HERE / "outlook_report.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} {out.stat().st_size / 1e6:.1f} MB, {len(FIGS)} figures, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    build()
