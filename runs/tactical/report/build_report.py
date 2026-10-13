"""Builds runs/tactical/report/tactical_report.html: ONE self-contained interactive report of the tactical session (owner,
2026-10-13). Plotly is embedded once; no external requests. Text is Russian. Every number comes from the committed result files
(runs/tactical/webdash/out/*.json, written by webdash/export.py and export2.py, and the files they cite) or is computed here from the
cut price file (hard cutoff 2025-07-20: runs/tactical/common/cutoff.py).

usage: python runs/tactical/report/build_report.py   (CPU, about 1-2 minutes)   -> tactical_report.html
Screenshots of every section (headless Edge, `?only=sN`): python runs/tactical/report/shoot.py"""
import base64
import html as H
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcommon import (T, C, TAC, ROOT, UP, DOWN, PINK, GOLD, TAN, VIOLET, SALMON, INK, INK2, MUTED, GRID, SURFACE, PAPER, AXIS, S,  # noqa: E402
                     jload, jfile, load_bars, pick_windows, span_text, fig_html, badge, jbadge, table, card, ru)
import geom_figs as G  # noqa: E402
import fut_sections as FS  # noqa: E402

t_start = time.time()
FIGS = []            # (section, key, title) of every figure, for the report's list
NEUT = T.NEUTRAL


def reg(sec, key, title):
    FIGS.append((sec, key, title))


def F(fig, sec, key, title, height, lift=0, subtitle=None):
    """Register and render one figure (plotly div)."""
    reg(sec, key, title)
    return fig_html(fig, key, height=height, lift=lift, subtitle=subtitle)


def pct(x, nd=1):
    return ru(x * 100, nd) + "%"


# ======================================================================================================== data
decay = jload("decay_monthly")
regimes = jload("regimes")
dmap = jload("decay_map")
verdicts = jload("verdicts")
geom = jload("geometry")
frontier = jload("frontier")
journal = jload("journal")
models = jload("models")
longtrain = jload("longtrain")
hdev = jload("hourly_dev")
hfin = jload("hourly_final")
# H81 (hill-climb v2 lockbox) is not in verdicts.json (written by export2.py): its row is added here from the committed result file
_lb = jfile("hc2/lockbox_result.json")
from scipy import stats as _sps  # noqa: E402
_mm = np.asarray(_lb["winner"]["lockbox"]["monthly_net_bps"], float)
_lo81 = float(_mm.mean() - _sps.t.ppf(0.975, len(_mm) - 1) * _mm.std(ddof=1) / np.sqrt(len(_mm)))
verdicts = verdicts + [{"id": "H81-19", "h": "H81", "name": "Hill-climb v2: победитель на сейфе", "unit": "Нижняя граница 95% интервала среднего месяца на сейфе, б.п. (12 месяцев)",
                        "effect": _lo81, "threshold": 0.0, "ratio": 1.0 if _lo81 > 0 else -1.0, "verdict": "прошла" if _lb["PASS"] else "не прошла"}]
PASSED = lambda v: v["verdict"].startswith("прошла")
raw_n = {r["month"]: r["n_top10"] for r in jfile("recut/lab/decay.json")}

# the English journal text of every row (LOG.md), for the searchable table
LOG = {}
for line in open(TAC / "LOG.md", encoding="utf-8"):
    m = re.match(r"\| (H\d+) \| (\d{4}-\d\d-\d\d) \| (.*) \|$", line.rstrip("\n"))
    if m:
        parts = m.group(3).split(" | ")
        LOG[m.group(1)] = {"q": parts[0], "res": parts[-2] if len(parts) >= 3 else "", "ev": parts[-1]}


def year_mean(rows, key, y0, y1):
    v = [r[key] for r in rows if y0 <= int(r["month"][:4]) <= y1 and r.get(key) is not None]
    return float(np.mean(v)), len(v)


# ======================================================================================================== section 1
def sec1():
    a_old, _ = year_mean(decay, "auc_nextopen", 2018, 2020)
    a_new, _ = year_mean(decay, "auc_nextopen", 2025, 2025)
    c_old, _ = year_mean(decay, "auc_close", 2018, 2020)
    c_new, _ = year_mean(decay, "auc_close", 2025, 2025)
    h_old, _ = year_mean(decay, "har_spearman", 2018, 2018)
    h_new, _ = year_mean(decay, "har_spearman", 2025, 2025)
    eq1h = json.load(open(TAC / "equity" / "b_h56_1h_rich_l1_365.json", encoding="utf-8"))["levels"]["0"]["stats"]
    npass = sum(1 for v in verdicts if PASSED(v))
    cnt = defaultdict(int)
    for r in journal:
        cnt[r["verdict"]] += 1
    tiles = [
        (f"{ru(c_old, 3)} → {ru(c_new, 3)}", "AUC направления, вход по закрытию", f"среднее 2018-20 → 2025; при входе по следующему открытию {ru(a_old, 3)} → {ru(a_new, 3)}"),
        (f"{ru(h_old, 2)} → {ru(h_new, 2)}", "Спирмен прогноза волатильности (HAR)", "2018 → 2025: волатильность стала предсказуемее, направление - наоборот"),
        ("0", "стратегий в цели hit > 60% и просадка < 5%", f"лучшая честная часовая: hit {pct(eq1h['hit'])}, просадка {ru(eq1h['max_drawdown_pct'], 1)}%"),
        (f"{npass} из {len(verdicts)}", "порогов исследования H67-H86 пройдено", "все пройденные - про волатильность или физику на оси дисперсии: H72, H77 и на новых данных H83 (прогноз), H85 (интервалы)"),
        ("-16,9%", "кандидат H62 вперёд (H82, Binance 2025-11..2026-09)", "hit 50,3%, 437 сделок, просадка 26% при пределе 15%; на подборе было hit 59%: преимущество исчезло"),
        ("+0,142", "скилл прогноза изменения волатильности на новых данных (H83)", "12 из 12 месяцев; на взгляде было +0,163. Единственный результат сессии, переживший невиданные данные"),
        (f"{len(journal)}", "гипотез в журнале H5-H86", f"эффект {cnt['effect']}, частично {cnt['partial']}, нет эффекта {cnt['none']}, не подтвердилось {cnt['failed']}, справка {cnt['info']}"),
        ("0,521", "AUC направления сейчас (с 2024-09)", "ступень решения H72: было 0,545 (2021-03..2024-08) и около 0,57 (2018-21)"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="big">{a}</div><div class="lab">{b}</div><div class="sub">{c}</div></div>' for a, b, c in tiles)
    areas = defaultdict(lambda: defaultdict(int))
    for r in journal:
        areas[r["area"]][r["verdict"]] += 1
    order = sorted(areas, key=lambda a: -sum(areas[a].values()))
    fig = go.Figure()
    for v, nm, col in (("effect", "&#10004; есть эффект", T.GOOD), ("partial", "&#9680; частично", T.WARNING), ("none", "&#10008; эффекта нет", T.CRITICAL),
                       ("failed", "&#10008; не подтвердилось", T.SERIOUS), ("info", "&#9679; справка", NEUT)):
        xs = [areas[a][v] for a in order]
        fig.add_trace(go.Bar(y=order, x=xs, orientation="h", name=nm, marker=dict(color=col, line=dict(color=SURFACE, width=1.5)),
                             text=[str(x) if x else "" for x in xs], textposition="inside", insidetextanchor="middle", textfont=dict(color=PAPER, size=11),
                             hovertemplate="%{y}: " + nm + " - %{x}<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis_title="число гипотез", yaxis=dict(autorange="reversed"))
    reg("s1", "verdict_areas", "Журнал по областям и вердиктам")
    body = (f'<div class="tiles">{tiles_html}</div>'
            '<div class="lead"><b>Итог.</b> Предсказуемость <i>направления</i> цены на коротких горизонтах была реальной в 2017-2020 годах и затухла: '
            'после 2024-09 она около 0,52 AUC против 0,57-0,60 раньше, а половина старого перевеса оказалась отскоком между bid и ask при входе по закрытию. '
            'Предсказуемость <i>волатильности</i> за то же время выросла (Спирмен HOLD → HNEW). Ни одна из проверенных стратегий не достигла цели '
            'владельца (hit &gt; 60% при просадке &lt; 5%): честная часовая стратегия даёт hit около 57% и просадку около 9-10%, а просадка &lt; 5% '
            'достигается только плечом около 0,4x. Геометрия индикаторов (наклоны, пересечения, оболочки, кривизна, площади Леви, фазовые портреты, '
            'Вассерштейн, Фишер-Рао) и синтетические индикаторы из сети не добавили направления ни к регрессии, ни к сети; из семи неизведанных направлений '
            'исследования порог прошли два, и оба - про волатильность. Проверка на будущем (раздел 8): победитель большого подбора H81 проваливает сейф, а замороженный кандидат H62 на невиданных данных Binance 2025-11..2026-09 даёт hit 50%, итог -16,9% и просадку 26% (H82). Из волатильностного трека (раздел 9) на новых данных держатся прогноз изменения волатильности на час (H83) и 90%-интервалы доходности по HAR (H85: интервальная оценка -9,4 б.п. против скользящей RV, 11 из 11 месяцев); переключение экспозиции BTC по волатильности (H84) и 15-минутный календарь (H86) не подтвердились. Отрицательный, оформленный с доказательствами ответ - допустимый результат (VISION).</div>').replace('HOLD', ru(h_old, 2)).replace('HNEW', ru(h_new, 2))
    body += card(f"Все {len(journal)} гипотез журнала: области и вердикты", F(fig, "s1", "verdict_areas", "Журнал", 380),
                 "Цвет и значок в легенде вместе обозначают вердикт (число - гипотез); полный список с поиском - в разделе 10.")
    return body


# ======================================================================================================== section 2
def sec2():
    mths = [r["month"] for r in decay]
    xs = [f"{m}-01" for m in mths]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.46, 0.27, 0.27],
                        subplot_titles=("Направление: AUC по месяцам (обучение 60 дней, стрелка на 15 минут)", "Волатильность: Спирмен HAR-прогноза с |r| за 15 минут",
                                        "Топ-10% самых уверенных сигналов: доля верных, вход по следующему открытию"))
    # chance band: n_eff ~ n_test * stride(10) / horizon(15), n_test ~ 10 * n_top10
    band = [1.96 / np.sqrt(3.0 * max(raw_n.get(m, 400) * 10 * 10 / 15.0, 100)) for m in mths]
    fig.add_trace(go.Scatter(x=xs, y=[0.5 + b for b in band], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[0.5 - b for b in band], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(137,135,129,0.25)",
                             name="шум: 95% интервал при чистой случайности (оценка)", hovertemplate="шум ±%{customdata:.3f}<extra></extra>",
                             customdata=band), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_close"] for r in decay], mode="lines+markers", line=dict(color=PINK, width=1.5), marker=dict(size=5, symbol="circle"),
                             name="вход по закрытию сигнального бара (H51)", hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra>вход по закрытию</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_nextopen"] for r in decay], mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=6, symbol="square"),
                             name="вход по следующему открытию (H59, честный)", hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra>вход по открытию</extra>"), row=1, col=1)
    # regime steps
    for i, sg in enumerate(regimes):
        x0 = f"{sg['from']}-01"
        x1 = (pd.Period(sg["to"], "M") + 1).start_time.strftime("%Y-%m-%d")
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(242,241,236,0.05)" if i % 2 == 0 else "rgba(242,241,236,0.0)", line_width=0, row="all", col=1)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[sg["mean"]] * 2, mode="lines", line=dict(color=GOLD, width=4), name="ступень (среднее режима)", legendgroup="reg",
                                 showlegend=(i == 0), hovertemplate=f"режим {sg['from']} .. {sg['to']}: среднее AUC {sg['mean']:.4f}, {sg['n']} мес.<extra></extra>"), row=1, col=1)
        fig.add_annotation(x=x0, y=sg["mean"], text=f"{ru(sg['mean'], 3)}", showarrow=False, yshift=14, xanchor="left", font=dict(size=11, color=GOLD), bgcolor="rgba(18,18,17,0.75)", row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"), row=1, col=1)
    # HAR
    har = [r["har_spearman"] for r in decay]
    fig.add_trace(go.Scatter(x=xs, y=har, mode="lines+markers", line=dict(color=TAN, width=2), marker=dict(size=5), name="Спирмен HAR (волатильность)",
                             hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra>HAR</extra>"), row=2, col=1)
    for y in range(2017, 2026):
        v, n = year_mean(decay, "har_spearman", y, y)
        fig.add_trace(go.Scatter(x=[f"{y}-01-01", f"{y + 1}-01-01"], y=[v, v], mode="lines", line=dict(color=INK2, width=2, dash="dash"), name="среднее за год",
                                 legendgroup="ym", showlegend=(y == 2017), hovertemplate=f"{y}: среднее {v:.3f} ({n} мес.)<extra></extra>"), row=2, col=1)
    # hit
    fig.add_trace(go.Scatter(x=xs, y=[r["hit10_nextopen"] for r in decay], mode="lines+markers", line=dict(color=VIOLET, width=1.5), marker=dict(size=5, symbol="square"),
                             name="доля верных в топ-10%", hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra>топ-10%</extra>"), row=3, col=1)
    fig.add_hline(y=0.6, line=dict(color=T.CRITICAL, width=1.5, dash="dash"), annotation_text="цель владельца 60%", annotation_position="top right", row=3, col=1)
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_yaxes(title_text="Спирмен", row=2, col=1)
    fig.update_yaxes(title_text="hit", row=3, col=1)
    fig.update_layout(hovermode="x unified")
    reg("s2", "decay", "Затухание: AUC направления, ступени, Спирмен HAR, hit")
    seg_rows = [[f"{s['from']} .. {s['to']}", s["n"], ru(s["mean"], 4)] for s in regimes]
    rows = []
    for y in range(2017, 2026):
        a, n = year_mean(decay, "auc_close", y, y)
        b, _ = year_mean(decay, "auc_nextopen", y, y)
        h, _ = year_mean(decay, "har_spearman", y, y)
        t10, _ = year_mean(decay, "hit10_nextopen", y, y)
        rows.append([y, n, ru(a, 3), ru(b, 3), ru(h, 3), pct(t10)])
    body = card("Затухание направления и рост предсказуемости волатильности, помесячно",
                F(fig, "s2", "decay", "", 1020, lift=44),
                "Серая полоса вокруг 0,5 - 95% интервал шума AUC за месяц (оценка по числу якорей; соседние метки делят бары, n_eff ≈ N/горизонт). "
                "Золотые ступени - режимы из теста разрывов H72 (BIC выбрал 5 разрывов против гладкого тренда, ΔBIC +41,8; sup-Chow p = 0,004), "
                "подобранные на AUC при входе по следующему открытию. Ни одно внешнее событие (нулевые комиссии, FTX, халвинги, ETF) не объясняет разрывы.")
    body += '<div class="two">' + card("Режимы AUC (ступени)", table(seg_rows, ["период", "месяцев", "среднее AUC"])) + \
            card("По годам (средние по месяцам)", table(rows, ["год", "мес.", "AUC, закрытие", "AUC, открытие", "Спирмен HAR", "hit топ-10%"])) + "</div>"
    return body


# ======================================================================================================== section 3
def sec3():
    cells = []
    for r in dmap:
        if r["cell"] not in cells:
            cells.append(r["cell"])
    years = sorted({r["year"] for r in dmap})
    z = [[None] * len(years) for _ in cells]
    mo = [[0] * len(years) for _ in cells]
    for r in dmap:
        z[cells.index(r["cell"])][years.index(r["year"])] = r["auc"]
        mo[cells.index(r["cell"])][years.index(r["year"])] = r["months"]
    txt = [[("" if v is None else f"{v:.3f}".replace(".", ",")) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=years, y=cells, customdata=mo, text=txt, texttemplate="%{text}", textfont=dict(size=10, color=INK),
                               colorscale=[[0, SURFACE], [0.35, "#3a2a35"], [1, PINK]], zmin=0.49, zmax=0.65, xgap=2, ygap=2,
                               colorbar=dict(title="AUC"), hovertemplate="%{y}, %{x}: AUC %{z:.3f} (месяцев %{customdata})<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="размер бара / горизонт (в барах)")
    fig.update_xaxes(title_text="год", type="category")
    body = card("Карта затухания: размер бара × горизонт × год (H52, пересчёт с отсечкой)", F(fig, "s3", "decay_map", "", 120 + 34 * len(cells)),
                "Каждая клетка - среднее месячных AUC направления (скользящая модель tb7, обучение 60 дней, вход по закрытию). Затухло везде и на всех масштабах; "
                "самые тёмные клетки - 2024-2025. Случайный уровень - 0,5; шум одного месяца около ±0,02, года - примерно в 3,5 раза меньше. Число месяцев - в подсказке.")
    return body


# ======================================================================================================== section 4: geometry
def geom_result(name_prefixes, extra_text=None):
    """A card with the measured results of the geometry rows whose name matches, a forest-style small table and the journal text."""
    rows = []
    for r in geom:
        if any(r["name"].startswith(p) for p in name_prefixes):
            if r["lo"] is None:
                ci = "ДИ не сохранён"
                vd = badge("info", "точка без интервала" + ("" if r["effect"] >= 0 else ", ниже базы"))
            else:
                ci = f"[{ru(r['lo'], 4, True)}; {ru(r['hi'], 4, True)}]"
                vd = badge("good", "выше базы") if r["lo"] > 0 else badge("bad", "значимо хуже базы") if r["hi"] < 0 else badge("info", "не отличимо от нуля")
            rows.append([r["h"], r["name"], ru(r["effect"], 4, True), ci, r["base"], vd])
    t = table(rows, ["гипотеза", "вариант", "Δ AUC направления", "95% ДИ", "база сравнения", "вывод"])
    return t


def nf_table(fam, ws):
    cols = []
    for w in ws:
        cols.append(NF_FN[fam](w["W"][None].astype(np.float64))[0])
    names = G.NF.NAMES[fam]
    rows = [[n] + [ru(float(c[i]), 3, True) for c in cols] for i, n in enumerate(names)]
    return table(rows, ["признак"] + [w["label"] for w in ws])


NF_FN = G.NF.FN


def jrow(h):
    r = next(x for x in journal if x["id"] == h)
    return f'{jbadge(r["verdict"])} <b>{r["id"]}</b>: {H.escape(r["test"])} - {H.escape(r["key"])}'


def blk(anchor, title, what, figs, tabs, result, jids):
    j = "".join(f"<p class='jr'>{jrow(h)}</p>" for h in jids)
    return (f'<div class="gblock" id="{anchor}"><h3>{title}</h3><p class="note">{what}</p>{figs}'
            f'<div class="two"><div class="card"><h4>Измерено на этих трёх окнах (код проверенной геометрии)</h4>{tabs}</div>'
            f'<div class="card"><h4>Результат теста на всех данных</h4>{result}{j}</div></div></div>')


def sec4(ws):
    wrows = [[w["label"], span_text(w), ru(w["sd_bps"], 1), ru(w["er"], 2), ru(w["net_bps"], 0)] for w in ws]
    intro = ('<p class="note">Каждая геометрия, проверенная за сессию, нарисована на одних и тех же трёх реальных 60-минутных окнах BTC/USDT (1-минутные бары, '
             'все до отсечки 2025-07-20). Окна выбраны правилом, а не вручную: спокойное - волатильность на 8-м процентиле; трендовое - наибольшее |чистое смещение| / путь '
             'среди окон с не меньшей медианной волатильностью; волатильное - максимум волатильности. Построения повторяют определения из кода, который тестировался; '
             'числа в таблицах рядом - выход этого же кода на окне. Справа - результат теста на всех данных: прирост AUC направления (95% интервал) к базе.</p>')
    intro += table(wrows, ["окно", "время (UTC)", "σ бара, б.п.", "|смещение| / путь", "чистое смещение, б.п."])
    out = [intro]
    # ---------------- H24 family
    figs = ""
    tabs = ""
    for i, w in enumerate(ws):
        f, tab = G.h24_figure(w)
        figs += f"<h4>{w['label']} окно: {span_text(w)}</h4>" + F(f, "s4", f"h24_{i}", f"H24 {w['label']}", 930, lift=34)
        tabs += f"<h5>{w['label']}</h5>" + tab
    what = ("<b>Индикаторная геометрия (H24, H43, H45, H47).</b> Слой <code>IndicatorGeometry</code> читает последовательность каждого канала: наклоны за 3 и 10 баров "
            "(отрезки: штриховой за 10, сплошной за 3, нормированы волатильностью канала), расстояние цены до канала (стрелка, asinh), пересечения за последние 5/10/20 баров "
            "(треугольники, чем крупнее - тем свежее), сжатие σ(10 баров)/σ(60) (серая полоса справа). Нарисованы EMA20, SMA20, полосы Боллинджера, RSI14, MACD 12/26/9 и Stoch 14 "
            "с <i>учебниковыми</i> периодами (как в H45/H47). Для осцилляторов пересечения - с линией 50 / сигналом / %D (соглашение рисунка).")
    out.append(blk("g_h24", "H24 / H43 / H45 / H47: геометрия индикаторов (наклоны, расстояние, пересечения, сжатие)", what, figs, tabs,
                   geom_result(["Геометрия обучаемых", "Геометрия + индикаторы", "Геометрия 5", "164 признака"]), ["H24", "H43", "H45", "H47"]))
    # ---------------- N1
    f1, pre = G.n1_figure(ws)
    tabs = "".join(f"<h5>N1</h5>" if False else "" for _ in [0]) + nf_table("N1", ws)
    out.append(blk("g_n1", "N1: кривизна и кручение траектории индикаторов",
                   "Последние 20 баров как кривая в 3-мерном пространстве (close−EMA20, RSI14, гистограмма MACD), каждая ось в σ окна. Цвет - кривизна κ (угол поворота / длина шага) "
                   "в вершине; розовый отрезок - касательная в конце; ромб - начало. Признаки: средняя и конечная кривизна, конечное кручение τ, суммарный поворот, извилистость (длина дуги / хорда).",
                   F(f1, "s4", "n1", "N1 кривизна и кручение, 3D", 600, lift=30), tabs, geom_result(["N1"]), ["H49"]))
    # ---------------- N2
    f2 = G.n2_figure(ws, 30)
    out.append(blk("g_n2", "N2: площади Леви",
                   "Двумерный путь последних 30 баров: по оси X накопленная лог-доходность, по Y - накопленный знаковый объём (верхний ряд) или RSI (нижний). Площадь между путём и хордой "
                   "(пунктир) разложена на треугольники от первой точки: фиолетовая штриховка / - положительная часть, жёлтая \\ - отрицательная (знак - и цветом, и узором). "
                   "Признак - 2A / (TVx·TVy) в [−1; 1].", F(f2, "s4", "n2", "N2 площади Леви", 760, lift=34), nf_table("N2", ws), geom_result(["N2"]), ["H49"]))
    # ---------------- N3
    f3 = G.n3_figure(ws)
    out.append(blk("g_n3", "N3: фазовый портрет (скорость и ускорение)",
                   "EMA5 закрытия: скорость v (шаг) и ускорение a (шаг скорости), каждая в своих σ окна. Путь со стрелками времени, квадранты затенены, звезда - последний бар; угол θ = atan2(a, v), "
                   "суммарная намотка - сумма поворотов угла за последние 30 баров.", F(f3, "s4", "n3", "N3 фазовый портрет", 520, lift=30), nf_table("N3", ws), geom_result(["N3"]), ["H49"]))
    # ---------------- N4
    f4 = G.n4_figure(ws)
    out.append(blk("g_n4", "N4: выпуклая оболочка (поддержка и сопротивление)",
                   "Верхняя оболочка по high и нижняя по low баров 0..58; последнее ребро каждой продолжено до бара 59 как линия сопротивления / поддержки, звезда - закрытие и его положение между линиями "
                   "(0 - поддержка, 1 - сопротивление; вне [0; 1] - цена вне коридора).", F(f4, "s4", "n4", "N4 оболочка", 560, lift=60), nf_table("N4", ws), geom_result(["N4"]), ["H49"]))
    # ---------------- N5
    fe, fh = G.n5_figures(ws)
    out.append(blk("g_n5", "N5: расстояние Вассерштейна внутри окна",
                   "Распределения 1-барных доходностей: первая половина против второй и последние 15 баров против предыдущих 44. Эмпирические функции распределения (ступени) и гистограммы; "
                   "закрашенная площадь между ECDF равна W1 в б.п. (по 40 серединным квантилям, как в коде признака).",
                   F(fe, "s4", "n5_ecdf", "N5 ECDF", 640, lift=34) + F(fh, "s4", "n5_hist", "N5 гистограммы", 560, lift=34), nf_table("N5", ws), geom_result(["N5"]), ["H49"]))
    # ---------------- N6
    f6 = G.n6_figure(ws)
    out.append(blk("g_n6", "N6: путь Фишера-Рао на полуплоскости Пуанкаре",
                   "Шесть подокон по 10 баров дают пары (μ, σ) доходностей (в σ всего окна); точки на полуплоскости (μ/√2, σ), розовые дуги - геодезические между соседними подокнами, "
                   "пунктир - прямая геодезическая от первого к последнему. Расстояние Фишера-Рао = √2 · гиперболическое расстояние. Признаки: длина пути, чистое смещение.",
                   F(f6, "s4", "n6", "N6 Фишер-Рао", 520, lift=30), nf_table("N6", ws), geom_result(["N6"]), ["H49"]))
    # ---------------- N7
    f7 = G.n7_figure(ws)
    out.append(blk("g_n7", "N7: выпуклость (контроль, известен на акциях)",
                   "Параболы, подогнанные к последним 15 / 30 / 60 лог-ценам (в барах), показаны на цене; признак - квадратичный коэффициент / σ 1-барных доходностей. Это контрольная гипотеза: перенос с акций.",
                   F(f7, "s4", "n7", "N7 выпуклость", 520, lift=30), nf_table("N7", ws), geom_result(["N7"]), ["H49"]))
    # ---------------- H67
    f67, st = G.h67_figure(ws)
    rows67 = [[w["label"], ru(a, 2), ru(b, 2), ru(c, 2, True)] for w, (a, b, c) in zip(ws, st)]
    sel = jfile("shape/results/sel_result.json")["table"]
    r60 = [[r["target"], r["model"], ru(r["mean"], 4, True), f"[{ru(r['lo95'], 4, True)}; {ru(r['hi95'], 4, True)}]", ru(r["p_holm"], 3)] for r in sel if r["T"] == 60]
    out.append(blk("g_h67", "H67: форма пути следующего часа в «своих часах»",
                   "Цель - не знак, а форма пути СЛЕДУЮЩЕГО часа: ρ = |чистое смещение| / размах, VR = (Σr)²/Σr², Z2 = сумма попарных произведений доходностей (лаг ≥ 2) / Σr². Сверху - путь по минутам, "
                   "снизу - тот же путь по «собственным часам» (по накопленной доле квадратичной вариации); серая полоса - 90% коридор броуновского пути с той же дисперсией. "
                   "Гипотеза: форму пути можно предсказать сверх волатильности.", F(f67, "s4", "h67", "H67 форма пути", 760, lift=34),
                   table(rows67, ["окно", "ρ", "VR", "Z2"]),
                   "<b>Прирост Спирмена над сильной базой волатильности, T = 60 (подбор 2019-2024):</b>" + table(r60, ["цель", "модель", "прирост", "95% ДИ", "p (Холм)"]), ["H67"]))
    # ---------------- H80
    f80, hm = G.h80_figure(ws)
    syn = open(TAC / "synth" / "report.txt", encoding="utf-8").read()
    arms = {}
    for blkname in ("lab", "wf"):
        m = re.search(r"=== %s:.*?\n(.*?)(?==== |\Z)" % blkname, syn, re.S)
        sect = m.group(1)
        cur = None
        for ln in sect.splitlines():
            if ln.startswith("-- dir"):
                cur = "dir"
            elif ln.startswith("-- vol"):
                cur = "vol"
            else:
                mm = re.match(r"\s+(\w+)\s+([\d.]+) \[([\d.]+),([\d.]+)\]", ln)
                if mm and cur:
                    arms.setdefault(mm.group(1), {})[(blkname, cur)] = (float(mm.group(2)), float(mm.group(3)), float(mm.group(4)))
    rows80 = []
    for a in ("tb7", "c3", "C1", "C2a", "C2b", "B", "A", "har"):
        if a not in arms:
            continue
        def f(k):
            v = arms[a].get(k)
            return "-" if not v else f"{ru(v[0], 3)} [{ru(v[1], 3)}; {ru(v[2], 3)}]"
        rows80.append([a, f(("lab", "dir")), f(("wf", "dir")), f(("lab", "vol")), f(("wf", "vol"))])
    f80html = (F(f80, "s4", "h80", "H80 синтетические ряды (случайная TCN, контроль C1)", 600, lift=34) + "<h4>Выход геометрического слоя по 12 синтетическим рядам</h4>" +
               F(hm, "s4", "h80_heat", "H80 признаки геометрии по рядам", 330, lift=30))
    out.append(blk("g_h80", "H80: синтетические индикаторы, выученные сетью, и геометрия поверх них",
                   "Рядов от сети в этой выкладке нет: веса обученного генератора не сохранены, поэтому показан <b>контроль C1 - случайные ряды</b> от случайной причинной TCN (K = 12, веса случайны, не обучены); "
                   "поверх них та же геометрия (наклоны, пересечения с ценой, сжатие). Именно такую геометрию на случайных рядах и сравнивали с обученными: B (генератор предобучен, заморожен) = случайные ряды. "
                   "Сверху - 6 из 12 рядов в z-оценках вместе с ценой; снизу - тепловая карта признаков по 12 рядам (цвет - в долях максимума строки, число - значение).",
                   f80html, "<p>Первичный эксперимент: 24 лабораторных среза + помесячная проверка 2021-01..2024-06.</p>" + table(rows80, ["рука", "направление, лаборатория AUC [ДИ]", "направление, помесячно", "волатильность, лаборатория (Спирмен)", "волатильность, помесячно"]),
                   geom_result(["Синтетические"]), ["H80"]))
    # ---------------- forest of all geometry
    rows = list(geom)
    fig = go.Figure()
    ys = [f"{r['h']} · {r['name']} ({'к сети' if 'сети' in r['base'] else 'к рег.'})" for r in rows]
    for i, r in enumerate(rows):
        if r["lo"] is None:
            fig.add_trace(go.Scatter(x=[r["effect"]], y=[ys[i]], mode="markers", marker=dict(symbol="circle-open", size=12, color=INK2, line=dict(color=INK2, width=2)), showlegend=False,
                                     hovertemplate=f"{ys[i]}<br>Δ AUC {r['effect']:.4f} (интервал не сохранён)<br>база: {r['base']}<extra></extra>"))
        else:
            if r["lo"] > 0:
                col, sym = T.GOOD, "diamond"
            elif r["hi"] < 0:
                col, sym = T.CRITICAL, "x"
            else:
                col, sym = NEUT, "circle"
            fig.add_trace(go.Scatter(x=[r["effect"]], y=[ys[i]], mode="markers", marker=dict(symbol=sym, size=11, color=col, line=dict(color=INK, width=1)),
                                     error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["effect"]], arrayminus=[r["effect"] - r["lo"]], color=col, thickness=2, width=6),
                                     showlegend=False, hovertemplate=f"{ys[i]}<br>Δ AUC {r['effect']:.4f} [{r['lo']:.4f}; {r['hi']:.4f}]<br>база: {r['base']}<extra></extra>"))
    for nm, col, sym in (("точка без ДИ (интервал не сохранён)", INK2, "circle-open"), ("ДИ покрывает 0", NEUT, "circle"), ("значимо хуже базы", T.CRITICAL, "x"), ("значимо лучше базы", T.GOOD, "diamond")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(symbol=sym, size=11, color=col)))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1.5, dash="dash"))
    fig.add_vline(x=0.005, line=dict(color=GOLD, width=1.5, dash="dot"), annotation_text="минимальный эффект +0,005", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="прирост AUC направления к базе (регрессия или базовая сеть), 95% ДИ")
    forest = card("Все варианты геометрии вместе: прирост AUC направления", F(fig, "s4", "forest", "", 120 + 34 * len(rows), lift=0),
                  "Ни один вариант не имеет интервала целиком выше нуля; N1, N2, N6 значимо хуже базы (вклад отрицательный, обычное следствие лишних признаков). Точки без интервала помечены в подсказке.")
    out.append(forest.replace('<div class="card">', '<div class="card" id="forest">', 1))
    return "".join(out)


# ======================================================================================================== section 5
def sec5():
    vs = verdicts
    names = [f"{v['h']} · {v['name']}" for v in vs]
    ratio = [v["ratio"] for v in vs]
    fig = go.Figure()
    for i, v in enumerate(vs):
        ok = PASSED(v)
        col, sym = (T.GOOD, "diamond") if ok else (T.CRITICAL, "x")
        zero = v["threshold"] == 0
        fig.add_trace(go.Bar(y=[names[i]], x=[ratio[i]], orientation="h", marker=dict(color=col, opacity=0.55, line=dict(color=col, width=1)), showlegend=False,
                             hovertemplate=f"{v['unit']}<br>эффект {v['effect']:+.4g}, порог {v['threshold']:.4g}" + (" (порог 0: показан знак)" if zero else "") +
                                           f"<br>вердикт: {v['verdict']}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[names[i]], x=[ratio[i]], mode="markers", marker=dict(symbol=sym, size=13, color=col, line=dict(color=INK, width=1)),
                                 showlegend=False, hoverinfo="skip"))
    for nm, col, sym in (("порог пройден", T.GOOD, "diamond"), ("не пройден", T.CRITICAL, "x")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(symbol=sym, size=12, color=col)))
    fig.add_vline(x=1, line=dict(color=GOLD, width=2, dash="dash"), annotation_text="порог = 1", annotation_position="top")
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="эффект / заранее записанный порог (для H78, H79, H81-H84 порог 0 - показан знак ±1)")
    fig.update_layout(bargap=0.35, showlegend=True)
    rows = [[v["h"], v["name"], v["unit"], ru(v["effect"], 4, True), ru(v["threshold"], 4), badge("good", v["verdict"]) if PASSED(v) else badge("bad", "не прошла")] for v in vs]
    body = card("Доска вердиктов H67-H86: эффект против порога, записанного до результата", F(fig, "s5", "verdict_board", "", 120 + 40 * len(vs)),
                "Единицы у гипотез разные (Спирмен, AUC, скилл в %, Calmar), поэтому по оси - отношение эффекта к порогу. Данные - вердикты журнала (экспорт export2.py); подробные карточки ниже.")
    body += card("Таблица вердиктов", table(rows, ["гипотеза", "направление", "мера", "эффект", "порог", "вердикт"]))
    charts = []
    # ---- H67 gains
    selr = jfile("shape/results/sel_result.json")["table"]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("T = 30 мин", "T = 60 мин", "T = 120 мин"), horizontal_spacing=0.04)
    for j, tt in enumerate((30, 60, 120), 1):
        rr = [r for r in selr if r["T"] == tt]
        lab = [f"{r['target']} · {r['model']}" for r in rr]
        for r, l in zip(rr, lab):
            col = {"rho": PINK, "vr": VIOLET, "z2": GOLD, "z3": TAN}[r["target"]]
            fig.add_trace(go.Scatter(x=[r["mean"]], y=[l], mode="markers", marker=dict(size=10, color=col, symbol="circle" if r["p_holm"] >= 0.05 else "diamond", line=dict(color=INK, width=1)),
                                     error_x=dict(type="data", symmetric=False, array=[r["hi95"] - r["mean"]], arrayminus=[r["mean"] - r["lo95"]], color=col), showlegend=False,
                                     hovertemplate=f"{l}, T={tt}<br>прирост {r['mean']:.4f} [{r['lo95']:.4f}; {r['hi95']:.4f}]<br>p(Холм) = {r['p_holm']:.3f}<extra></extra>"), row=1, col=j)
        fig.add_vline(x=0.02, line=dict(color=GOLD, width=2, dash="dash"), row=1, col=j)
        fig.add_vline(x=0, line=dict(color=NEUT, width=1), row=1, col=j)
    fig.update_xaxes(title_text="прирост Спирмена над базой")
    fig.update_yaxes(autorange="reversed")
    charts.append(card("H67 · форма пути: прирост над базой волатильности (подбор 2019-2024, Холм)",
                       F(fig, "s5", "h67_gain", "", 400), "Золотая линия - порог +0,02. Ромб - значимо после Холма. Прошла бы только точка с интервалом правее линии; такой нет."))
    # ---- H68 rotation
    rot = jfile("shape/results/rotation.json")
    hv = rot["halves"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("Норма вектора коэффициентов по полугодиям", "Косинус между векторами полугодий"), horizontal_spacing=0.12)
    hx = [h["half"] for h in hv]
    nm = [h["norm"] for h in hv]
    se = [h["norm_se_approx"] for h in hv]
    fig.add_trace(go.Scatter(x=hx, y=[a + b for a, b in zip(nm, se)], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hx, y=[a - b for a, b in zip(nm, se)], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(137,135,129,0.25)", name="шум нормы (≈ SE)",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hx, y=nm, mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=6), name="норма",
                             hovertemplate="%{x}: норма %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=rot["cos_matrix"], x=hx, y=hx, zmin=-0.5, zmax=1, colorscale=[[0, DOWN], [0.33, SURFACE], [1, VIOLET]], colorbar=dict(title="cos", x=1.02),
                             hovertemplate="%{y} и %{x}: cos %{z:.2f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    charts.append(card("H68 · поворот или сжатие старого преимущества направления",
                       F(fig, "s5", "h68", "", 500, lift=20),
                       f"Тест отношения правдоподобия отвергает «чистое сжатие» (p = 0,005): вектор коэффициентов поворачивается (cos(2019, 2024H1) = {ru(rot['cos_2019_2024H1'], 2)}), "
                       "но адаптивные модели (Калман +0,0017 AUC, Нагель +0,0005) возвращают почти ничего."))
    # ---- H69 Omori
    om = jfile("omori/results/sel_result.json")["by_T"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Скилл log-loss степенного ядра над базой, %", "Показатель Омори p по годам (T = 30)"), horizontal_spacing=0.1)
    for i, tt in enumerate(("15", "30", "60")):
        c = om[tt]["cells"]["ll_pow_vs_B"]
        fig.add_trace(go.Scatter(x=[f"T = {tt} мин"], y=[c["mean"] * 100], mode="markers", marker=dict(size=12, color=VIOLET, line=dict(color=INK, width=1)),
                                 error_y=dict(type="data", symmetric=False, array=[(c["hi"] - c["mean"]) * 100], arrayminus=[(c["mean"] - c["lo"]) * 100], color=VIOLET, thickness=2),
                                 showlegend=False, hovertemplate=f"T = {tt}: {c['mean'] * 100:.3f}% [{c['lo'] * 100:.3f}; {c['hi'] * 100:.3f}]<extra></extra>"), row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color=GOLD, width=2, dash="dash"), annotation_text="порог +1%", row=1, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    py = om["30"]["p_by_year"]
    fig.add_trace(go.Scatter(x=list(py), y=list(py.values()), mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=9, symbol="square"), showlegend=False,
                             hovertemplate="%{x}: p = %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_hrect(y0=0.6, y1=1.3, fillcolor="rgba(137,135,129,0.15)", line_width=0, row=1, col=2)
    fig.add_annotation(text="серая полоса: p ∈ [0,6; 1,3] (критерий диапазона)", xref="x2 domain", yref="y2 domain", x=0.02, y=0.02, showarrow=False, xanchor="left", font=dict(size=10, color=INK2), row=1, col=2)
    fig.update_yaxes(title_text="скилл, %", row=1, col=1)
    fig.update_yaxes(title_text="p", row=1, col=2)
    charts.append(card("H69 · афтершоки Омори", F(fig, "s5", "h69", "", 420),
                       "Критерий (≥ +1% и нижняя граница > 0) не выполнен ни на одном горизонте; показатель p меняется по годам от 0,63 до 1,3 (диапазон 0,67 против допустимых 0,2). "
                       "Степенное ядро не отличается от экспоненты: память HAR и «время с последнего выброса» его поглощают."))
    # ---- H70 VaR
    vc = jfile("omori/results/var_sel_result.json")["cells"]
    fig = go.Figure()
    for c in vc:
        lab = f"T = {c['T']}, {'левый' if c['tail'] == 'left' else 'правый'} хвост, α = {c['alpha'] * 100:g}%"
        g = c["c_vs_fhs"]["gain"] * 100
        lo = c["c_vs_fhs"]["lo"] * 100
        col = PINK if c["primary"] else INK2
        fig.add_trace(go.Scatter(x=[g], y=[lab], mode="markers", marker=dict(size=11, color=col, symbol="diamond" if c["primary"] else "circle", line=dict(color=INK, width=1)), showlegend=False,
                                 hovertemplate=f"{lab}<br>выигрыш FZ0: {g:.2f}%<br>нижняя граница 95%: {lo:.2f}%" + ("<br>основная ячейка" if c["primary"] else "<br>вторичная") + "<extra></extra>"))
        fig.add_trace(go.Scatter(x=[lo], y=[lab], mode="markers", marker=dict(size=8, color=col, symbol="line-ns", line=dict(width=2, color=col)), showlegend=False, hoverinfo="skip"))
    for nm, col, sym in (("основная ячейка", PINK, "diamond"), ("вторичная ячейка (α = 1%)", INK2, "circle")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(size=11, color=col, symbol=sym)))
    fig.add_vline(x=1, line=dict(color=GOLD, width=2, dash="dash"), annotation_text="порог +1%")
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="выигрыш потери FZ0 к HAR-FHS, % (отметка | - нижняя граница одностороннего ДИ)")
    charts.append(card("H70 · VaR/ES по потере FZ0", F(fig, "s5", "h70", "", 120 + 30 * len(vc)),
                       "Лучшая ячейка +0,98% (T = 15, левый хвост, α = 1%, вторичная), основная лучшая +0,85%: ни одна не доходит до +1%. Выигрыш приходит от семивариации и «часов», состояние шока вклада не даёт."))
    # ---- TD
    conf = jfile("tdconfirm/confirm.json")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("H74: nn_td против ЛУЧШЕЙ базы (подтверждение 2017-07..2018-12)", "H74: nn_td против нерегуляризованной прямой сети", "H71: nn_td против прямой сети (подбор / взгляд)"),
                        horizontal_spacing=0.07)
    Ws = [30, 60, 90, 365]
    for Hh, col in ((240, PINK), (1440, VIOLET)):
        for key, cc, nm in (("gain_best", 1, "к лучшей базе"), ("gain_direct", 2, "к прямой сети")):
            xs, ys, lo, hi = [], [], [], []
            for w in Ws:
                g = conf["conf"]["cells"][f"{w}_{Hh}"][key]
                xs.append(w)
                ys.append(g["mean"])
                lo.append(g["t_lo"])
                hi.append(g["t_hi"])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=7),
                                     error_y=dict(type="data", symmetric=False, array=[h - y for h, y in zip(hi, ys)], arrayminus=[y - l for y, l in zip(ys, lo)], color=col, thickness=1),
                                     name=f"H = {Hh} мин", legendgroup=f"H{Hh}", showlegend=(cc == 1), hovertemplate=f"H={Hh}, {nm}<br>окно %{{x}} дн.: %{{y:.1f}}%<extra></extra>"), row=1, col=cc)
    for stage, dash in (("sel", "solid"), ("look", "dot")):
        d = json.load(open(TAC / "tdvar" / f"{'selection' if stage == 'sel' else 'look'}.json", encoding="utf-8"))
        for Hh, col in ((240, PINK), (1440, VIOLET)):
            rr = [r for r in d["curve_nn_td_vs_nn_direct"] if r["H"] == Hh]
            fig.add_trace(go.Scatter(x=[r["W"] for r in rr], y=[r["gain"] for r in rr], mode="lines+markers", line=dict(color=col, dash=dash, width=2), marker=dict(size=6),
                                     error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["gain"] for r in rr], arrayminus=[r["gain"] - r["lo"] for r in rr], color=col, thickness=1),
                                     name=f"H71, H = {Hh}, {'подбор (сплошная)' if stage == 'sel' else 'взгляд (пунктир)'}", showlegend=(stage == "sel" and False),
                                     hovertemplate=f"H={Hh}, {'подбор' if stage == 'sel' else 'взгляд'}<br>окно %{{x}} дн.: %{{y:.1f}}%<extra></extra>"), row=1, col=3)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_xaxes(type="log", title_text="окно обучения, дней", tickvals=[30, 60, 90, 365, 1095], ticktext=["30", "60", "90", "365", "1095"])
    fig.update_yaxes(title_text="выигрыш QLIKE, %", col=1)
    fig.update_layout(legend=dict(orientation="h"))
    charts.append(card("H71 / H74 · TD-оценка дисперсии: выигрыш против длины окна обучения", F(fig, "s5", "td", "", 520, lift=36),
                       "В H71 нейросетевой TD выигрывал у прямой сети на коротких окнах (+3..+12% на 30-60 дн.) и тем самым выглядел как гипотеза. H74 (заранее записано) показал: "
                       "по отношению к <i>регуляризованной</i> прямой сети и QLIKE-MIDAS на 60 днях nn_td проигрывает (−11,1% при H = 240); сигнатура «выигрыш падает с окном» воспроизводится только против нерегуляризованной сети - это эффект регуляризации."))
    # ---- arrow of time
    ar = jfile("arrow/results.json")
    mf = {r["month"]: r["ep"] for r in ar["monthly_ep"]["full"]}
    ma = {r["month"]: r["ep"] for r in ar["monthly_ep"]["abs"]}
    ms = sorted(mf)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[f"{m}-01" for m in ms], y=[mf[m] for m in ms], mode="lines+markers", line=dict(color=PINK, width=2), name="энтропия производства EP (полная), нат",
                             hovertemplate="%{x|%Y-%m}: EP %{y:.3f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=[f"{m}-01" for m in ms], y=[ma[m] for m in ms], mode="lines", line=dict(color=TAN, width=2, dash="dash"), name="часть от асимметрии волатильности",
                             hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=[f"{m}-01" for m in ms], y=[mf[m] - ma[m] for m in ms], mode="lines", line=dict(color=GOLD, width=2), name="направленная часть EP",
                             hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra></extra>"), secondary_y=False)
    tb = ar["tb7_auc_monthly"]
    tm = sorted(tb)
    fig.add_trace(go.Scatter(x=[f"{m}-01" for m in tm], y=[tb[m] for m in tm], mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=5, symbol="square"),
                             name="AUC направления tb7 (правая ось)", hovertemplate="%{x|%Y-%m}: AUC %{y:.3f}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="EP, нат", secondary_y=False)
    fig.update_yaxes(title_text="AUC", secondary_y=True)
    fig.update_layout(hovermode="x unified")
    tr = ar["i_tracking"]["EP_full"]
    charts.append(card("H76 · стрела времени: энтропия производства и направление", F(fig, "s5", "arrow", "", 460),
                       f"Сеть, отличающая окно от его обращения во времени, даёт месячную EP; слежение за AUC направления отсутствует (разностный ρ = {ru(tr['diff']['rho'], 2)}, "
                       f"95% блочный ДИ [{ru(tr['diff']['ci95_block_boot'][0], 2)}; {ru(tr['diff']['ci95_block_boot'][1], 2)}], порог ≥ 0,4). Затвор по EP: верх-низ квинтиля +0,011 при пороге +0,015. "
                       "Две трети EP - асимметрия волатильности; направленная часть падает с 2020 г."))
    body += "".join(charts)
    return body


# ======================================================================================================== section 6
def lr_coefs():
    """Standardised logistic coefficients of the tb7 regression (C = 0.1, 15-bar direction, deadband 5 bps) in two eras, with a month-block bootstrap."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load("D:/nt/nt_recut_cache/decay/1m.npz")
    Fm, anch, close = z["F"], z["anch"], z["close"]
    t = pd.DatetimeIndex(z["t"].astype("datetime64[ns]"))
    Hh = 15
    ok = anch + Hh - 1 < len(close)
    Fm, anch, t = Fm[ok], anch[ok], t[ok]
    r = close[anch + Hh - 1] / close[anch - 1] - 1
    y = (r > 0).astype(int)
    keep = np.abs(r) > 5e-4
    names = ["расстояние до EMA20", "RSI14", "гистограмма MACD", "Bollinger %b", "ATR14", "Stoch %K14", "CCI20", "доходность 1", "доходность 5", "доходность 10",
             "доходность 15", "доходность 20", "доходность 30"]
    out = {}
    rng = np.random.default_rng(0)
    for lab, (a, b) in (("2018-2020 (старый рынок)", ("2018-01-01", "2020-12-31")), ("2023-2025-07 (новый рынок)", ("2023-01-01", "2025-07-20"))):
        m = (t >= a) & (t <= pd.Timestamp(b) + pd.Timedelta(hours=23, minutes=59)) & keep
        X, yy, mon = Fm[m], y[m], np.asarray(t[m].to_period("M").astype(str))
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        base = LogisticRegression(C=0.1, max_iter=1000).fit(Xs, yy).coef_[0]
        ums = np.unique(mon)
        idx = {u: np.flatnonzero(mon == u) for u in ums}
        bs = []
        for _ in range(30):
            pick = rng.choice(ums, len(ums), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            bs.append(LogisticRegression(C=0.1, max_iter=500).fit(Xs[ii], yy[ii]).coef_[0])
        bs = np.array(bs)
        out[lab] = (base, np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0), int(m.sum()))
    return names, out


def sec6():
    ref = next(m for m in models if m["step"] == "L0")
    fig = go.Figure()
    fam_col = {"Пересборка, 11,5 дня": PINK, "Новая сеть": VIOLET}
    ms = sorted(models, key=lambda m: m["auc"])
    for m in ms:
        col = fam_col.get(m["family"], GOLD)
        fig.add_trace(go.Scatter(x=[m["auc"]], y=[m["model"]], mode="markers", marker=dict(size=11, color=col, symbol="circle", line=dict(color=INK, width=1)),
                                 error_x=dict(type="data", symmetric=False, array=[m["hi"] - m["auc"]], arrayminus=[m["auc"] - m["lo"]], color=col, thickness=2), showlegend=False,
                                 hovertemplate=f"{m['model']} ({m['family']})<br>AUC {m['auc']:.4f} [{m['lo']:.4f}; {m['hi']:.4f}]<extra></extra>"))
    for nm, col in fam_col.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(size=11, color=col)))
    fig.add_vline(x=ref["auc"], line=dict(color=GOLD, width=2, dash="dash"), annotation_text=f"регрессия {ref['auc']:.3f}", annotation_position="top")
    fig.add_vline(x=0.5, line=dict(color=NEUT, width=1, dash="dot"))
    fig.update_xaxes(title_text="AUC направления (среднее трёх горизонтов), 24 лабораторных среза, 95% t-интервал")
    body = card("Модели против регрессии: AUC направления на лабораторных срезах", F(fig, "s6", "models", "", 120 + 46 * len(ms)),
                "Золотая линия - логистическая регрессия на 13 признаках (учебниковые индикаторы + 6 прошлых доходностей). Никто из сетей (GRU, нелинейная сеть, PatchTST-lite, ансамбли) значимо её не обогнал: "
                "лучший прирост +0,0026 [+0,0001; +0,0052] у ансамбля PatchTST + регрессия на 3 годах обучения.")
    # long-training
    spans = ["11.5d", "90d", "1y", "3y"]
    order = sorted({l["span"] for l in longtrain}, key=lambda s: (float(re.sub(r"[a-z]", "", s)) * (1 if s.endswith("d") else 365 if s.endswith("y") else 1)))
    nmap = {"L0": ("Регрессия sklearn", GOLD), "L5": ("43 признака", TAN), "L6": ("+ нелинейная сеть", PINK), "L7": ("+ GRU по окну", VIOLET)}
    fig = go.Figure()
    for st, (nm, col) in nmap.items():
        rr = [l for l in longtrain if l["step"] == st]
        rr.sort(key=lambda l: order.index(l["span"]))
        fig.add_trace(go.Scatter(x=[l["span"] for l in rr], y=[l["auc"] for l in rr], mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=8),
                                 name=nm, hovertemplate=nm + ": окно обучения %{x}, AUC %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="длина обучения", categoryorder="array", categoryarray=order)
    fig.update_yaxes(range=[0.56, 0.58], tickformat=".3f")
    fig.update_yaxes(title_text="AUC направления (3 горизонта)")
    body += card("Длинное обучение: AUC по лестнице моделей", F(fig, "s6", "longtrain", "", 420),
                 "Лестница L0 → L7 (H28, H35) при росте обучения от 11,5 дней: более сложные модели не обгоняют регрессию, картина с ростом данных не меняется. Интервалы в этом экспорте не сохранены (только среднее по срезам).")
    # coefficients
    try:
        names, co = lr_coefs()
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[f"{k}, n = {v[3]:,}".replace(",", " ") for k, v in co.items()], horizontal_spacing=0.04)
        for j, (k, (b, lo, hi, n)) in enumerate(co.items(), 1):
            col = [PINK if x < 0 else VIOLET for x in b]
            fig.add_trace(go.Bar(y=names, x=b, orientation="h", marker=dict(color=col, line=dict(color=INK, width=0.5), pattern=dict(shape=["\\" if x < 0 else "/" for x in b], fgcolor=PAPER, size=5, solidity=0.2)),
                                 error_x=dict(type="data", symmetric=False, array=hi - b, arrayminus=b - lo, color=INK2, thickness=1.5), showlegend=False,
                                 hovertemplate="%{y}: коэффициент %{x:.3f}<extra></extra>"), row=1, col=j)
            fig.add_vline(x=0, line=dict(color=NEUT, width=1), row=1, col=j)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="стандартизованный коэффициент логита P(рост через 15 минут)")
        body += card("Что выучила регрессия: коэффициенты в два рынка", F(fig, "s6", "coefs", "", 560, lift=20),
                     "Модель H27/H51 (tb7, логистическая регрессия C = 0,1, стрелка 15 минут, мёртвая зона 5 б.п.) обучена на всём периоде эпохи; интервалы - блочный бутстреп по месяцам (30 пересэмплов). "
                     "Фиолетовый / - вклад в рост, розовый \ - вклад в падение. В старом рынке главный вклад - сильный отрицательный коэффициент при Bollinger %b (возврат к средней), "
                     "компенсируемый положительными CCI и гистограммой MACD: признаки сильно коррелируют, поэтому знаки читать по отдельности нельзя, важна комбинация. "
                     "В новом рынке всё сжалось: главный вклад - отрицательное расстояние до EMA20 (возврат к средней), интервалы шире, остальное около нуля.")
    except Exception as e:  # the cache is machine-local; the report still builds without it
        body += f'<p class="note">Панель коэффициентов недоступна: {H.escape(str(e))}</p>'
    return body


# ======================================================================================================== section 7
def sec7():
    out = []
    # ---- goal map
    pts = []
    for f in sorted((TAC / "equity").glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        st = d["levels"]["0"]["stats"]
        pts.append((d["strategy"][:60], st["hit"], st["max_drawdown_pct"], st["cum_return_pct"], "равное плечо 1x, издержки 0 (H58 пересчёт)"))
    NAMES = {"a_1h_rich_l1_365": "Часовой сигнал (a)", "b_h62_4h_rich_logreg_730": "Кандидат H62 (b)"}
    stops = json.load(open(TAC / "stops" / "results_look.json", encoding="utf-8"))["combos"]
    for combo, c in stops.items():
        sg, ver = combo.split("|")
        for rule, rr in c["rules"].items():
            lv = rr["levels"].get("2")
            if lv:
                st = lv["stats"]
                pts.append((f"{NAMES.get(sg, sg)}, порог {'прошлого' if ver == 'prev' else 'того же'} месяца, {rule}", st["hit"], st["max_drawdown_pct"], st["cum_return_pct"],
                            "издержки 2 б.п., вес 1 (H79)"))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=5, y0=0.6, y1=0.7, fillcolor="rgba(12,163,12,0.18)", line=dict(color=T.GOOD, width=1.5, dash="dash"))
    fig.add_annotation(x=2.5, y=0.65, text="цель владельца:<br>hit > 60% и просадка < 5%", showarrow=False, font=dict(color=T.GOOD, size=12))
    for i, (nm, hit, dd, cum, note) in enumerate(pts):
        col = S[i % len(S)]
        fig.add_trace(go.Scatter(x=[dd], y=[hit], mode="markers", marker=dict(size=10, color=PINK if "(H58" in note else VIOLET, symbol="circle" if "H58" in note else "square",
                                                                              line=dict(color=INK, width=1)), showlegend=False,
                                 hovertemplate=f"{nm}<br>макс. просадка {dd:.1f}%, hit {hit * 100:.1f}%, итог {cum:+.1f}%<br>{note}<extra></extra>"))
    for nm, col, sym in (("H58: равное плечо, 0 издержек", PINK, "circle"), ("H79: правила выхода, 2 б.п.", VIOLET, "square")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(size=10, color=col, symbol=sym)))
    fig.add_hline(y=0.5, line=dict(color=NEUT, width=1, dash="dot"))
    fig.update_xaxes(title_text="максимальная просадка, % (плечо 1)", range=[0, 25])
    fig.update_yaxes(title_text="доля верных сделок (hit)", range=[0.45, 0.7])
    out.append(card("Карта цели: hit против просадки по всем проверенным стратегиям", F(fig, "s7", "goal", "", 520),
                    "Зелёная область - цель владельца. Ни одна точка не попадает в неё: hit 45-56%, просадка 9-20% при плече 1 (оси обрезаны по данным). Просадку можно снизить плечом (см. границу ниже), hit плечом не меняется."))
    # ---- frontier
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Накопленный итог против максимальной просадки по весу позиции", "Максимальная просадка по весу позиции"), horizontal_spacing=0.08)
    series = []
    for r in frontier:
        if r["series"] not in series:
            series.append(r["series"])
    dash = {"X0_time": "solid", "X6_best_plus_filter": "dash"}
    for i, sname in enumerate(series):
        rr = [r for r in frontier if r["series"] == sname]
        col = S[i % len(S)]
        fig.add_trace(go.Scatter(x=[r["dd"] for r in rr], y=[r["cum"] for r in rr], mode="lines+markers", line=dict(color=col, width=2, dash=dash[rr[0]["rule"]]), marker=dict(size=6),
                                 name=sname, legendgroup=sname, customdata=[r["w"] for r in rr],
                                 hovertemplate=sname + "<br>вес %{customdata:.2f}: просадка %{x:.2f}%, итог %{y:.2f}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[r["w"] for r in rr], y=[r["dd"] for r in rr], mode="lines+markers", line=dict(color=col, width=2, dash=dash[rr[0]["rule"]]), marker=dict(size=6),
                                 name=sname, legendgroup=sname, showlegend=False, hovertemplate=sname + "<br>вес %{x:.2f}: просадка %{y:.2f}%<extra></extra>"), row=1, col=2)
    fig.add_vline(x=5, line=dict(color=T.CRITICAL, width=2, dash="dash"), annotation_text="просадка 5%", row=1, col=1)
    fig.add_hline(y=5, line=dict(color=T.CRITICAL, width=2, dash="dash"), annotation_text="5%", row=1, col=2)
    fig.update_xaxes(title_text="максимальная просадка, %", row=1, col=1)
    fig.update_yaxes(title_text="накопленный итог за период, %", row=1, col=1)
    fig.update_xaxes(title_text="вес позиции (1 = полное плечо)", row=1, col=2)
    fig.update_yaxes(title_text="просадка, %", row=1, col=2)
    out.append(card("Граница плеча: цена просадки < 5% (H79, период взгляда 2024-07..2025-07-20, 2 б.п.)", F(fig, "s7", "frontier", "", 640, lift=70),
                    "Каждая линия - один сигнал / версия порога / правило выхода; точки - вес позиции от 0,1 вверх. Порог «прошлого месяца» известен на входе (честная версия), «того же месяца» использует "
                    "месячный разброс прогнозов и оптимистичен (утечка, найденная в H79). Просадка < 5% достигается весом около 0,34-0,5, итог при этом +17..+24% за 12,6 месяца у кандидата H62."))
    # ---- stops table
    rows = []
    for combo, c in stops.items():
        sg, ver = combo.split("|")
        for rule, rr in c["rules"].items():
            lv = rr["levels"].get("2")
            if lv:
                st = lv["stats"]
                rows.append([NAMES.get(sg, sg), "прошлого" if ver == "prev" else "того же", rule, st["n_trades"], pct(st["hit"]), ru(st["mean_bps"], 2), ru(st["cum_return_pct"], 1) + "%",
                             ru(st["max_drawdown_pct"], 1) + "%", ru(st.get("calmar", float("nan")), 2) if st.get("calmar") is not None else "-"])
    out.append(card("Все правила выхода: таблица результатов на периоде взгляда (2 б.п., вес 1)", table(rows, ["сигнал", "порог месяца", "выход", "сделок", "hit", "б.п./сделку", "итог", "макс. просадка", "Calmar"]),
                    "X0 - выход по времени; X1-X3 - стоп = тейк на 1 / 1,5 / 2 σ HAR; X4 - тейк 2 / стоп 1; X5 - скользящий 1σ; X6 - лучшее правило + фильтр. Ни одно правило не обгоняет выход по времени."))
    # ---- hourly dev map
    d = sorted(hdev, key=lambda r: r["bps"])
    fig = go.Figure()
    for i, r in enumerate(d):
        col = PINK if r["feat"] == "tb7" else VIOLET
        sym = "circle" if r["primary"] == "logreg" else "square"
        lab = f"T{r['T']} tp {r['tp']:g} sl {r['sl']:g} · {r['feat']} · {r['primary']} · {r['target']} · мета {r['meta']}"
        fig.add_trace(go.Scatter(x=[i], y=[r["bps"]], mode="markers", marker=dict(size=7, color=col, symbol=sym, line=dict(color=INK, width=0.5)),
                                 error_y=dict(type="data", symmetric=False, array=[r["bps_hi"] - r["bps"]], arrayminus=[r["bps"] - r["bps_lo"]], color=col, thickness=1), showlegend=False,
                                 hovertemplate=lab + f"<br>{r['bps']:.1f} б.п. [{r['bps_lo']:.1f}; {r['bps_hi']:.1f}]<br>случайный 95%-порог {r['null95']:.1f}, избыток {r['excess']:+.1f}<br>hit {r['hit']:.3f}, сделок/день {r['tpd']:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(range(len(d))), y=[r["null95"] for r in d], mode="markers", marker=dict(size=6, color=GOLD, symbol="line-ew", line=dict(width=2, color=GOLD)),
                             name="95-й процентиль случайных входов", hoverinfo="skip"))
    for nm, col, sym in (("признаки tb7", PINK, "circle"), ("признаки rich", VIOLET, "circle"), ("логрегрессия", INK2, "circle"), ("градиентный бустинг", INK2, "square")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, marker=dict(size=8, color=col, symbol=sym)))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_xaxes(title_text=f"конфигурация (по возрастанию, {len(d)} штук)", showticklabels=False)
    fig.update_yaxes(title_text="средний итог сделки, б.п. (метамодель, период выбора)")
    out.append(card("Карта часовых конфигураций тройного барьера (H38, подбор): 94 конфигурации с интервалами", F(fig, "s7", "hdev", "", 520, lift=30),
                    "Каждая точка - одна конфигурация (барьер T, тейк / стоп, признаки, первичная модель, цель, метамодель); интервал - 95% по месяцам; золотая метка - случайный 95-й процентиль. "
                    "Выбор по периоду подбора оптимистичен: из 94 берётся победитель."))
    # heat map T x tp/sl
    cfg = defaultdict(list)
    for r in hdev:
        cfg[(r["T"], f"{r['tp']:g}/{r['sl']:g}")].append(r["excess"])
    Ts = sorted({k[0] for k in cfg})
    ts = sorted({k[1] for k in cfg})
    z = [[float(np.mean(cfg[(a, b)])) if (a, b) in cfg else None for b in ts] for a in Ts]
    n = [[len(cfg[(a, b)]) if (a, b) in cfg else 0 for b in ts] for a in Ts]
    fig = go.Figure(go.Heatmap(z=z, x=ts, y=[f"T = {a}" for a in Ts], customdata=n, colorscale=[[0, DOWN], [0.5, SURFACE], [1, VIOLET]], zmid=0, xgap=2, ygap=2,
                               text=[[("" if v is None else f"{v:+.1f}".replace(".", ",")) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=11, color=INK),
                               colorbar=dict(title="б.п."), hovertemplate="барьер %{x}, %{y}: средний избыток над случайным %{z:.1f} б.п. (конфигураций %{customdata})<extra></extra>"))
    fig.update_xaxes(title_text="тейк / стоп (в σ)", type="category")
    fig.update_yaxes(autorange="reversed", title_text="горизонт барьера, баров")
    out.append(card("Средний избыток над случайным 95-м процентилем: горизонт × тейк/стоп", F(fig, "s7", "hmap", "", 120 + 56 * len(Ts)),
                    "Среднее по всем признакам, моделям и мета-вариантам в клетке; положительное значение - в среднем выше случайного порога."))
    # held-out
    fig = go.Figure()
    for i, r in enumerate(hfin):
        lab = f"#{r['rank']} {r['tag']} ({r['selection']})"
        col = PINK if r["selection"] == "meta" else VIOLET if r["selection"] == "primary" else TAN
        fig.add_trace(go.Bar(y=[lab], x=[r["bps"]], orientation="h", marker=dict(color=col, opacity=0.7), showlegend=False,
                             error_x=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["bps"]], arrayminus=[r["bps"] - r["ci_lo"]], color=INK2, thickness=1.5),
                             hovertemplate=lab + f"<br>{r['bps']:.2f} б.п. [{r['ci_lo']:.1f}; {r['ci_hi']:.1f}] (по месяцам)<br>случайный 95%-порог {r['null95']:.2f}; после 10 б.п. издержек {r['net10']:.2f}<br>сделок {r['n']}, hit {r['hit']:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[lab], x=[r["null95"]], mode="markers", marker=dict(symbol="line-ns", size=18, color=GOLD, line=dict(width=3, color=GOLD)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="95-й процентиль случайных входов", marker=dict(symbol="line-ns", size=12, color=GOLD, line=dict(width=3, color=GOLD))))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="средний итог сделки, б.п.")
    out.append(card("Отложенный период: финальные результаты часового барьера (H38)", F(fig, "s7", "hfin", "", 120 + 46 * len(hfin)),
                    "Интервалы по месяцам пересекают 0 во всех строках; победитель по подбору (+9,8 б.п.) не выше случайного порога по интервалу. Это исходные числа H38: они читали данные до 2025-09-29 (нарушение протокола H66) и вход по закрытию; после пересчёта H73 с отсечкой 2025-07-20 и входом по открытию победитель +6,95 б.п. над случайным порогом, интервал по месяцам по-прежнему пересекает 0."))
    # equity pngs
    imgs = ""
    for f in sorted((TAC / "equity").glob("*.png")):
        b64 = base64.b64encode(open(f, "rb").read()).decode()
        imgs += f'<figure class="pngfig"><img alt="{f.stem}" src="data:image/png;base64,{b64}"><figcaption>{f.stem}</figcaption></figure>'
        reg("s7", "equity_" + f.stem, f"Кривая капитала {f.stem} (сохранённый PNG)")
    out.append(card("Сохранённые кривые капитала (H58, пересчёт H73, период взгляда 2024-07..2025-07-20)", f'<div class="pngs">{imgs}</div>',
                    "Помесячные ряды кривых капитала в JSON не сохранены (хранятся только итоги и граница плеча), поэтому показаны сохранённые рисунки скрипта equity (матплотлиб, не интерактивные): капитал при издержках 0/1/2/5 б.п., просадка, случайный контроль."))
    return "".join(out)


# ======================================================================================================== section 8
def sec8():
    rows = []
    for r in journal:
        L = LOG.get(r["id"], {})
        rows.append({"id": r["id"], "n": int(r["id"][1:]), "date": r["date"], "area": r["area"], "test": r["test"], "verdict": r["verdict"], "vtext": JV_TEXT[r["verdict"]],
                     "key": r["key"], "q": L.get("q", ""), "res": L.get("res", ""), "ev": L.get("ev", "")})
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    areas = sorted({r["area"] for r in journal})
    chips = "".join(f'<button class="chip" data-a="{a}">{a}</button>' for a in areas)
    vchips = "".join(f'<button class="chip vchip" data-v="{v}">{JV_TEXT[v]}</button>' for v in JV_TEXT)
    return (f'<div class="card"><div class="filters"><input id="jq" type="search" placeholder="Поиск по журналу (русский и английский текст)..."><div>{vchips}</div><div>{chips}</div></div>'
            '<table class="rt jt" id="jt"><thead><tr><th data-k="n">№ &#8693;</th><th data-k="date">дата &#8693;</th><th data-k="area">область &#8693;</th><th>что проверено</th>'
            '<th data-k="verdict">результат &#8693;</th><th>ключевая цифра</th></tr></thead><tbody></tbody></table>'
            '<p class="note" id="jcount"></p></div>'
            f'<script id="jdata" type="application/json">{data}</script>')


JV_TEXT = {"effect": "есть эффект", "partial": "частично", "none": "эффекта нет", "failed": "не подтвердилось", "info": "справка"}
JV_BADGE = {k: jbadge(k) for k in JV_TEXT}

# ======================================================================================================== assemble
CSS = """
:root{--bg:%(PAPER)s;--sf:%(SURFACE)s;--ink:%(INK)s;--ink2:%(INK2)s;--mut:%(MUTED)s;--grid:%(GRID)s;--axis:%(AXIS)s;--pink:%(PINK)s;--gold:%(GOLD)s}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink2);font-family:%(FONT)s;font-size:15px;line-height:1.5}
nav{position:fixed;top:0;left:0;bottom:0;width:250px;overflow:auto;padding:18px 14px;background:#161615;border-right:1px solid var(--axis)}
nav h1{font-size:15px;color:var(--ink);margin:0 0 4px}nav .sub{font-size:12px;color:var(--mut);margin-bottom:12px}
nav a{display:block;color:var(--ink2);text-decoration:none;padding:4px 8px;border-radius:4px;font-size:13px}nav a:hover,nav a.on{background:#262624;color:var(--ink)}
nav a.sm{padding-left:22px;font-size:12px;color:var(--mut)}
main{margin-left:250px;padding:22px 30px 80px;max-width:1700px}
section{margin-bottom:46px;scroll-margin-top:12px}
h2{color:var(--ink);font-size:24px;margin:8px 0 14px;padding-bottom:6px;border-bottom:1px solid var(--axis)}
h3{color:var(--ink);font-size:17px;margin:4px 0 8px}h4{color:var(--ink);font-size:14px;margin:10px 0 6px}h5{color:var(--ink2);font-size:13px;margin:10px 0 4px}
.card,.gblock{background:var(--sf);border:1px solid var(--axis);border-radius:8px;padding:14px 16px;margin:0 0 18px}
.gblock{padding:6px 18px 14px;border-color:#4a4a46}.gblock .card{background:#1f1f1e;margin:0}
.note{color:var(--mut);font-size:13px;margin:2px 0 10px;max-width:1300px}.note b,.lead b{color:var(--ink)}
.lead{background:var(--sf);border-left:4px solid var(--gold);padding:14px 18px;margin:14px 0 18px;font-size:16px;color:var(--ink);max-width:1300px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin:6px 0 10px}
.tile{background:var(--sf);border:1px solid var(--axis);border-radius:8px;padding:14px 16px}.tile .big{font-size:30px;color:var(--ink);font-weight:600;letter-spacing:-.5px}
.tile .lab{color:var(--ink2);font-size:13px;margin-top:2px}.tile .sub{color:var(--mut);font-size:12px;margin-top:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}
@media(max-width:1200px){.two{grid-template-columns:1fr}nav{display:none}main{margin-left:0}}
table.rt{border-collapse:collapse;font-size:12.5px;width:100%%;margin:4px 0 8px}table.rt th,table.rt td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:left;vertical-align:top}
table.rt th{color:var(--ink);font-weight:600;background:#222220;position:sticky;top:0}table.rt td:nth-child(n+2){font-variant-numeric:tabular-nums}
.badge{display:inline-block;border:1px solid;border-radius:10px;padding:0 8px;font-size:11.5px;white-space:nowrap}
.jr{margin:6px 0;font-size:12.5px}code{background:#262624;padding:1px 5px;border-radius:3px;color:var(--ink)}
.filters{display:flex;flex-direction:column;gap:8px;margin-bottom:10px}#jq{background:#121211;border:1px solid var(--axis);color:var(--ink);padding:8px 10px;border-radius:6px;font-size:14px}
.chip{background:#262624;border:1px solid var(--axis);color:var(--ink2);border-radius:14px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;cursor:pointer}.chip.on{background:var(--gold);color:#121211;border-color:var(--gold)}
#jt th[data-k]{cursor:pointer}#jt tr.main{cursor:pointer}#jt tr.main:hover td{background:#222220}#jt td.det{background:#171716;color:var(--ink2);font-size:12px}
.pngs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}.pngfig{margin:0;background:#fff;border-radius:6px;padding:4px}.pngfig img{width:100%%;display:block}.pngfig figcaption{color:#333;font-size:12px;padding:2px 6px}
.js-plotly-plot{width:100%%}footer{color:var(--mut);font-size:12px;margin-top:30px}
body.only nav{display:none}body.only main{margin-left:0}
""" % dict(PAPER=PAPER, SURFACE=SURFACE, INK=INK, INK2=INK2, MUTED=MUTED, GRID=GRID, AXIS=AXIS, PINK=PINK, GOLD=GOLD, FONT=T.FONT)

JS = """
(function(){
const q=new URLSearchParams(location.search).get('only');
const blk=new URLSearchParams(location.search).get('blk');
if(q){document.body.classList.add('only');document.querySelectorAll('main > section').forEach(s=>{if(s.id!==q)s.style.display='none'});}
if(q&&blk){[...document.getElementById(q).children].forEach(c=>{if(c.tagName!=='H2'&&c.id!==blk&&!(blk==='intro'&&!c.id&&c.tagName!=='H2'&&c.className!=='gblock'))c.style.display='none'});}
addEventListener('load',()=>setTimeout(()=>window.dispatchEvent(new Event('resize')),500));
const links=[...document.querySelectorAll('nav a')];
function onscroll(){let cur=null;document.querySelectorAll('main > section').forEach(s=>{if(s.getBoundingClientRect().top<140)cur=s.id});links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+cur))}
addEventListener('scroll',onscroll);
// journal table
const el=document.getElementById('jdata');if(!el)return;
const rows=JSON.parse(el.textContent);let sortK='n',sortD=1,qs='',vf=new Set(),af=new Set();
const BADGE=%s;
const tb=document.querySelector('#jt tbody');
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function render(){
 const t=qs.toLowerCase();
 let r=rows.filter(x=>(!vf.size||vf.has(x.verdict))&&(!af.size||af.has(x.area))&&(!t||(x.id+' '+x.date+' '+x.area+' '+x.test+' '+x.key+' '+x.vtext+' '+x.q+' '+x.res+' '+x.ev).toLowerCase().includes(t)));
 r.sort((a,b)=>(a[sortK]>b[sortK]?1:a[sortK]<b[sortK]?-1:0)*sortD);
 tb.innerHTML=r.map(x=>`<tr class="main" data-id="${x.id}"><td><b>${x.id}</b></td><td>${x.date}</td><td>${esc(x.area)}</td><td>${esc(x.test)}</td><td>${BADGE[x.verdict]}</td><td>${esc(x.key)}</td></tr>`+
 `<tr class="detrow" id="d_${x.id}" style="display:none"><td></td><td colspan="5" class="det"><b>Вопрос (журнал, англ.):</b> ${esc(x.q)}<br><b>Результат (журнал, англ.):</b> ${esc(x.res)}<br><b>Файлы:</b> ${esc(x.ev)}</td></tr>`).join('');
 document.getElementById('jcount').textContent='Показано '+r.length+' из '+rows.length+'. Нажмите на строку, чтобы раскрыть полный текст журнала.';
 tb.querySelectorAll('tr.main').forEach(tr=>tr.onclick=()=>{const d=document.getElementById('d_'+tr.dataset.id);d.style.display=d.style.display==='none'?'':'none'});
}
document.getElementById('jq').addEventListener('input',e=>{qs=e.target.value;render()});
document.querySelectorAll('#jt th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortK===k)sortD=-sortD;else{sortK=k;sortD=1}render()});
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{c.classList.toggle('on');const s=c.dataset.v?vf:af,v=c.dataset.v||c.dataset.a;s.has(v)?s.delete(v):s.add(v);render()});
render();
})();
"""


def build():
    print("loading bars ...", flush=True)
    d = load_bars()
    ws = pick_windows(d)
    for w in ws:
        print(" window", w["label"], span_text(w), flush=True)
    secs = []
    t0 = time.time()
    for sid, title, fn, args in (("s1", "1. Итог", sec1, ()), ("s2", "2. Затухание рынка", sec2, ()), ("s3", "3. Карта затухания", sec3, ()),
                                 ("s4", "4. Геометрия", sec4, (ws,)), ("s5", "5. Неизведанные направления", sec5, ()), ("s6", "6. Модели и регрессия", sec6, ()),
                                 ("s7", "7. Торговля", sec7, ()),
                                 ("s9", "8. Проверка на будущем", lambda: FS.sec9(F, reg), ()), ("s10", "9. Волатильность", lambda: FS.sec10(F, reg), ()),
                                 ("s8", "10. Журнал H5-H86", sec8, ())):
        body = fn(*args)
        secs.append((sid, title, body))
        print(f" {sid} done {time.time() - t0:.0f}s", flush=True)
    nav = ""
    for sid, title, _ in secs:
        nav += f'<a href="#{sid}">{title}</a>'
        if sid == "s4":
            for a, t in (("g_h24", "H24/43/45/47 индикаторы"), ("g_n1", "N1 кривизна, кручение"), ("g_n2", "N2 площади Леви"), ("g_n3", "N3 фазовый портрет"), ("g_n4", "N4 оболочка"),
                         ("g_n5", "N5 Вассерштейн"), ("g_n6", "N6 Фишер-Рао"), ("g_n7", "N7 выпуклость"), ("g_h67", "H67 форма пути"), ("g_h80", "H80 синтетика"), ("forest", "все варианты вместе")):
                nav += f'<a class="sm" href="#{a}">{t}</a>'
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    parts = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Тактическая сессия: отчёт</title><style>" + CSS + "</style><script>" + po.get_plotlyjs() + "</script></head><body>",
             f"<nav><h1>Тактическая сессия</h1><div class='sub'>neural_trade · отчёт 2026-10-13<br>данные до {C.CUTOFF[:10]}</div>{nav}</nav><main>"]
    for sid, title, body in secs:
        parts.append(f'<section id="{sid}"><h2>{title}</h2>{body}</section>')
    parts.append(f"<footer>Сгенерировано runs/tactical/report/build_report.py (коммит основы {sha}); данные: runs/tactical/webdash/out/*.json и файлы результатов, "
                 f"на которые они ссылаются. Цены - только до {C.CUTOFF} (runs/tactical/common/cutoff.py). Исследовательская работа тактической сессии: одно зерно, только dev-сплиты, "
                 "не влияет на умолчания MVP (D-062).</footer></main>")
    parts.append("<script>" + JS % json.dumps(JV_BADGE, ensure_ascii=False) + "</script></body></html>")
    out = Path(__file__).resolve().parent / "tactical_report.html"
    out.write_text("".join(parts), encoding="utf-8")
    json.dump([{"section": s, "key": k, "title": t} for s, k, t in FIGS], open(Path(__file__).resolve().parent / "figures.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {out} {out.stat().st_size / 1e6:.1f} MB, {len(FIGS)} figures, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    build()
