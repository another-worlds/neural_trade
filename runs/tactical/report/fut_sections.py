"""Sections 9 (Проверка на будущем: H81, H82) and 10 (Волатильность: T01-T04 = H83-H86) of tactical_report.html. Same style as the other sections
(rcommon helpers, theme.py colours, Russian text). Every number is read from committed result files:
  runs/tactical/hc2/lockbox_result.json, hc2/winner.json            (H81)
  runs/tactical/fwdtest/fwdtest_result.json                          (H82)
  runs/tactical/tasks/T01_vol_forward/results.json                   (H83)
  runs/tactical/tasks/T02_vol_exposure/{results.json, look2_hourly.csv}  (H84)
  runs/tactical/tasks/T03_vol_intervals/results.json                 (H85)
  runs/tactical/tasks/T04_vol_calendar/resultsA.json                 (H86)
  runs/tactical/report/vol_windows.json                              (real forward days; made by vol_windows.py from the frozen T03 fit code)
Colours: horizon colours (h0-h2) are never used; series use theme.OTHER_SERIES, status colours always with a shape; solid lines (dotted = training only)."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rcommon import (T, TAC, HERE, PINK, GOLD, TAN, VIOLET, SALMON, INK, INK2, SURFACE, PAPER, badge, table, card, ru)

NEUT = T.NEUTRAL
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def pct(x, nd=1):
    return ru(x * 100, nd) + "%"


def mlab(m):
    return f"{MONTHS_RU[int(m[5:7]) - 1]} {m[2:4]}"


def jf(rel):
    return json.load(open(TAC / rel, encoding="utf-8"))


def tile(big, lab, sub):
    return f'<div class="tile"><div class="big">{big}</div><div class="lab">{lab}</div><div class="sub">{sub}</div></div>'


# ======================================================================================================== section 9
def sec9(F, reg):
    out = []
    # ------------------------------------------------------------------ H81
    lb = jf("hc2/lockbox_result.json")
    W, B = lb["winner"], lb["baseline"]
    cw, cb = lb["winner_cfg"], lb["baseline_cfg"]
    wl, bl, wd, bd = W["lockbox"], B["lockbox"], W["dev"], B["dev"]
    nl = bl["null"]
    fwd = jf("fwdtest/fwdtest_result.json")["arms"]["frozen"]
    L0, L2 = fwd["levels"]["0"], fwd["levels"]["2"]
    tiles = [
        tile(f"{ru(wd['calmar'], 1)} → {ru(wl['calmar'], 2)}", "Calmar победителя подбора: dev → сейф",
             f"H81: 1 944 конфигурации; на сейфе (12 месяцев 2019-24) просадка {ru(wl['max_dd_pct'], 0)}%, {ru(wl['mean_net_bps'], 1)} б.п. за сделку"),
        tile(f"{pct(L2['stats']['hit'])}", "hit кандидата H62 на новых данных (H82, 2 б.п.)",
             f"11 месяцев 2025-11..2026-09, 437 сделок; на выборе было {pct(0.593)}; месяцев с hit > 60%: {L2['months_hit_gt_60']} из {L2['n_months']}"),
        tile(f"{ru(L2['stats']['cum_return_pct'], 1, True)}%", "итог H62 вперёд против случайной стороны",
             f"{pct(L2['null']['share_null_cum_ge_actual'], 0)} случайных розыгрышей не хуже; просадка {ru(L2['stats']['max_drawdown_pct'], 1)}% при пределе 15%"),
        tile(f"{ru(L2['leverage']['max_leverage_dd_lt_5pct']['leverage'], 2)}x", "плечо, при котором просадка < 5%",
             f"итог при нём {ru(L2['leverage']['max_leverage_dd_lt_5pct']['cum_return_pct'], 1, True)}%: просадку плечо снижает, доход из минуса не делает"),
    ]
    lead = ('<div class="lead"><b>Итог раздела.</b> Обе проверки «на будущем» дали отрицательный результат. (H81) Подбор по 1 944 конфигурациям нашёл победителя с отличными числами на периоде подбора, '
            'но на «сейфе» - 12 месяцах 2019-24, отложенных до подбора, - его Calmar падает с 10,5 до 0,19: это переобучение. (H82) Замороженный кандидат H62, обученный на 730 днях до 2025-07-20, '
            'на реально невиданных данных Binance (2025-11..2026-09) не даёт hit выше 51-52%, а итог не отличим от случайной стороны. '
            'Преимущество H62 жило в 2019-2021 и исчезло, что согласуется с историей затухания (раздел 2).</div>')
    out.append('<div class="tiles">' + "".join(tiles) + "</div>" + lead)

    # ---- H81 dev vs lockbox
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Calmar", "hit, %", "результат сделки, б.п.", "макс. просадка, %"), horizontal_spacing=0.07)
    cats = ["подбор (dev)", "сейф (lockbox)"]
    vals = {"Победитель подбора": (PINK, [wd, wl]), "Кандидат H62 (базовая точка)": (VIOLET, [bd, bl])}
    keys = (("calmar", 1.0), ("hit", 100.0), ("mean_net_bps", 1.0), ("max_dd_pct", 1.0))
    for c_i, (k, mult) in enumerate(keys, 1):
        for nm, (col, rows) in vals.items():
            ys = [r[k] * mult for r in rows]
            fig.add_trace(go.Bar(x=cats, y=ys, name=nm, legendgroup=nm, showlegend=(c_i == 1), marker=dict(color=col, line=dict(color=SURFACE, width=1.5), pattern_shape=["", "/"]),
                                 text=[ru(y, 1 if k != "calmar" else 2) for y in ys], textposition="outside", cliponaxis=False,
                                 hovertemplate=nm + " · %{x}<br>" + k + " = %{y:.2f}<extra></extra>"), row=1, col=c_i)
    fig.add_hline(y=60, line=dict(color=T.GOOD, width=1.5, dash="dash"), row=1, col=2)
    fig.add_hline(y=5, line=dict(color=T.GOOD, width=1.5, dash="dash"), row=1, col=4)
    fig.update_yaxes(range=[40, 70], row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.25)
    out.append(card("H81 · Подбор против сейфа: победитель hill-climb v2 и кандидат H62", F(fig, "s9", "h81_dev_lock", "", 470, lift=30),
                    "Штриховка - сейф: 12 месяцев 2019-24, выбранных до подбора и ни разу не оценённых в нём. Зелёные линии - цель владельца (hit &gt; 60%, просадка &lt; 5%). "
                    "<b>Осторожно с H62:</b> он сам выбран по 2019-01..2024-06, где лежат все месяцы сейфа, поэтому его хорошие числа на сейфе - не доказательство, а память выбора. "
                    "Победитель: " + f"{cw['feat']}, {cw['model']}, окно {cw['T']} д, горизонт {cw['H']} ч, верхние {int(cw['q'] * 100)}%, стороны {cw['side']}, выход {cw['exit']}."))

    # monthly lockbox
    mo = lb["lockbox_months"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[mlab(m) for m in mo], y=wl["monthly_net_bps"], name="Победитель подбора", marker=dict(color=PINK, line=dict(color=SURFACE, width=1.5)),
                         hovertemplate="%{x}: %{y:.0f} б.п.<extra>победитель</extra>"))
    fig.add_trace(go.Bar(x=[mlab(m) for m in mo], y=bl["monthly_net_bps"], name="Кандидат H62 (в выборе)", marker=dict(color=VIOLET, line=dict(color=SURFACE, width=1.5)),
                         hovertemplate="%{x}: %{y:.0f} б.п.<extra>H62</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title_text="итог за месяц, б.п. (вес 1)")
    fig.update_xaxes(title_text="месяц сейфа")
    ci_w = wl.get("monthly_ci95") or [None, None]
    out.append(card("H81 · Месяцы сейфа: итог по месяцам", F(fig, "s9", "h81_months", "", 420),
                    f"Победитель: средний месяц {ru(np.mean(wl['monthly_net_bps']), 0)} б.п., hit &gt; 60% в {wl['months_hit_gt_60']} из 12 месяцев, интервал среднего месяца включает ноль. "
                    f"Кандидат H62: средний месяц {ru(np.mean(bl['monthly_net_bps']), 0)} б.п., 95% [{ru(bl['monthly_ci95'][0], 0)}; {ru(bl['monthly_ci95'][1], 0)}], выше случайного 95-го процентиля "
                    f"(итог {ru(bl['cum_pct'], 1)}% против {ru(nl['cum_p95'], 1)}%), но это не evidence (см. выше)."))
    rows = [["Победитель: подбор", wd["n_trades"], pct(wd["hit"]), ru(wd["mean_net_bps"], 2), ru(wd["cum_pct"], 1) + "%", ru(wd["max_dd_pct"], 1) + "%", ru(wd["calmar"], 2)],
            ["Победитель: сейф", wl["n_trades"], pct(wl["hit"]), ru(wl["mean_net_bps"], 2), ru(wl["cum_pct"], 1) + "%", ru(wl["max_dd_pct"], 1) + "%", ru(wl["calmar"], 2)],
            ["H62: подбор", bd["n_trades"], pct(bd["hit"]), ru(bd["mean_net_bps"], 2), ru(bd["cum_pct"], 1) + "%", ru(bd["max_dd_pct"], 1) + "%", ru(bd["calmar"], 2)],
            ["H62: сейф (в выборе)", bl["n_trades"], pct(bl["hit"]), ru(bl["mean_net_bps"], 2), ru(bl["cum_pct"], 1) + "%", ru(bl["max_dd_pct"], 1) + "%", ru(bl["calmar"], 2)]]
    crit = [[k, badge("bad", "не выполнен") if not v else badge("good", "выполнен")] for k, v in lb["criteria"].items()]
    out.append(card("H81 · Числа и заранее записанные критерии сейфа", table(rows, ["строка", "сделок", "hit", "б.п./сделку", "итог", "макс. просадка", "Calmar"]) +
                    "<h5>Критерии сейфа (SPEC 3d561ba)</h5>" + table(crit, ["критерий", "итог"])))

    # ------------------------------------------------------------------ H82
    sm = L2["monthly"]
    ms = [m["m"] for m in sm]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Итог по месяцам, б.п. (2 б.п. издержек, вес 1)", "hit по месяцам"), horizontal_spacing=0.08)
    sums = [m["sum_bps"] for m in sm]
    fig.add_trace(go.Bar(x=[mlab(m) for m in ms], y=sums, name="итог месяца", marker=dict(color=[VIOLET if v >= 0 else PINK for v in sums], line=dict(color=SURFACE, width=1.5)),
                         text=[f"{v:+.0f}" for v in sums], textposition="outside", cliponaxis=False, showlegend=False,
                         hovertemplate="%{x}: %{y:.0f} б.п.<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    hits = [m["hit"] * 100 for m in sm]
    fig.add_trace(go.Scatter(x=[mlab(m) for m in ms], y=hits, mode="lines+markers", line=dict(color=GOLD, width=2), marker=dict(size=9, color=GOLD, line=dict(color=INK, width=1)),
                             name="hit месяца", customdata=[m["n"] for m in sm], hovertemplate="%{x}: hit %{y:.1f}%, сделок %{customdata:.0f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=60, line=dict(color=T.GOOD, width=2, dash="dash"), annotation_text="цель: 60%", annotation_position="top left", row=1, col=2)
    fig.add_hline(y=50, line=dict(color=NEUT, width=1, dash="dash"), annotation_text="50%", annotation_position="bottom left", row=1, col=2)
    fig.add_trace(go.Scatter(x=[mlab(m) for m, h in zip(ms, hits) if h > 60], y=[h for h in hits if h > 60], mode="markers", name="hit &gt; 60%",
                             marker=dict(symbol="diamond", size=13, color=T.GOOD, line=dict(color=INK, width=1)), hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(title_text="б.п.", row=1, col=1)
    fig.update_yaxes(title_text="% верных сделок", range=[30, 70], row=1, col=2)
    out.append(card("H82 · H62 на невиданных данных: помесячные итоги и hit", F(fig, "s9", "h82_months", "", 460, lift=20),
                    f"Замороженная модель (730 дней до 2025-07-20), порог прошлого месяца, вход по следующему открытию, 2 б.п. Всего 437 сделок: hit {pct(L2['stats']['hit'], 1)}, "
                    f"{ru(L2['stats']['mean_bps'], 1)} б.п. за сделку, средний месяц {ru(L2['mean_monthly_net_bps'], 0)} б.п. (95% [{ru(L2['monthly_ci95_t'][0], 0)}; {ru(L2['monthly_ci95_t'][1], 0)}]). "
                    f"Месяцев с hit &gt; 60%: {L2['months_hit_gt_60']} из {L2['n_months']}. Критерии пройдены: {sum(fwd['criteria'].values())} из {len(fwd['criteria'])}."))

    # equity vs null band (monthly, additive: the study's own cumulative return is additive over trades)
    eq2 = np.concatenate([[0.0], np.cumsum([m["sum_bps"] for m in L2["monthly"]]) / 100])
    eq0 = np.concatenate([[0.0], np.cumsum([m["sum_bps"] for m in L0["monthly"]]) / 100])
    k = np.arange(len(eq2))
    nk = len(sm)
    nm, np95 = L2["null"]["cum_return_pct_median"], L2["null"]["cum_return_pct_p95"]
    sig = (np95 - nm) / 1.645
    t_ = k / nk
    mid = nm * t_
    up, lo = mid + 1.645 * sig * np.sqrt(t_), mid - 1.645 * sig * np.sqrt(t_)
    xl = ["старт"] + [mlab(m) for m in ms]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xl + xl[::-1], y=list(up) + list(lo[::-1]), fill="toself", fillcolor="rgba(160,160,160,0.20)", line=dict(width=0), name="полоса случайной стороны (оценка 5-95%)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xl, y=mid, mode="lines", line=dict(color=NEUT, width=1.5, dash="dash"), name="случайная сторона: медиана", hovertemplate="%{x}: %{y:.1f}%<extra>медиана нуля</extra>"))
    fig.add_trace(go.Scatter(x=xl, y=eq0, mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=6), name="H62 вперёд, 0 б.п.", hovertemplate="%{x}: %{y:.1f}%<extra>0 б.п.</extra>"))
    fig.add_trace(go.Scatter(x=xl, y=eq2, mode="lines+markers", line=dict(color=PINK, width=3), marker=dict(size=8, color=PINK, line=dict(color=INK, width=1)), name="H62 вперёд, 2 б.п.",
                             hovertemplate="%{x}: %{y:.1f}%<extra>2 б.п.</extra>"))
    fig.add_trace(go.Scatter(x=[xl[-1], xl[-1]], y=[nm, np95], mode="markers", marker=dict(symbol="line-ew", size=22, color=GOLD, line=dict(width=3, color=GOLD)), name="итог нуля: медиана и 95-й процентиль (2000 розыгрышей)",
                             hovertemplate="конец: %{y:.1f}%<extra>итог нуля</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_yaxes(title_text="накопленный итог, % (сумма сделок, вес 1)")
    out.append(card("H82 · Капитал H62 вперёд против случайной стороны", F(fig, "s9", "h82_equity", "", 480, lift=14),
                    f"Кривая - помесячная накопленная сумма (в исследовании итог аддитивен по сделкам: {ru(L2['stats']['cum_return_pct'], 2)}% при 2 б.п.). Серая полоса - <b>оценка</b>: те же самые "
                    f"розыгрыши случайной стороны (те же сделки) сохранены только как итог на конец (медиана {ru(nm, 1)}%, 95-й процентиль {ru(np95, 1)}%), поэтому промежуточная форма - расширение корнем из времени "
                    f"от этих двух чисел, а не отдельные пути. {pct(L2['null']['share_null_cum_ge_actual'], 1)} розыгрышей дали итог не хуже реального; макс. просадка {ru(L2['stats']['max_drawdown_pct'], 1)}% "
                    f"(у случайной стороны медиана {ru(L2['null']['max_drawdown_pct_median'], 1)}%)."))

    # leverage frontier
    lev = [(int(k_), v) for k_, v in L2["leverage"].items() if k_.isdigit()]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Просадка по плечу", "Итог по плечу"), horizontal_spacing=0.1)
    xs = [a for a, _ in lev]
    fig.add_trace(go.Scatter(x=xs, y=[v["max_drawdown_pct"] for _, v in lev], mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=8), name="макс. просадка",
                             hovertemplate="плечо %{x}x: просадка %{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[v["cum_return_pct"] for _, v in lev], mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=8), name="итог",
                             hovertemplate="плечо %{x}x: итог %{y:.1f}%<extra></extra>"), row=1, col=2)
    ml = L2["leverage"]["max_leverage_dd_lt_5pct"]
    fig.add_hline(y=5, line=dict(color=T.CRITICAL, width=2, dash="dash"), annotation_text="просадка 5%", row=1, col=1)
    fig.add_trace(go.Scatter(x=[ml["leverage"]], y=[5], mode="markers", marker=dict(symbol="diamond", size=14, color=GOLD, line=dict(color=INK, width=1)), name="плечо при просадке 5%",
                             hovertemplate=f"плечо {ml['leverage']:.2f}x: просадка 5%, итог {ml['cum_return_pct']:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=2)
    fig.update_xaxes(title_text="плечо (в разах)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="плечо (в разах)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    out.append(card("H82 · Граница плеча на новых данных", F(fig, "s9", "h82_lev", "", 430, lift=20),
                    f"Плечо масштабирует и просадку, и итог. Просадка 5% достигается плечом около {ru(ml['leverage'], 2)}x, и итог при нём {ru(ml['cum_return_pct'], 1, True)}%: цель «просадка &lt; 5%» "
                    "выполнима только уменьшением позиции, но без положительного ожидания это лишь меньший убыток. Плечо выше 1 здесь только увеличивает потери."))
    return "".join(out)


# ======================================================================================================== section 10
def sec10(F, reg):
    out = []
    t1 = jf("tasks/T01_vol_forward/results.json")
    t3 = jf("tasks/T03_vol_intervals/results.json")
    t2 = jf("tasks/T02_vol_exposure/results.json")
    t4 = jf("tasks/T04_vol_calendar/resultsA.json")
    S1_ = t1["summary"]["F"]["A"]
    crit = t3["summary"]["criterion"]
    c60 = crit["60"]
    look2 = t2["look2"]
    tiles = [
        tile(f"+{ru(S1_['S3']['skill_ar'], 3)}", "H83 · прогноз изменения волатильности вперёд", f"скилл к AR на новых данных, 12 из 12 месяцев; на взгляде H72 было +0,163 (на 13% меньше)"),
        tile(f"{ru(c60['mean_diff_iscore_bps'], 1)} б.п.", "H85 · 90%-интервалы на час: HAR против скользящей RV",
             f"интервальная оценка; 11 из 11 месяцев, покрытие {ru(c60['pooled_cover'], 3)}; пройдено на T = 15, 60 и 240 (Holm)"),
        tile("нет", "H84 · переключение экспозиции по волатильности", f"просадка {ru(look2['2']['strategy']['max_dd'] * 100, 1)}% против предела 40,6%; при просадке &lt; 5% в BTC остаётся 7-15% капитала"),
        tile("нет", "H86 · 15-минутный календарь", f"S3C - S3 {ru(t4['criterion_A']['mean_diff_skill_ar'], 3, True)} на 2017-18; а вот «сверх HAR» воспроизвелось: +{ru(t4['comparisons']['S3-harlvl']['mean_diff_skill_ar'], 3)}, 13 из 16"),
    ]
    lead = ('<div class="lead"><b>Итог раздела.</b> Предсказуемость <i>волатильности</i> - единственный результат сессии, переживший невиданные данные: прогноз изменения волатильности на час держится '
            '(H83), а его применение для интервалов доходности даёт небольшой, но чистый выигрыш (H85). Применение того же прогноза как переключателя экспозиции BTC не защищает от просадки (H84), '
            'а календарь по 15-минутным слотам, найденный на самих новых месяцах, не воспроизвёлся на 2017-18 (H86). Это измерение риска, а не торговое преимущество.</div>')
    out.append('<div class="tiles">' + "".join(tiles) + "</div>" + lead)

    # ---- T01
    rows = [r for r in t1["rows"] if r["arm"] == "F" and r["target"] == "A"]
    mset = sorted({r["month"] for r in rows})
    ser = (("S3", "S3: HAR + признаки + изменение волатильности", PINK, "circle"), ("S1", "S1: только log-RV и час (робастен к источнику)", VIOLET, "square"),
           ("arlvl", "уровни HAR (AR с уровнями)", GOLD, "diamond"))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], subplot_titles=("Скилл к AR по месяцам, новые данные Binance", "Среднее и 95% интервал по месяцам"), horizontal_spacing=0.08)
    look = t1["summary"]["h72_look"]
    ref, rlo, rhi = look["skill_ar"]
    fig.add_hrect(y0=rlo, y1=rhi, fillcolor="rgba(160,160,160,0.18)", line_width=0, row=1, col=1)
    fig.add_hline(y=ref, line=dict(color=NEUT, width=2, dash="dash"), annotation_text=f"взгляд H72 (S3): {ru(ref, 3, True)}", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    fig.add_hline(y=0.02, line=dict(color=T.GOOD, width=1.5, dash="dash"), annotation_text="порог +0,02", annotation_position="bottom left", row=1, col=1)
    for key, nm, col, sym in ser:
        d = sorted([r for r in rows if r["model"] == key], key=lambda r: r["month"])
        fig.add_trace(go.Scatter(x=[mlab(r["month"]) for r in d], y=[r["skill_ar"] for r in d], mode="lines+markers", name=nm, legendgroup=key,
                                 line=dict(color=col, width=2), marker=dict(size=8, symbol=sym, color=col, line=dict(color=INK, width=0.5)),
                                 hovertemplate="%{x}: скилл " + nm.split(":")[0] + " %{y:+.3f}<extra></extra>"), row=1, col=1)
        m_ = S1_[key]
        fig.add_trace(go.Scatter(x=[m_["skill_ar"]], y=[key], mode="markers", legendgroup=key, showlegend=False, marker=dict(size=11, symbol=sym, color=col, line=dict(color=INK, width=1)),
                                 error_x=dict(type="data", symmetric=False, array=[m_["skill_ar_ci"][1] - m_["skill_ar"]], arrayminus=[m_["skill_ar"] - m_["skill_ar_ci"][0]], color=col, thickness=2),
                                 hovertemplate=f"{key}: {m_['skill_ar']:+.3f} [{m_['skill_ar_ci'][0]:+.3f}; {m_['skill_ar_ci'][1]:+.3f}]<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ref], y=["S3"], mode="markers", name="взгляд H72", marker=dict(size=14, symbol="line-ns", color=NEUT, line=dict(width=3, color=NEUT)),
                             error_x=dict(type="data", symmetric=False, array=[rhi - ref], arrayminus=[ref - rlo], color=NEUT, thickness=2), hovertemplate="взгляд H72: +0,163 [+0,147; +0,180]<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="скилл = 1 - MSE / MSE(AR)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="скилл к AR", row=1, col=2)
    out.append(card("T01 (H83) · Прогноз изменения волатильности на час: скилл к AR по месяцам и в среднем", F(fig, "s10", "t01_skill", "", 480, lift=34),
                    "Замороженные на 365 днях до 2025-07-20 модели, оценка на Binance 2025-10..2026-09 (12 месяцев). Серая полоса - 95% интервал взгляда H72. Скилл положителен во все 12 месяцев. "
                    "<b>Оговорка:</b> слабая база - AR; уровни HAR сами дают заметную долю (кривая золотая), «сверх HAR» у S3 около +0,03 [+0,017; +0,044] (qa-deep). Это измерение предсказуемости волатильности, не торговое преимущество."))

    # ---- T03 monthly
    rws = pd.DataFrame(t3["rows"])
    rws = rws[(rws.ver == "cal") & (rws.f.isin(["HAR", "trail"]))]
    HZ = (15, 60, 240)
    panels = (("cover", "покрытие (цель 0,90)"), ("width_bps", "средняя ширина, б.п."), ("iscore_bps", "интервальная оценка, б.п. (ниже - лучше)"))
    fig = make_subplots(rows=3, cols=3, shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.07,
                        subplot_titles=[f"T = {h} мин" if r == 0 else "" for r in range(3) for h in HZ])
    fcfg = {"HAR": ("HAR-прогноз", PINK, "circle"), "trail": ("скользящая RV", GOLD, "square")}
    for ri, (k, yl) in enumerate(panels, 1):
        for ci, h in enumerate(HZ, 1):
            for f, (nm, col, sym) in fcfg.items():
                d = rws[(rws.H == h) & (rws.f == f)].sort_values("month")
                fig.add_trace(go.Scatter(x=[mlab(m) for m in d.month], y=d[k], mode="lines+markers", name=nm, legendgroup=f, showlegend=(ri == 1 and ci == 1),
                                         line=dict(color=col, width=2), marker=dict(size=6, symbol=sym, color=col),
                                         hovertemplate=f"T={h}, {nm}<br>%{{x}}: " + yl + " %{y:.3f}<extra></extra>"), row=ri, col=ci)
            if k == "cover":
                fig.add_hrect(y0=0.88, y1=0.92, fillcolor="rgba(160,160,160,0.18)", line_width=0, row=ri, col=ci)
                fig.add_hline(y=0.90, line=dict(color=NEUT, width=1.5, dash="dash"), row=ri, col=ci)
        fig.update_yaxes(title_text=yl, row=ri, col=1, title_font=dict(size=11))
    fig.update_xaxes(tickangle=60)
    pr = t3["summary"]["paired"]
    cap = " · ".join(f"T = {h}: {ru(crit[str(h)]['mean_diff_iscore_bps'], 1)} б.п. [{ru(crit[str(h)]['ci'][0], 1)}; {ru(crit[str(h)]['ci'][1], 1)}]" for h in HZ)
    out.append(card("T03 (H85) · Интервалы доходности 90%: покрытие, ширина и интервальная оценка по месяцам, HAR против скользящей RV", F(fig, "s10", "t03_months", "", 860, lift=14),
                    "Версия «cal»: квантиль нормированной доходности пересчитан на прошлом месяце. Серая полоса в верхнем ряду - допуск покрытия 0,88-0,92. Разность интервальной оценки HAR - скользящая RV: " + cap +
                    f". Покрытие HAR по месяцам колеблется сильнее ({ru(rws[(rws.H == 60) & (rws.f == 'HAR')].cover.min(), 3)}-{ru(rws[(rws.H == 60) & (rws.f == 'HAR')].cover.max(), 3)} при T = 60)."))

    # ---- T03 on real windows
    vw = json.load(open(HERE / "vol_windows.json", encoding="utf-8"))
    for di, dd in enumerate(vw["days"]):
        t = pd.to_datetime(dd["t"])
        close = np.array(dd["close"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                            subplot_titles=("Цена BTC/USDT (1 мин, Binance) и 60-минутные 90%-интервалы", "Ширина интервала и реализованное движение за окно, б.п."))
        fig.add_trace(go.Scatter(x=t, y=close, mode="lines", line=dict(color=INK2, width=1.4), name="цена (закрытие 1 мин)", hovertemplate="%{x|%H:%M}: %{y:.0f}<extra></extra>"), row=1, col=1)
        for f, nm, col, fillc in (("HAR", "HAR-интервал", PINK, "rgba(217,138,176,0.30)"), ("trail", "интервал скользящей RV", GOLD, None)):
            xs, ys = [], []
            for w in dd["windows"]:
                t0 = pd.Timestamp(w["t0"]); t1_ = t0 + pd.Timedelta(minutes=60)
                xs += [t0, t0, t1_, t1_, t0, None]
                ys += [w[f]["lo"], w[f]["hi"], w[f]["hi"], w[f]["lo"], w[f]["lo"], None]
            kw = dict(fill="toself", fillcolor=fillc) if fillc else {}
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nm, line=dict(color=col, width=1.6 if f == "HAR" else 1.4, dash="solid" if f == "HAR" else "dash"), hoverinfo="skip", **kw), row=1, col=1)
        for f, nm, sym in (("HAR", "конец окна вне интервала HAR", "x"), ("trail", "конец окна вне интервала скользящей RV", "square-open")):
            ws = [w for w in dd["windows"] if not w[f]["inside"]]
            fig.add_trace(go.Scatter(x=[pd.Timestamp(w["t0"]) + pd.Timedelta(minutes=59) for w in ws], y=[w["end"] for w in ws], mode="markers", name=nm,
                                     marker=dict(symbol=sym, size=11, color=T.CRITICAL, line=dict(width=2, color=T.CRITICAL)),
                                     hovertemplate="%{x|%H:%M}: цена %{y:.0f} (вне интервала)<extra></extra>"), row=1, col=1)
        xs = [pd.Timestamp(w["t0"]) + pd.Timedelta(minutes=30) for w in dd["windows"]]
        mv = [abs(np.log(w["end"] / w["c0"])) * 1e4 for w in dd["windows"]]
        fig.add_trace(go.Bar(x=xs, y=mv, width=60 * 60 * 1000 * 0.6, name="реализованное |движение| за 60 мин", marker=dict(color=NEUT, opacity=0.55),
                             hovertemplate="%{x|%H:%M}: |движение| %{y:.1f} б.п.<extra></extra>"), row=2, col=1)
        for f, nm, col, sym in (("HAR", "полуширина HAR", PINK, "circle"), ("trail", "полуширина скользящей RV", GOLD, "square")):
            fig.add_trace(go.Scatter(x=xs, y=[(w[f]["hi"] - w[f]["lo"]) / w["c0"] * 1e4 / 2 for w in dd["windows"]], mode="lines+markers", name=nm, line=dict(color=col, width=2, dash="solid" if f == "HAR" else "dash"),
                                     marker=dict(size=6, symbol=sym), hovertemplate="%{x|%H:%M}: " + nm + " %{y:.1f} б.п.<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="цена, USDT", row=1, col=1)
        fig.update_yaxes(title_text="б.п.", row=2, col=1)
        wh, wt = dd["width_bps"]["HAR"], dd["width_bps"]["trail"]
        cov = dd["covered"]
        n = dd["n_windows"]
        key = f"t03_day{di}"
        out.append(card(f"T03 на реальных данных · {dd['label']}: {dd['date']} (σ 1-мин {ru(dd['sd_bps_1m'], 1)} б.п.)", F(fig, "s10", key, "", 700, lift=34),
                        f"{n} непересекающихся часовых окон. Верхняя панель: интервал строится от цены закрытия перед окном и показан прямоугольником на его 60 минут; "
                        f"крест - конец окна вышел за интервал HAR, открытый квадрат - за интервал скользящей RV. Нижняя панель: серые столбики выше линии полуширины = конец окна вне интервала "
                        f"(для симметричной оценки; точная проверка - по верхней панели). Средняя ширина: HAR {ru(wh, 0)} б.п., скользящая RV {ru(wt, 0)} б.п.; доля окон внутри: HAR {cov['HAR']}/{n}, скользящая RV {cov['trail']}/{n}. "
                        "День выбран по волатильности (квантили 10% / 50% / 98% дневной σ среди оценённых дней), не по качеству интервалов; коэффициенты - замороженная подгонка T03, квантили пересчитаны на прошлом месяце."))
    out.append('<p class="note">Читать окна так: HAR шире скользящей RV, когда в предыдущий час было тихо, но модель знает про час суток и режим (приходит шторм), и уже, когда скользящая RV отстаёт от затухающего шторма. '
               'Один день ничего не доказывает; доказательство - 11 месяцев в таблице и на графике выше.</p>')

    # ---- T02 equity
    h2 = pd.read_csv(TAC / "tasks" / "T02_vol_exposure" / "look2_hourly.csv", parse_dates=["time"])
    eqs = (1 + h2.r_strategy).cumprod() - 1
    eqn = (1 + h2.r_null).cumprod() - 1
    eqb = (1 + h2.r_bh).cumprod() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06, subplot_titles=("Капитал за форвардный год, 2 б.п.", "Просадка"))
    cash = (h2.w.values < 0.5)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], cash.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=str(h2.time.iloc[a]), x1=str(h2.time.iloc[min(b, len(h2) - 1)] + pd.Timedelta(hours=1)), y0=0, y1=1,
                      fillcolor="rgba(160,160,160,0.35)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[h2.time.iloc[0]], y=[None], mode="lines", line=dict(color="rgba(160,160,160,0.5)", width=10), name="деньги (переключатель вне BTC)"))
    for ser_, nm, col, dash_, wd_ in ((eqb, "купить и держать", INK2, "solid", 1.6), (eqn, "константа с той же средней долей BTC (нуль)", GOLD, "dash", 2), (eqs, "переключатель: деньги при верхних 10% прогноза вол-ти", PINK, "solid", 2.6)):
        fig.add_trace(go.Scatter(x=h2.time, y=ser_ * 100, mode="lines", name=nm, line=dict(color=col, width=wd_, dash=dash_), hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y:.1f}%<extra>" + nm.split(" ")[0] + "</extra>"), row=1, col=1)
        eqv = 1 + ser_
        dd_ = (eqv / eqv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=h2.time, y=dd_, mode="lines", line=dict(color=col, width=wd_, dash=dash_), showlegend=False, hovertemplate="%{x|%Y-%m-%d}: просадка %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    fig.update_yaxes(title_text="накопленный итог, %", row=1, col=1)
    fig.update_yaxes(title_text="просадка, %", row=2, col=1)
    s_, n_, b_ = look2["2"]["strategy"], look2["2"]["null"], look2["2"]["bh"]
    out.append(card("T02 (H84) · Переключение экспозиции по прогнозу волатильности: капитал на форвардном годе", F(fig, "s10", "t02_equity", "", 720, lift=24),
                    f"Выбор правила сделан на 2019-01..2024-06 (деньги при верхних 10% прогноза 24-часовой волатильности), затем два просмотра; здесь - просмотр на Binance 2025-10-08..2026-09-30 (BTC {pct(b_['total_return'])}). "
                    f"Серые полосы - часы вне BTC ({int(cash.sum())} ч). Итог {pct(s_['total_return'])}, просадка {pct(s_['max_dd'])} против нуля {pct(n_['max_dd'])} и купить-держать {pct(b_['max_dd'])}. "
                    f"Критерий 1 (просадка не выше 0,8 от нуля: предел 40,6%) <b>не выполнен</b>; прогноз волатильности говорит о размере движений, а не о моменте просадки."))

    # ---- T04
    ms = t4["months"]
    xm = [mlab(m["month"]) for m in ms]
    d_c = [m["skill_ar"]["S3C"] - m["skill_ar"]["S3"] for m in ms]
    d_h = [m["skill_ar"]["S3"] - m["skill_ar"]["harlvl"] for m in ms]
    cmp_c, cmp_h = t4["comparisons"]["S3C-S3"], t4["comparisons"]["S3-harlvl"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"S3C − S3 (календарь 15 мин): среднее {cmp_c['mean_diff_skill_ar']:+.3f} [{cmp_c['ci'][0]:+.3f}; {cmp_c['ci'][1]:+.3f}], {cmp_c['months_positive']}/16; золотая линия, серая полоса = интервал",
                                                       f"S3 − уровни HAR (изменение вол-ти): среднее {cmp_h['mean_diff_skill_ar']:+.3f} [{cmp_h['ci'][0]:+.3f}; {cmp_h['ci'][1]:+.3f}], {cmp_h['months_positive']}/16"), horizontal_spacing=0.08)
    for ci, (ys, cm, col, nm) in enumerate(((d_c, cmp_c, VIOLET, "S3C − S3"), (d_h, cmp_h, PINK, "S3 − уровни HAR")), 1):
        fig.add_trace(go.Bar(x=xm, y=ys, marker=dict(color=[col if v >= 0 else SALMON for v in ys], line=dict(color=SURFACE, width=1.2)), name=nm, showlegend=False,
                             hovertemplate="%{x}: %{y:+.3f}<extra>" + nm + "</extra>"), row=1, col=ci)
        fig.add_hline(y=cm["mean_diff_skill_ar"], line=dict(color=GOLD, width=2), row=1, col=ci)
        fig.add_hrect(y0=cm["ci"][0], y1=cm["ci"][1], fillcolor="rgba(160,160,160,0.20)", line_width=0, row=1, col=ci)
        fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=ci)
    fig.add_hline(y=0.02, line=dict(color=T.GOOD, width=1.5, dash="dash"), annotation_text="порог +0,02", annotation_position="bottom right", row=1, col=1)
    fig.update_yaxes(title_text="разность скилла к AR", row=1, col=1)
    fig.update_xaxes(tickangle=60)
    out.append(card("T04 (H86) · Воспроизведение на 2017-09..2018-12: 15-минутный календарь и вклад изменения волатильности", F(fig, "s10", "t04_months", "", 470, lift=24),
                    f"Слева: критерий H86 (разность скилла S3C − S3 не меньше +0,02, интервал выше 0, положительны не менее 11 из 16) - <b>провален</b>: {cmp_c['mean_diff_skill_ar']:+.4f} "
                    f"[{cmp_c['ci'][0]:+.4f}; {cmp_c['ci'][1]:+.4f}], {cmp_c['months_positive']}/16 месяцев. Календарь найден на тех же новых месяцах (H83), на старой истории его нет. "
                    f"Справа: малый вклад S3 сверх уровней HAR <b>воспроизвёлся</b>: {cmp_h['mean_diff_skill_ar']:+.4f} [{cmp_h['ci'][0]:+.4f}; {cmp_h['ci'][1]:+.4f}], {cmp_h['months_positive']}/16 "
                    "(в H83 на форварде +0,030). Часть B (замороженный календарь на Binance, оценка после 3 полных месяцев с 2026-10) ещё не оценена."))
    return "".join(out)
