"""Builds runs/tactical/report/outlook_report.html: ONE self-contained interactive report of the OUTLOOK tactical session
(owner request 2026-10-13: a positive outlook on geometry, physics, regression, synthetic indicators). Same look as
tactical_report.html (rcommon helpers, the CSS of build_report.py read as text, plotly embedded once, no external requests,
text in Russian). Every number is read from the committed result files of runs/tactical/outlook/O*/ or computed here from their
rows (paired month differences with 95% t-intervals); candles come from the committed windows (interp.json) or the cut price
file (hard cutoff 2025-07-20). O2, O6 and O7 read their REPORT.md and score files (asserts tie the recomputed numbers to the reports).

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
# D-014: h0/h1/h2 colours are only for horizons. Spans, models and experiments use the other theme series colours.
SPANC = {"sel": TAN, "look": PINK, "fwd": VIOLET}
MODC = {"R0": NEUT, "L0": NEUT, "HAR": TAN, "L1": PINK, "L2": VIOLET, "L2sh": NEUT}
EXPC = {"O1": PINK, "O2": "#c3c2b7", "O3": TAN, "O4": SALMON, "O5": VIOLET, "O6": INK2, "O7": "#8a8984", "O8": SALMON}
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


def fmt7(t, d, nd=2):
    """interval-score difference (bps; negative = first model better): mean [95% t], months with a - b < 0."""
    return f"{sg(t[0], nd)} [{sg(t[1], nd)}; {sg(t[2], nd)}], лучше {sum(v < 0 for v in d.values())}/{len(d)}"


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
R6 = jl("O6_geom_hourly/results.jsonl")
R7 = jl("O7_geom_intervals/results.jsonl")
interp6 = jf("O6_geom_hourly/interp.json")
sum7 = jf("O7_geom_intervals/summary.json")
sha6 = (OUT / "O6_geom_hourly/frozen_B.sha256").read_text().strip()
o6_report = txt("O6_geom_hourly/REPORT.md")
o7_report = txt("O7_geom_intervals/REPORT.md")
assert "Verdict: **fail**" in o6_report and "Verdict: **pass**" in o7_report
assert o2_report and "NO DETECTABLE EFFECT" in o2_report

# O2: per-slice scores of the 8-epoch runs (seeds 0, 1) and of the 4-epoch references (phys7 seed 0, phys7b seed 1)
TACD = OUT.parent


def o2_scores(path, seeds):
    out = {}
    for r in json.load(open(TACD / path, encoding="utf-8")):
        if r["seed"] in seeds:
            out.setdefault(r["slice"], []).append(r)
    return out


def o2_cells(epochs):
    """{(cell, slice): {metric: value}} averaged over the two seeds. Values: spearman_h (3), qlike_h (3), har_tod spearman_h/qlike_h."""
    res = {}
    for cell in ("on", "off"):
        if epochs == 8:
            sc = o2_scores(f"outlook/O2_phys_var8/scores_o2_none_{cell}.json", (0, 1))
        else:
            a = o2_scores(f"phys7/scores_phys7_none_{cell}.json", (0,))
            b = o2_scores(f"phys7b/scores_phys7b_none_{cell}.json", (1,))
            sc = {k: a[k] + b[k] for k in a}
        for sl, rs in sc.items():
            assert len(rs) == 2, (epochs, cell, sl)
            res[cell, sl] = {
                "sp": np.mean([r["net"]["spearman_h"] for r in rs], axis=0), "ql": np.mean([r["net"]["qlike_h"] for r in rs], axis=0),
                "hsp": np.mean([r["har_tod"]["spearman_h"] for r in rs], axis=0), "hql": np.mean([r["har_tod"]["qlike_h"] for r in rs], axis=0),
                "sp_all": np.mean([r["net"]["spearman"] for r in rs]), "ql_all": np.mean([r["net"]["qlike"] for r in rs]),
                "hsp_all": np.mean([r["har_tod"]["spearman"] for r in rs]), "hql_all": np.mean([r["har_tod"]["qlike"] for r in rs])}
    return res


O2S = {8: o2_cells(8), 4: o2_cells(4)}
O2_SLICES = sorted({sl for (_, sl) in O2S[8]})
assert len(O2_SLICES) == 6 and sorted({sl for (_, sl) in O2S[4]}) == O2_SLICES
O2J = jf("O2_phys_var8/report_tables.json")


def o2_onoff(ep, key="sp", h=None):
    """per-slice on - off; h None = mean over the three horizons (the key '<key>_all' for the pooled mean is not used)."""
    v = []
    for sl in O2_SLICES:
        a, b = O2S[ep]["on", sl][key], O2S[ep]["off", sl][key]
        v.append(float(np.mean(a - b)) if h is None else float(a[h] - b[h]))
    return np.array(v)


def o2_gap(ep, cell, key="sp"):
    hk = "h" + key
    return np.array([float(np.mean(O2S[ep][cell, sl][key] - O2S[ep][cell, sl][hk])) for sl in O2_SLICES])


O2_ONOFF = {(ep, h): tci(o2_onoff(ep, "sp", h)) for ep in (4, 8) for h in (None, 0, 1, 2)}
O2_D = tci(o2_onoff(8) - o2_onoff(4))
O2_QL = {ep: tci(o2_onoff(ep, "ql")) for ep in (4, 8)}
O2_GAP = {(ep, c): tci(o2_gap(ep, c)) for ep in (4, 8) for c in ("on", "off")}
O2_GAPQ = {(ep, c): tci(o2_gap(ep, c, "ql")) for ep in (4, 8) for c in ("on", "off")}
for (ep, h), nm in (((8, None), "8ep_seeds01_spearman_on_minus_off"), ((8, 0), "8ep_seeds01_spearman_h0_on_minus_off"), ((4, None), "4ep_seeds01_spearman_on_minus_off"),
                    ((8, 2), "8ep_seeds01_spearman_h2_on_minus_off"), ((4, 1), "4ep_seeds01_spearman_h1_on_minus_off")):
    assert np.allclose(O2_ONOFF[ep, h][:3], O2J[nm][:3], atol=1e-9), (nm, O2_ONOFF[ep, h], O2J[nm])
assert np.allclose(O2_D[:3], O2J["d8_minus_d4_spearman"][:3], atol=1e-9)
assert np.allclose(O2_GAP[8, "on"][:3], O2J["8ep_seeds01_on_spearman_net_minus_har"][:3], atol=1e-9)
assert np.allclose(O2_GAP[4, "off"][:3], O2J["4ep_seeds01_off_spearman_net_minus_har"][:3], atol=1e-9)
assert np.allclose(O2_GAPQ[8, "off"][:3], O2J["8ep_seeds01_off_qlike_net_minus_har"][:3], atol=1e-9)
assert abs(O2_ONOFF[8, None][0] - 0.0094) < 6e-5 and O2_ONOFF[8, None][3] == 3 and abs(O2_D[0] + 0.0062) < 6e-5
assert "+0.0094 [-0.0068, +0.0257]" in o2_report and "-0.0062 [-0.0238, +0.0114]" in o2_report

# O6: judgement months 2019-01..2020-12, L2 (L1 + hourly geometry) against L1
O6 = {H_: tci(list(paired(R6, "L2", "L1", span="judge", H=H_).values())) for H_ in (15, 60, 240)}
O6sh = {H_: tci(list(paired(R6, "L2sh", "L1", span="judge", H=H_).values())) for H_ in (15, 60, 240)}
O6q = {H_: tci(list(paired(R6, "L2", "L1", key="qlike", span="judge", H=H_).values())) for H_ in (15, 60, 240)}
assert abs(O6[240][0] + 0.0143) < 6e-5 and O6[240][3] == 8 and O6[240][4] == 24 and abs(O6[60][0] + 0.0039) < 6e-5 and abs(O6sh[240][0] + 0.0009) < 6e-5, O6
assert abs(O6q[240][0] + 0.0007) < 6e-5
assert "-0.0143 [-0.0231, -0.0055]" in o6_report

# O7: interval score (bps; lower is better), judgement months 2019-02..2020-12
O7R0 = {T_: paired(R7, "L1", "R0", key="iscore_bps", span="judge", H=T_) for T_ in (15, 60, 240)}
O7HAR = {T_: paired(R7, "L1", "HAR", key="iscore_bps", span="judge", H=T_) for T_ in (15, 60, 240)}
O7C = {T_: tci(list(O7R0[T_].values())) for T_ in (15, 60, 240)}
O7H = {T_: tci(list(O7HAR[T_].values())) for T_ in (15, 60, 240)}


def O7better(d):
    """months with a - b < 0 (a better)."""
    return int(sum(v < 0 for v in d.values()))

assert abs(O7C[60][0] + 1.75) < 6e-3 and O7better(O7R0[60]) == 18 and len(O7R0[60]) == 23, O7C
assert abs(O7H[60][0] - 0.65) < 6e-3 and O7better(O7HAR[60]) == 16 and abs(O7C[15][0] + 1.84) < 6e-3
assert abs(sum7["paired_T60_L1_R0"]["mean"] - O7C[60][0]) < 1e-9
assert "-1.75 bps on average, 95% t-interval [-2.64, -0.86]" in o7_report


def o7_pool(span, T_, m):
    rr = [r for r in R7 if r["span"] == span and r["H"] == T_ and r["model"] == m]
    return (sum(r["n_in"] for r in rr) / sum(r["n"] for r in rr), float(np.mean([r["width_bps"] for r in rr])), float(np.mean([r["iscore_bps"] for r in rr])), len(rr))


assert abs(o7_pool("judge", 60, "L1")[0] - 0.8925) < 6e-5 and abs(o7_pool("judge", 60, "HAR")[2] - 305.11) < 6e-3 and abs(o7_pool("judge", 240, "R0")[1] - 377.31) < 6e-3

# O8: a vol-band strategy on the L1 size forecast (tests B, C, A; the configuration was chosen on the selection months)
o8_report = txt("O8_volband_strategy/REPORT.md")
o8_sel = jl("O8_volband_strategy/selection.jsonl")
o8_tests = jl("O8_volband_strategy/tests.jsonl")
o8_choice = jf("O8_volband_strategy/choice.json")
o8_win = jf("O8_volband_strategy/windows.json")
o8_tab = txt("O8_volband_strategy/tables.txt")
o8_seltab = txt("O8_volband_strategy/tables_selection.txt")
O8CR = o8_tests[-1]["criteria"]
O8V = o8_tests[-1]["verdict"]
O8BH = o8_tests[-1]["buy_hold"]
O8R = {(r["span"], r["arm"], r["variant"]): r for r in o8_tests[:-1]}
O8_COST = 0.0002                              # 2 bps round trip, the primary cost of the SPEC
assert len(o8_sel) == 24 and len(O8R) == 24 and o8_choice["config"] == "BRK_k1.5_H60_s1.0"
assert "strategy works on independent data: no" in o8_report and "geometry adds to the strategy: no" in o8_report
assert not O8V["works"] and not O8V["adds"] and not any(r["target_met"] for r in O8R.values())
assert ("+1.971% [+0.050, +3.892]" in o8_tab and "-1.057% [-3.374, +1.260]" in o8_tab
        and "-1.242% [-3.401, +0.916]" in o8_tab and "-0.375% [-1.407, +0.657]" in o8_tab)
assert abs(O8CR["c1_primary"]["mean"] - 0.019709) < 1e-6 and O8CR["c1_primary"]["pos"] == 17 and O8CR["c1_primary"]["n"] == 26
_ch = [s for s in o8_sel if s["config"] == o8_choice["config"]][0]
assert abs(_ch["t_2bps"] - 3.49) < 5e-3 and abs(_ch["mean_m_2bps"] * 100 - 4.044) < 5e-4 and _ch["n_trades"] == 3645
assert max(s["t_2bps"] for s in o8_sel) == _ch["t_2bps"] and all(s["mean_m_2bps"] < 0 for s in o8_sel if s["family"] == "REV")
assert "+3.49" in o8_seltab and "All 12 band-reversion (REV) configurations lost" in o8_report

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
        ("O2", "Физические члены на голове дисперсии при 8 эпохах вместо 4", "bad", "нет детектируемого эффекта",
         f"вкл−выкл, Спирмен |r|: 8 эпох {sg(O2_ONOFF[8, None][0])} [{sg(O2_ONOFF[8, None][1])}; {sg(O2_ONOFF[8, None][2])}], 3/6 срезов; 4 эпохи {sg(O2_ONOFF[4, None][0])}; 8 эпох = 4 в пределах шума"),
        ("O3", "Синтетические индикаторы волатильности (обучаемые ядра памяти)", "bad", "нет",
         f"LK−R0 на взгляде {sg(O3['look'][0])} [{sg(O3['look'][1])}; {sg(O3['look'][2])}]: ядра ничего не выучили; побочная находка BANK (плотный HAR)"),
        ("O4", "Подтверждение BANK на 2017-09..2018-12", "bad", "не подтверждено",
         f"BANK−R0, H=60: {sg(A4[60][0])} [{sg(A4[60][1])}; {sg(A4[60][2])}]; один месяц (2017-09, 10,7 дня обучения) тянет среднее вниз; часть B заморожена"),
        ("O5", "Геометрия поверх лучшей базы (BANK) на 2018-07..2020-12", "good", "геометрия добавляет",
         f"L1−L0, H=60: {sg(J5[60][0])} [{sg(J5[60][1])}; {sg(J5[60][2])}], {J5[60][3]}/{J5[60][4]} месяцев; запас по порогу малый; часть B заморожена"),
        ("O6", "Часовая геометрия (G60) для H=240", "bad", "провал",
         f"L2−L1, H=240: {sg(O6[240][0])} [{sg(O6[240][1])}; {sg(O6[240][2])}], {O6[240][3]}/{O6[240][4]} месяцев; лестница геометрии останавливается на 15 минутах"),
        ("O7", "Геометрия как 90%-интервалы доходности", "part", "проходит, но крошечно",
         f"интервальная оценка L1−R0, T=60: {sg(O7C[60][0], 2)} б.п. [{sg(O7C[60][1], 2)}; {sg(O7C[60][2], 2)}], {O7better(O7R0[60])}/23 месяцев лучше (0,6% от 307 б.п.); "
         f"против HAR: {sg(O7H[60][0], 2)} б.п. [{sg(O7H[60][1], 2)}; {sg(O7H[60][2], 2)}] - не лучше"),
        ("O8", "Стратегия «пробой полосы» на прогнозе размера (L1), независимые данные B, C, A", "bad", "не работает; геометрия не добавляет",
         f"L1 нетто 2 б.п.: {pct(O8CR['c1_primary']['mean'])} % в месяц [{pct(O8CR['c1_primary']['lo'])}; {pct(O8CR['c1_primary']['hi'])}] на B + C, но при PESS-заливках {pct(O8CR['c1_PESS']['mean'])} %; "
         f"L1−SHUF {pct(O8CR['c2_L1_minus_SHUF']['mean'])} %, L1−R0 {pct(O8CR['c3_L1_minus_R0']['mean'])} %; цель владельца (попаданий &gt; 60 %, просадка &lt; 5 %) не выполнена нигде"),
    ]
    rows = [[f"<b>{i}</b>", n, badge(k, t), d] for i, n, k, t, d in verdicts]
    board = table(rows, ["", "вопрос", "вердикт", "ключевое число"])

    # forest of the headline effects: three panels because the units differ
    itA = [("O1 G*−R0 · подбор", O1["sel", 60], "O1"), ("O1 G*−R0 · взгляд", O1["look", 60], "O1"), ("O1 G*−R0 · вперёд", O1["fwd", 60], "O1"),
           ("O3 LK−R0 · взгляд", O3["look"], "O3"), ("O3 LK−R0 · вперёд", O3["fwd"], "O3"), ("O3 BANK−R0 · взгляд", O3b["look"], "O3"), ("O3 BANK−R0 · вперёд", O3b["fwd"], "O3"),
           ("O4 BANK−R0 · 2017-09..2018-12", A4[60], "O4"), ("O5 L1−L0 · 2018-07..2020-12", J5[60], "O5"),
           ("O6 L2−L1 · H=60 (вторичный)", O6[60], "O6"), ("O6 L2−L1 · H=240 (основной)", O6[240], "O6")]
    itB = [("O2 вкл−выкл · 4 эпохи", O2_ONOFF[4, None], "O2"), ("O2 вкл−выкл · 8 эпох", O2_ONOFF[8, None], "O2")]
    itC = [("O7 L1−R0 · T=60", O7C[60], "O7"), ("O7 L1−HAR · T=60", O7H[60], "O7")]
    fig = make_subplots(rows=3, cols=1, row_heights=[0.66, 0.15, 0.15], vertical_spacing=0.13,
                        subplot_titles=["А. Спирмен по ln RV (больше - лучше); H=60, кроме O6 H=240",
                                        "Б. O2: Спирмен головы дисперсии с |r| на 10/15/20 мин (другая метрика), 6 срезов",
                                        "В. O7: интервальная оценка 90%-интервала, базисные пункты (меньше - лучше)"])
    for r, items, nd in ((1, itA, 4), (2, itB, 4), (3, itC, 2)):
        for nm, t, ex in items:
            c = EXPC[ex]
            cnt = f"единиц с плюсом {t[3]} из {t[4]}" if r < 3 else f"месяцев с отрицательной разностью {O7better(O7R0[60] if 'R0' in nm else O7HAR[60])} из {t[4]}"
            fig.add_trace(go.Scatter(x=[t[0]], y=[nm], mode="markers", marker=dict(color=c, size=11, symbol="diamond"),
                                     error_x=dict(type="data", symmetric=False, array=[t[2] - t[0]], arrayminus=[t[0] - t[1]], color=c, thickness=2.2),
                                     showlegend=False, hovertemplate=nm + f"<br>{ru(t[0], nd, True)} [{ru(t[1], nd, True)}; {ru(t[2], nd, True)}]<br>{cnt}<extra></extra>"), row=r, col=1)
        fig.add_vline(x=0, line=dict(color=NEUT, width=1), row=r, col=1)
        fig.update_yaxes(autorange="reversed", row=r, col=1)
    fig.add_vline(x=0.005, line=dict(color=T.GOOD, width=1, dash="dash"), row=1, col=1, annotation_text="порог +0,005", annotation_font_color=T.GOOD)
    fig.add_vline(x=0.010, line=dict(color=T.GOOD, width=1, dash="dash"), row=2, col=1, annotation_text="порог SPEC +0,010", annotation_font_color=T.GOOD)
    fig.update_xaxes(title_text="выигрыш по Спирмену, среднее по месяцам ± 95% t-интервал", row=1, col=1)
    fig.update_xaxes(title_text="вкл − выкл, среднее по срезам ± 95% t-интервал", row=2, col=1)
    fig.update_xaxes(title_text="разность интервальной оценки, б.п. (отрицательное - первая модель лучше)", row=3, col=1)
    fig.update_annotations(font_size=12, font_color=INK2)
    f1 = F(fig, "o0", "ov_forest", "Все ключевые эффекты", 860, subtitle="парные разности; три панели, потому что метрики разные; эффекты порядка 0,005 на базе 0,73-0,80")

    tiles = [(f"{sg(O1['look', 60][0])}", "O1: выигрыш геометрии, взгляд (H=60)", f"вперёд {sg(O1['fwd', 60][0])}; 12/13 и 12/12 месяцев в плюсе (взгляд и вперёд)"),
             (f"{sg(J5[15][0], 4)}", "O5: выигрыш геометрии при H=15", f"{J5[15][3]}/{J5[15][4]} месяцев; при H=240 ничего ({sg(J5[240][0])})"),
             (f"{sg(O7C[60][0], 2)} б.п.", "O7: геометрия как интервалы, T=60", f"{O7better(O7R0[60])}/23 месяцев лучше, 0,6% оценки; против HAR {sg(O7H[60][0], 2)} б.п. - не лучше"),
             (f"{sg(O6[240][0])}", "O6: часовая геометрия, H=240", f"{O6[240][3]}/{O6[240][4]} месяцев в плюсе: вредит; на H=60 {sg(O6[60][0])}"),
             (f"{pct(O8CR['c1_PESS']['mean'])} %", "O8: стратегия, честные заливки (PESS)", f"основные заливки {pct(O8CR['c1_primary']['mean'])} %/мес на B + C; геометрия не добавляет: L1−R0 {pct(O8CR['c3_L1_minus_R0']['mean'])} %"),
             (f"{sg(O2_ONOFF[8, None][0])}", "O2: физика, 8 эпох", f"4 эпохи {sg(O2_ONOFF[4, None][0])}; разность {sg(O2_D[0])} [{sg(O2_D[1])}; {sg(O2_D[2])}]: шум"),
             (f"{sg(O3['look'][0])}", "O3: синтетические ядра", "ядра остались около равномерных: «нет»")]
    th = "".join(f'<div class="tile"><div class="big">{a}</div><div class="lab">{b}</div><div class="sub">{c}</div></div>' for a, b, c in tiles)

    body = ('<div class="lead">Запрос владельца: найти <b>положительный прогноз</b> (outlook) для четырёх идей: <b>геометрия индикаторов</b>, <b>физика</b>, '
            '<b>регрессия</b> и <b>синтетические индикаторы</b>. Общий вывод: геометрия (15-минутная кривизна тренда EMA) помогает предсказывать <b>размер движений</b> (волатильность) '
            'на горизонтах 15 и 60 минут, на 240 минутах - нет. <b>Направление не заявляется</b>: ни один результат не про направление цены. Стратегия на прогнозе размера (O8, пробой полосы) на независимых данных не работает.</div>'
            f'<div class="tiles">{th}</div>'
            + card("Доска вердиктов O1-O8", board, "Единица - месяц (O2: срез); 95% t-интервалы парных разностей; «k/n» - единиц с плюсом. Всё одно зерно и только CPU; O2 - GPU, 2 зерна.")
            + card("Все ключевые эффекты, три оси", f1, "Панель А - выигрыш по Спирмену против опорной модели (цель ln RV). Порог из SPEC - +0,005 (O1/O3/O4), +0,003 (O5 и O6). "
                   "Панель Б - O2, другая метрика и данные. Панель В - O7, базисные пункты.")
            + card("Общий вывод", "<ul class='note'>"
                   f"<li><b>Геометрия работает на коротких масштабах:</b> 15-минутная кривизна тренда EMA добавляет к прогнозу размера движения на 15-60 минутах поверх лучшей линейной базы "
                   f"(O1 {sg(O1['look', 60][0])} на взгляде и O5 {sg(J5[60][0])} на 2018-2020; обе с QA PASS). Эффект стабилен по знаку, мал по размеру.</li>"
                   f"<li><b>Как интервалы - крошечный выигрыш:</b> O7, {sg(O7C[60][0], 2)} б.п. при оценке около 307 б.п. (0,6%); не лучше HAR ({sg(O7H[60][0], 2)} б.п. [{sg(O7H[60][1], 2)}; {sg(O7H[60][2], 2)}]).</li>"
                   f"<li><b>Часовая геометрия вредит:</b> O6, L2−L1 {sg(O6[240][0])} при H=240 и {sg(O6[60][0])} при H=60; лестница геометрии останавливается на 15 минутах.</li>"
                   f"<li><b>Физика - малый эффект раннего обучения:</b> O2, {sg(O2_ONOFF[4, None][0])} на 4 эпохах и {sg(O2_ONOFF[8, None][0])} на 8; не растёт с обучением и не отличим от нуля.</li>"
                   "<li><b>Синтетические ядра - нет</b> (O3). <b>BANK не подтверждён ретроспективно</b> (O4).</li>"
                   f"<li><b>Стратегия на прогнозе размера не работает</b> (O8): пробой полосы m(1 ± 1,5σ̂) даёт {pct(O8CR['c1_primary']['mean'])} % в месяц с основными заливками, но {pct(O8CR['c1_PESS']['mean'])} % при честных (PESS); "
                   f"L1 не лучше ни перемешанной σ̂, ни σ̂ без геометрии: размер прогнозируем, но на споте без направления это не деньги.</li>"
                   "<li><b>Ждут проспективной оценки:</b> score_B для O4, O5, O6 и O7 с 2026-10/11 (см. «Что ещё не готово»).</li>"
                   "<li><b>Оговорки:</b> прибавка O1 затухает на форварде и не держится против бустинга; порог размера пройден на грани; интервалы по месяцам оптимистичны из-за перекрытия меток.</li></ul>"))
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
    GC = {"price_bands": TAN, "volatility": PINK, "oscillators": TAN, "volume": VIOLET}
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
    parts = []
    lab = [f"срез {sl[:7]}" for sl in O2_SLICES]
    EPC = {4: TAN, 8: PINK}
    EPN = {4: "4 эпохи (зёрна 0 и 1)", 8: "8 эпох (зёрна 0 и 1)"}
    # fig 1: per slice on - off, and by horizon
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.1,
                        subplot_titles=["по срезам: вкл − выкл, среднее по трём горизонтам", "по горизонтам 10/15/20 минут: среднее по срезам ± 95% t"])
    for ep in (4, 8):
        v = o2_onoff(ep)
        fig.add_trace(go.Bar(x=lab, y=v, name=EPN[ep], legendgroup=str(ep), marker=dict(color=EPC[ep], line=dict(color=SURFACE, width=0.8)),
                             hovertemplate="%{x}: %{y:+.4f}<extra>" + EPN[ep] + "</extra>"), row=1, col=1)
        xs = ["среднее", "h0 (10 мин)", "h1 (15 мин)", "h2 (20 мин)"]
        ts = [O2_ONOFF[ep, h] for h in (None, 0, 1, 2)]
        fig.add_trace(go.Bar(x=xs, y=[t[0] for t in ts], name=EPN[ep], legendgroup=str(ep), showlegend=False, marker=dict(color=EPC[ep], line=dict(color=SURFACE, width=0.8)),
                             error_y=dict(type="data", symmetric=False, array=[t[2] - t[0] for t in ts], arrayminus=[t[0] - t[1] for t in ts], color=INK2),
                             text=[ru(t[0], 4, True) for t in ts], textposition="outside",
                             hovertemplate="%{x}: %{y:+.4f}<extra>" + EPN[ep] + "</extra>"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=c)
        fig.add_hline(y=0.010, line=dict(color=T.GOOD, width=1, dash="dot"), row=1, col=c)
    fig.add_annotation(xref="x2 domain", yref="y2", x=1.0, y=0.010, text="порог SPEC +0,010", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=T.GOOD, size=11))
    fig.update_layout(barmode="group", yaxis_title="вкл − выкл, Спирмен с |r|")
    fig.update_annotations(font_size=12, font_color=INK2)
    f1 = F(fig, "o2", "o2_onoff", "O2 эффект физики на голове дисперсии: 4 и 8 эпох", 470, lift=10,
           subtitle="одни и те же 6 срезов и те же два зерна; 4 эпохи - прежние прогоны phys7 (зерно 0) и phys7b (зерно 1)")
    rows = []
    for nm, h in (("среднее по горизонтам", None), ("h0 (10 мин)", 0), ("h1 (15 мин)", 1), ("h2 (20 мин)", 2)):
        rows.append([nm, fmt_ci(O2_ONOFF[4, h]), fmt_ci(O2_ONOFF[8, h])])
    rows.append(["QLIKE, среднее (меньше - лучше)", fmt_ci(O2_QL[4], 3), fmt_ci(O2_QL[8], 3)])
    t1 = table(rows, ["вкл − выкл", "4 эпохи, среднее [95% ИИ], срезов с плюсом", "8 эпох"])
    parts.append(card("Вердикт: нет детектируемого эффекта; 8 эпох = 4 в пределах шума", f1 + t1 +
                      f"<p class='note'><b>Парная разность d8 − d4</b> (те же срезы и зёрна): {fmt_ci(O2_D)} - интервал содержит 0. Эффект на 8 эпохах {sg(O2_ONOFF[8, None][0])} "
                      f"[{sg(O2_ONOFF[8, None][1])}; {sg(O2_ONOFF[8, None][2])}] ниже порога +0,010 и интервал содержит 0; на 4 эпохах эффект был "
                      f"{sg(O2_ONOFF[4, None][0])} [{sg(O2_ONOFF[4, None][1])}; {sg(O2_ONOFF[4, None][2])}] на тех же срезах и зёрнах. Эффект не растёт с обучением; точечная оценка чуть сжимается, "
                      "но это не установлено (2 зерна, 6 срезов).</p>",
                      "Цель - ранговая корреляция головы дисперсии с |r| на отложенном блоке фолда −2; физика включена против шести лямбд, равных 0; цена не предсказывается (PRICE_HEAD none)."))

    # fig 2: net minus HAR+tod per cell, 4 vs 8 epochs (Spearman and QLIKE)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["Спирмен: сеть − HAR+tod (больше - лучше)", "QLIKE: сеть − HAR+tod (меньше - лучше)"])
    cells = [("on", "физика вкл"), ("off", "физика выкл")]
    for ci, G, G0 in ((1, O2_GAP, "sp"), (2, O2_GAPQ, "ql")):
        for ep in (4, 8):
            ts = [G[ep, c] for c, _ in cells]
            fig.add_trace(go.Bar(x=[n for _, n in cells], y=[t[0] for t in ts], name=EPN[ep], legendgroup=str(ep), showlegend=(ci == 1),
                                 marker=dict(color=EPC[ep], line=dict(color=SURFACE, width=0.8)),
                                 error_y=dict(type="data", symmetric=False, array=[t[2] - t[0] for t in ts], arrayminus=[t[0] - t[1] for t in ts], color=INK2),
                                 text=[ru(t[0], 3, True) for t in ts], textposition="outside",
                                 hovertemplate="%{x}: %{y:+.4f}<extra>" + EPN[ep] + "</extra>"), row=1, col=ci)
            for c, n in cells:
                v = o2_gap(ep, c, G0)
                fig.add_trace(go.Scatter(x=[n] * len(v), y=v, mode="markers", showlegend=False, legendgroup=str(ep),
                                         marker=dict(color=INK2, size=5, opacity=0.55, symbol="circle-open"), hovertext=lab,
                                         hovertemplate="%{hovertext}: %{y:+.4f}<extra>" + EPN[ep] + "</extra>"), row=1, col=ci)
        fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=ci)
    fig.update_layout(barmode="group", yaxis_title="сеть − HAR+tod, Спирмен", yaxis2_title="сеть − HAR+tod, QLIKE")
    fig.update_annotations(font_size=12, font_color=INK2)
    f2 = F(fig, "o2", "o2_gap", "O2 сеть против HAR+tod: 4 и 8 эпох", 470, lift=10,
           subtitle="столбец - среднее по 6 срезам ± 95% t; кружки - отдельные срезы")
    rows = []
    for c, n in cells:
        rows.append([n, fmt_ci(O2_GAP[4, c]), fmt_ci(O2_GAP[8, c]), sg(float(np.mean(o2_gap(8, c)) - np.mean(o2_gap(4, c))), 4),
                     fmt_ci(O2_GAPQ[4, c], 3), fmt_ci(O2_GAPQ[8, c], 3)])
    t2 = table(rows, ["ячейка", "Спирмен, 4 эпохи", "8 эпох", "изменение 8 − 4", "QLIKE, 4 эпохи (плюс - хуже)", "8 эпох"])
    parts.append(card("Дольше обучение не закрывает разрыв с HAR", f2 + t2,
                      "Сеть остаётся на 0,06-0,07 по Спирмену ниже HAR+tod; на 8 эпохах разрыв чуть шире (в пределах шума). Больший разрыв, чем эффект физики, - главный вывод по этой оси."))

    parts.append(card("Прогоны и ограничения", "<ul class='note'><li>24 прогона (6 срезов × 2 зерна × вкл/выкл), все прошли, нет нефинитных шагов, нет константной дисперсии, 8 эпох в каждом (раннего останова нет).</li>"
                      "<li>GPU: 2 ч 23 мин по часам (бюджет 3,1 ч), 2 процесса, ниже обычного приоритета. Отклонений от SPEC нет.</li>"
                      "<li>Ограничения: 2 зерна и 8 эпох вместо полной длины; интервал широкий (sd 0,0155 по 6 срезам). Обслуживаемая (лучшая по валидации) эпоха в результатах не записана.</li></ul>"
                      "<p class='note'>Ниже полный REPORT.md, как лежит в репозитории.</p>" + md(o2_report), None))
    return "".join(parts)


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
        fig.add_trace(go.Scatter(x=taus + taus[::-1], y=list(q90) + list(q10[::-1]), fill="toself", fillcolor=T.rgba(TAN, 0.34), line=dict(width=0), name="10-90% по месяцам",
                                 legendgroup="q", showlegend=(k == 0), hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=taus, y=q50, mode="lines", line=dict(color=TAN, width=1.6), name="медиана по месяцам", legendgroup="med", showlegend=(k == 0),
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
    series = [("LK − R0 (кандидат)", "LK", "R0", PINK), ("BANK − R0 (контроль)", "BANK", "R0", TAN), ("LK − BANK (добавляет ли синтетика)", "LK", "BANK", VIOLET)]
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
    IC = [VIOLET, PINK, TAN, "#8a8984", SALMON, "#c3c2b7"]
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
    cols = [SALMON if m == "2017-09" else TAN for m in ms]
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
    fig = go.Figure(go.Bar(x=["S3 ridge", "L0 = BANK ridge", "N0 = HGB на S3+bank"], y=[s[m_] for m_ in ("S3", "L0", "N0")], marker=dict(color=[NEUT, TAN, SALMON], line=dict(color=SURFACE, width=1)),
                           text=[ru(s[m_], 4) for m_ in ("S3", "L0", "N0")], textposition="outside", hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(yaxis=dict(title="средний Спирмен (подбор, H=60)", range=[0.70, 0.745]))
    f0 = F(fig, "o5", "o5_base", "O5 выбор базы на подборе", 340, subtitle="42 месяца 2021-01..2024-06; ось не от нуля")
    parts.append(card("Выбор базы (только подбор)", f0 + table(rows, ["модель", "Спирмен", "QLIKE", ""]),
                      "B* = L0 (ridge BANK); кандидат L1 = L0 + геометрия (G1 + G15, 328 столбцов). Бустинг хуже ridge на подборе."))

    d = paired(R5, "L1", "L0", span=JSP, H=60)
    dq = paired(R5, "L1", "L0", span=JSP, H=60, key="qlike")
    ms = list(d)
    fig = go.Figure()
    for y, c_ in (("2018", TAN), ("2019", PINK), ("2020", VIOLET)):
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


# ======================================================================================================== section O6
def sec_o6():
    parts = []
    # monthly L2 - L1 at H=240 (Spearman, QLIKE)
    d = paired(R6, "L2", "L1", span="judge", H=240)
    dq = paired(R6, "L2", "L1", key="qlike", span="judge", H=240)
    ds = paired(R6, "L2sh", "L1", span="judge", H=240)
    ms = list(d)
    assert len(ms) == 24 and ms[0] == "2019-01" and ms[-1] == "2020-12"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Спирмен: L2 − L1 (больше - лучше; порог +0,003)", "QLIKE: L2 − L1 (меньше - лучше)"])
    fig.add_trace(go.Bar(x=ms, y=[d[m] for m in ms], name="L2 − L1 (L1 + часовая геометрия)", marker=dict(color=MODC["L2"], line=dict(color=SURFACE, width=0.5)),
                         hovertemplate="%{x}: %{y:+.4f}<extra>L2−L1</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ms, y=[ds[m] for m in ms], mode="lines+markers", name="контроль: L2sh − L1 (G60 перемешана)", line=dict(color=NEUT, width=1.6), marker=dict(size=5),
                             hovertemplate="%{x}: %{y:+.4f}<extra>L2sh−L1</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=ms, y=[dq[m] for m in ms], name="QLIKE L2 − L1", showlegend=False, marker=dict(color=MODC["L2"], line=dict(color=SURFACE, width=0.5)),
                         hovertemplate="%{x}: %{y:+.4f}<extra>QLIKE</extra>"), row=2, col=1)
    vlines_band(fig, ms[0], ms[-1], O6[240], MODC["L2"], "")
    fig.add_hline(y=0.003, line=dict(color=T.GOOD, width=1, dash="dot"), row=1, col=1, annotation_text="порог +0,003", annotation_font_color=T.GOOD)
    for r in (1, 2):
        fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=r, col=1)
    fig.update_xaxes(type="category", tickangle=-60, nticks=30, row=2, col=1)
    fig.update_yaxes(title_text="L2 − L1, Спирмен", row=1, col=1)
    fig.update_yaxes(title_text="L2 − L1, QLIKE", row=2, col=1)
    fig.update_annotations(font_size=12, font_color=INK2)
    f1 = F(fig, "o6", "o6_monthly", "O6 помесячно: L2 − L1 при H=240", 640, lift=10,
           subtitle="24 месяца суждения 2019-01..2020-12; штрих и заливка - среднее и 95% t по месяцам")
    qm = O6q[240]
    parts.append(card("Вердикт: провал", f1,
                      f"<b>Критерий (H=240, L2 − L1):</b> Спирмен {fmt_ci(O6[240])} при пороге ≥ +0,003 с нижней границей > 0 и ≥ 16 месяцев из 24 - не выполнен; месяцев с плюсом {O6[240][3]}/24 (нужно 16). "
                      f"QLIKE {sg(qm[0])} [{sg(qm[1])}; {sg(qm[2])}] - выполнен только по среднему. Контроль с перемешанной G60 (L2sh − L1) {fmt_ci(O6sh[240])} - внутри ±0,003, то есть вердикт «провал», а не «сломано». "
                      "Вред приходит от самих часовых признаков, а не от числа столбцов (прочтение руководителя, не проверялось)."))

    # per-horizon summary
    fig = go.Figure()
    for nm, G, col in (("L2 − L1 (часовая геометрия)", O6, MODC["L2"]), ("L2sh − L1 (контроль, G60 перемешана)", O6sh, NEUT)):
        xs = [f"H={H_} мин" for H_ in (15, 60, 240)]
        ts = [G[H_] for H_ in (15, 60, 240)]
        fig.add_trace(go.Bar(x=xs, y=[t[0] for t in ts], name=nm, marker=dict(color=col, line=dict(color=SURFACE, width=1)),
                             error_y=dict(type="data", symmetric=False, array=[t[2] - t[0] for t in ts], arrayminus=[t[0] - t[1] for t in ts], color=INK2),
                             text=[ru(t[0], 4, True) for t in ts], textposition="outside", hovertemplate="%{x}: %{y:+.4f}<extra>" + nm + "</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(barmode="group", yaxis_title="разность Спирмена против L1")
    f2 = F(fig, "o6", "o6_horizons", "O6 по горизонтам", 420, subtitle="24 месяца суждения; среднее ± 95% t")
    base = {}
    for H_ in (15, 60, 240):
        for m_ in ("L1", "L2", "L2sh"):
            rr = [r for r in R6 if r["span"] == "judge" and r["H"] == H_ and r["model"] == m_]
            base[H_, m_] = (float(np.mean([r["spearman"] for r in rr])), float(np.mean([r["qlike"] for r in rr])), len(rr))
    assert abs(base[240, "L1"][0] - 0.6280) < 6e-5 and abs(base[60, "L2"][1] - 0.2851) < 6e-5 and base[15, "L2sh"][2] == 24
    rows = [[f"H={H_}"] + [f"{ru(base[H_, m_][0], 4)} / {ru(base[H_, m_][1], 4)}" for m_ in ("L1", "L2", "L2sh")] + [fmt_ci(O6[H_]), fmt_ci(O6q[H_])] for H_ in (15, 60, 240)]
    t1 = table(rows, ["горизонт", "L1: Спирмен / QLIKE", "L2", "L2sh", "L2 − L1, Спирмен", "L2 − L1, QLIKE (меньше - лучше)"])
    # spent spans (descriptive)
    sp_n = {"spent1": "2021-01..2024-06", "spent2": "2024-07..2025-07"}
    rows2 = []
    for sp in ("spent1", "spent2"):
        rows2.append([sp_n[sp]] + [fmt_ci(tci(list(paired(R6, "L2", "L1", span=sp, H=H_).values()))) for H_ in (15, 60, 240)])
    assert abs(tci(list(paired(R6, "L2", "L1", span="spent1", H=240).values()))[0] + 0.0085) < 6e-5
    t2 = table(rows2, ["потраченный отрезок (описательно)", "H=15", "H=60", "H=240"])
    parts.append(card("По горизонтам и по потраченным отрезкам", f2 + t1 + t2,
                      f"Побочно (без вердикта): на H=60 L2 − L1 {fmt_ci(O6[60])}, Холм p 0,021; на H=15 {fmt_ci(O6[15])}. Знак отрицательный на каждом отрезке и каждом горизонте."))

    # coefficients
    top = interp6["top_g60_coefficients"]
    assert len(top) == 15 and top[0]["feature"] == "g60.ema20.slope3" and abs(top[0]["coef_standardised"] + 1.1428) < 5e-4
    cf = [t["coef_standardised"] for t in top]
    fig = go.Figure(go.Bar(y=[t["feature"] for t in top], x=cf, orientation="h",
                           marker=dict(color=[MODC["L2"] if c > 0 else PINK for c in cf], line=dict(color=SURFACE, width=1)),
                           text=[ru(c, 2, True) for c in cf], textposition="outside",
                           hovertemplate="%{y}<br>коэффициент %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="стандартизованный коэффициент замороженной L2 (H=240); фиолетовый - плюс, розовый - минус", margin=dict(l=190))
    f3 = F(fig, "o6", "o6_coefs", "O6 топ-15 коэффициентов G60", 560, subtitle="подгонка на Binance 2025-10-05..2026-09-30, стандартизованные признаки")
    parts.append(card("Топ-15 коэффициентов часовой геометрии", f3,
                      "Коэффициенты велики и имеют противоположные знаки на коллинеарных часовых наклонах EMA (ema20.slope3 −1,14; ema10.slope10 +0,60; ema40.slope3 +0,56): "
                      "это подгонка нестационарной структуры, а не читаемый сигнал."))

    # real hourly windows
    W = interp6["windows"]
    picks = [("g60.ema20.slope3", "highest"), ("g60.ema20.slope3", "lowest"), ("g60.ema10.slope10", "highest")]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.035, subplot_titles=["a", "b", "c"])
    titles = []
    for c, (ft, wh) in enumerate(picks, 1):
        w = next(x for x in W[ft] if x["which"] == wh)
        A = np.array(w["hourly_ohlcv"], float)
        end = pd.Timestamp(w["last_bar_end"])
        assert end <= pd.Timestamp(C.CUTOFF)
        t = pd.date_range(end=end, periods=len(A), freq="60min")
        assert t.max() <= pd.Timestamp(C.CUTOFF) and len(A) == 60
        titles.append(f"{ft}: {ru(w['feature_value'], 2, True)} ({wh}); конец {str(end)[:10]}")
        fig.add_trace(go.Candlestick(x=t, open=A[:, 0], high=A[:, 1], low=A[:, 2], close=A[:, 3], showlegend=False,
                                     increasing=dict(line=dict(color=UP), fillcolor=UP), decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
                                     hovertemplate="%{x|%Y-%m-%d %H:%M}<br>O %{open:.0f} H %{high:.0f} L %{low:.0f} C %{close:.0f}<extra></extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=t, y=ema(A[:, 3], 20), mode="lines", line=dict(color=GOLD, width=2), name="EMA20", legendgroup="e20", showlegend=(c == 1), hovertemplate="EMA20 %{y:.0f}<extra></extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=t, y=ema(A[:, 3], 40), mode="lines", line=dict(color=PINK, width=2, dash="dash"), name="EMA40", legendgroup="e40", showlegend=(c == 1), hovertemplate="EMA40 %{y:.0f}<extra></extra>"), row=1, col=c)
        fig.update_xaxes(rangeslider_visible=False, row=1, col=c)
    for i, tt in enumerate(titles):
        fig.layout.annotations[i].text = tt
    fig.update_annotations(font_size=11, font_color=INK2)
    f4 = F(fig, "o6", "o6_windows", "O6 реальные часовые окна для двух главных признаков G60", 520, lift=6,
           subtitle="60 часовых свечей до конца окна; линии - EMA20 и EMA40 по закрытиям; данные не позже 2025-07-20")
    parts.append(card("Реальные часовые окна BTC", f4,
                      "Окна из interp.json (самые высокие и самые низкие значения признака на отрезке до 2025-07-20). Время свечей - по концу последнего часового бара, по часу назад (метки приблизительны: окна определены позициями баров)."))

    parts.append(card("Практический вывод и заморозка", "<p class='note'>Волатильная модель остаётся на G1 + G15 (L1 из O5); для H=240 нужно что-то другое, чем геометрия. "
                      f"Артефакт части B заморожен (frozen_B.json, sha256 <code>{sha6}</code>), но score_B.py не запускался: после провала он почти не нужен.</p>"))
    return "".join(parts)


# ======================================================================================================== section O7
def sec_o7():
    parts = []
    MODS = ("L1", "R0", "HAR")
    ms = sorted({r["month"] for r in R7 if r["span"] == "judge" and r["H"] == 60})
    assert len(ms) == 23 and ms[0] == "2019-02" and ms[-1] == "2020-12"
    # fig 1: monthly differences at T = 60, coverage, width
    dR, dH = O7R0[60], O7HAR[60]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=["Интервальная оценка, б.п.: разность по месяцам (отрицательное - L1 лучше)", "Покрытие по месяцам (цель 0,90; допуск 0,88-0,92)", "Средняя ширина интервала, б.п."])
    fig.add_trace(go.Bar(x=ms, y=[dR[m] for m in ms], name="L1 − R0", marker=dict(color=PINK, line=dict(color=SURFACE, width=0.5)), hovertemplate="%{x}: %{y:+.2f} б.п.<extra>L1−R0</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=ms, y=[dH[m] for m in ms], name="L1 − HAR", marker=dict(color=TAN, line=dict(color=SURFACE, width=0.5)), hovertemplate="%{x}: %{y:+.2f} б.п.<extra>L1−HAR</extra>"), row=1, col=1)
    for m_ in MODS:
        rr = {r["month"]: r for r in R7 if r["span"] == "judge" and r["H"] == 60 and r["model"] == m_}
        col = MODC[m_]
        fig.add_trace(go.Scatter(x=ms, y=[rr[m]["cover"] for m in ms], mode="lines+markers", name=m_, legendgroup=m_, line=dict(color=col, width=1.6), marker=dict(size=5),
                                 hovertemplate="%{x}: %{y:.3f}<extra>" + m_ + "</extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ms, y=[rr[m]["width_bps"] for m in ms], mode="lines+markers", name=m_, legendgroup=m_, showlegend=False, line=dict(color=col, width=1.6), marker=dict(size=5),
                                 hovertemplate="%{x}: %{y:.1f} б.п.<extra>" + m_ + "</extra>"), row=3, col=1)
    fig.add_hrect(y0=0.88, y1=0.92, fillcolor=T.rgba(T.GOOD, 0.10), line_width=0, row=2, col=1)
    fig.add_hline(y=0.90, line=dict(color=T.GOOD, width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=1)
    fig.update_xaxes(type="category", tickangle=-60, nticks=30, row=3, col=1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="б.п.", row=1, col=1)
    fig.update_yaxes(title_text="покрытие", row=2, col=1)
    fig.update_yaxes(title_text="ширина, б.п.", row=3, col=1)
    fig.update_annotations(font_size=12, font_color=INK2)
    f1 = F(fig, "o7", "o7_monthly", "O7 помесячно при T=60: оценка, покрытие, ширина", 900, lift=10,
           subtitle="23 месяца суждения 2019-02..2020-12; каждый месяц оценён по подгонке предыдущего месяца")
    p = {m_: o7_pool("judge", 60, m_) for m_ in MODS}
    parts.append(card("Вердикт: проходит, но крошечно; не лучше HAR", f1 +
                      table([[m_, ru(p[m_][0], 4), ru(p[m_][1], 2), ru(p[m_][2], 2)] for m_ in MODS], ["модель (T=60, суждение)", "покрытие (собранное)", "средняя ширина, б.п.", "средняя интервальная оценка, б.п."]),
                      f"<b>Критерий (T=60, L1 − R0):</b> {sg(O7C[60][0], 2)} б.п. [{sg(O7C[60][1], 2)}; {sg(O7C[60][2], 2)}], L1 лучше в {O7better(dR)} из 23 месяцев (нужно 15), "
                      f"собранное покрытие L1 {ru(p['L1'][0], 4)} внутри [0,88; 0,92] - выполнен. Размер: −1,75 б.п. при оценке около 307 б.п. (0,6%). Ширина не меняется ({sg(p['L1'][1] - p['R0'][1], 2)} б.п.): выигрыш - меньше и мельче промахи при том же покрытии. "
                      f"<b>Против HAR</b> L1 не лучше: {fmt7(O7H[60], dH)}; месяц 2020-03 один говорит в пользу HAR."))

    # fig 2: per-T differences
    fig = go.Figure()
    for nm, G, col in (("L1 − R0 (S3 + геометрия против S3)", O7C, PINK), ("L1 − HAR", O7H, TAN)):
        ts = [G[T_] for T_ in (15, 60, 240)]
        fig.add_trace(go.Bar(x=[f"T={T_} мин" for T_ in (15, 60, 240)], y=[t[0] for t in ts], name=nm, marker=dict(color=col, line=dict(color=SURFACE, width=1)),
                             error_y=dict(type="data", symmetric=False, array=[t[2] - t[0] for t in ts], arrayminus=[t[0] - t[1] for t in ts], color=INK2),
                             text=[ru(t[0], 2, True) for t in ts], textposition="outside", hovertemplate="%{x}: %{y:+.2f} б.п.<extra>" + nm + "</extra>"))
    fig.add_hline(y=0, line=dict(color=NEUT, width=1))
    fig.update_layout(barmode="group", yaxis_title="разность интервальной оценки, б.п. (меньше - лучше)")
    f2 = F(fig, "o7", "o7_horizons", "O7 по горизонтам", 430, subtitle="23 месяца суждения; среднее ± 95% t")
    rows = []
    for T_ in (15, 60, 240):
        pp = {m_: o7_pool("judge", T_, m_) for m_ in MODS}
        hp = sum7["summary"]["judge|L1-HAR"][str(T_)]["holm"]
        rows.append([f"T={T_}"] + [f"{ru(pp[m_][0], 3)} / {ru(pp[m_][1], 1)} / {ru(pp[m_][2], 1)}" for m_ in MODS] +
                    [fmt7(O7C[T_], O7R0[T_]), fmt7(O7H[T_], O7HAR[T_]), ru(hp, 3)])
    assert "0,65" in rows[1][5]
    t1 = table(rows, ["горизонт", "L1: покрытие / ширина / оценка", "R0", "HAR", "L1 − R0, б.п. [ИИ], месяцев лучше", "L1 − HAR, б.п.", "Холм p (L1−HAR)"])
    # descriptive spans
    rows2 = []
    spn = {"spent1": "2021-02..2024-06", "spent2": "2024-08..2025-07"}
    for sp in ("spent1", "spent2"):
        cells = []
        for T_ in (15, 60, 240):
            dd = paired(R7, "L1", "R0", key="iscore_bps", span=sp, H=T_)
            tt = tci(list(dd.values()))
            cells.append(f"{sg(tt[0], 2)} б.п., {O7better(dd)}/{len(dd)}")
        rows2.append([spn[sp]] + cells)
    assert rows2[0][2].startswith("−0,76") and rows2[1][3].startswith("−2,30")
    t2 = table(rows2, ["потраченный отрезок (описательно), L1 − R0", "T=15", "T=60", "T=240"])
    parts.append(card("По горизонтам: покрытие, ширина, оценка, различия", f2 + t1 + t2,
                      "T=15: L1 лучше R0 в 23 из 23 месяцев, но от HAR отличие на границе значимости ({}). На T=240 эффекта нет. На потраченных отрезках знак тот же; покрытие всех моделей 0,896-0,900."
                      .format(fmt7(O7H[15], O7HAR[15]))))

    # cover-width pooled figure over horizons
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=["Собранное покрытие (цель 0,90)", "Средняя ширина, б.п."])
    for m_ in MODS:
        xs = [f"T={T_} мин" for T_ in (15, 60, 240)]
        pp = [o7_pool("judge", T_, m_) for T_ in (15, 60, 240)]
        fig.add_trace(go.Bar(x=xs, y=[q[0] for q in pp], name=m_, legendgroup=m_, marker=dict(color=MODC[m_], line=dict(color=SURFACE, width=1)),
                             hovertemplate="%{x}: %{y:.4f}<extra>" + m_ + "</extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=xs, y=[q[1] for q in pp], name=m_, legendgroup=m_, showlegend=False, marker=dict(color=MODC[m_], line=dict(color=SURFACE, width=1)),
                             hovertemplate="%{x}: %{y:.1f} б.п.<extra>" + m_ + "</extra>"), row=1, col=2)
    fig.add_hline(y=0.90, line=dict(color=T.GOOD, width=1, dash="dot"), row=1, col=1)
    fig.update_yaxes(range=[0.86, 0.92], row=1, col=1, title_text="покрытие (ось не от нуля)")
    fig.update_yaxes(title_text="ширина, б.п.", row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_annotations(font_size=12, font_color=INK2)
    f3 = F(fig, "o7", "o7_cover_width", "O7 покрытие и ширина по горизонтам", 430, lift=10, subtitle="суждение 2019-02..2020-12; собранное покрытие = попавшие / все")
    parts.append(card("Покрытие и ширина по моделям", f3, "Все три модели калибруются на предыдущем месяце, поэтому покрытие почти одинаково (0,891-0,894); различия видны в ширине и в числе и размере промахов (интервальная оценка)."))

    parts.append(card("Оговорки O7", "<ul class='note'><li>Месяцы суждения - те же, по которым O5 ранжировал L1: это ещё один взгляд на них.</li>"
                      "<li>Интервалы t по месяцам оптимистичны: разности коррелированы (перекрытие меток и режимы волатильности).</li>"
                      "<li>Покрытие в отдельных месяцах от 0,80 до 0,95 у каждой модели; худшие 2020-03, 2020-11 и 2019-05.</li>"
                      "<li>Месяц M оценён по подгонке месяца M−1 (прогноз на месяц старше); вариант с подгонкой месяца M не запускался.</li>"
                      "<li>Проспективная часть: score_B.py сравнит замороженную L1 с L0 (не R0); калибровка на 2026-10, оценка 2026-11..2027-01, один раз. Это единственная непотраченная проверка.</li>"
                      "<li>Без QA-агента: результат малый и вторичный к O5 (QA PASS); критерий проверен руководителем.</li></ul>"))
    return "".join(parts)


# ======================================================================================================== section O8
O8D = OUT / "O8_volband_strategy"
O8ARMS = [("L1", "primary", "L1 (основной)", PINK, "solid"), ("L1", "PESS", "L1, PESS-заливки", PINK, "dash"),
          ("R0", "primary", "R0 (σ̂ без геометрии)", GOLD, "solid"), ("SHUF", "primary", "SHUF (σ̂ перемешана)", VIOLET, "solid"),
          ("TRAIL", "primary", "TRAIL (σ по прошлому)", NEUT, "solid")]
O8SPANS = [("B", "B: 2024-07..2025-07-20, шаг вперёд (вердикт)"), ("C", "C: 2025-10-05..2026-10-12, заморожено (вердикт)"),
           ("A", "A: 2018-07..2020-12, шаг вперёд (отдельно, не вердикт)")]
O8_BH_RANGE = {"B": ("2024-07-01", C.CUTOFF), "A": ("2018-07-01", "2020-12-31 23:59")}


def px(x):
    return f"{x:,.0f}".replace(",", " ")


def pct(x, nd=2, sign=True):
    return ru(x * 100, nd, sign)


def o8_sortable(rows, header):
    """rows: [[(shown, sort value), ...]]; a click on a header sorts the table (script at the end of the page)."""
    h = "".join(f"<th>{x}</th>" for x in header)
    b = "".join("<tr>" + "".join(f'<td data-v="{v}">{s}</td>' for s, v in r) + "</tr>" for r in rows)
    return f'<table class="rt sortable"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def o8_trade_traces(fig, ex, row, col, show_leg):
    """Candles of one real example, its two stop orders, the entry, the stop level and the exit."""
    b = pd.DataFrame(ex["bars"])
    t = pd.to_datetime(b["t"])
    assert t.max() <= pd.Timestamp(C.CUTOFF) and t.min() < pd.Timestamp(ex["entry_time"]) < t.max()
    kw = dict(row=row, col=col)
    fig.add_trace(go.Candlestick(x=t, open=b["o"], high=b["h"], low=b["l"], close=b["c"], showlegend=False,
                                 increasing=dict(line=dict(color=UP), fillcolor=UP), decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
                                 hovertemplate="%{x|%Y-%m-%d %H:%M}<br>O %{open:.0f} H %{high:.0f} L %{low:.0f} C %{close:.0f}<extra></extra>"), **kw)
    td, te, tx, tts = (pd.Timestamp(ex[k]) for k in ("decision_time", "entry_time", "exit_time", "time_stop_time"))
    U, L, m, sgm = ex["buy_stop_U"], ex["sell_stop_L"], ex["m"], ex["sigma"]
    assert abs(U - m * (1 + 1.5 * sgm)) < 1e-6 * m and abs(L - m * (1 - 1.5 * sgm)) < 1e-6 * m
    assert abs(ex["stop_px"] - ex["entry_px"] * (1 - ex["side"] * sgm)) < 1e-6 * m and tts - te == pd.Timedelta(minutes=60)
    td15 = td + pd.Timedelta(minutes=15)

    def seg(x0, x1, y, name, color, dash, width, show):
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", line=dict(color=color, width=width, dash=dash), name=name, legendgroup=name, showlegend=show,
                                 hovertemplate=name + ": %{y:.0f}<extra></extra>"), **kw)

    seg(td, tx, m, "якорь m (цена на момент решения)", MUTED, "dot", 1.6, show_leg)
    seg(td, td15, U, "стоп-ордера m(1 ± 1,5σ̂), живут 15 минут", GOLD, "dash", 2.4, show_leg)
    seg(td, td15, L, "стоп-ордера m(1 ± 1,5σ̂), живут 15 минут", GOLD, "dash", 2.4, False)
    seg(te, tx, ex["stop_px"], "защитный стоп: вход ∓ 1σ̂", SALMON, "dash", 2.4, show_leg)
    lo, hi = min(b["l"].min(), L), max(b["h"].max(), U)
    if tts <= t.max():
        fig.add_trace(go.Scatter(x=[tts, tts], y=[lo, hi], mode="lines", line=dict(color=VIOLET, width=1.6, dash="dot"), name="тайм-стоп: вход + 60 минут",
                                 legendgroup="тайм-стоп", showlegend=show_leg, hoverinfo="skip"), **kw)
    long_ = ex["side"] == 1
    fig.add_trace(go.Scatter(x=[te], y=[ex["entry_px"]], mode="markers", marker=dict(symbol="triangle-up" if long_ else "triangle-down", size=14, color=INK, line=dict(color=PAPER, width=1.5)),
                             name="вход" + (" (покупка)" if long_ else " (продажа)"), legendgroup="вход", showlegend=show_leg,
                             hovertemplate="вход %{x|%H:%M}, %{y:.0f}<extra></extra>"), **kw)
    xc = SALMON if ex["reason"] == "stop" else VIOLET
    fig.add_trace(go.Scatter(x=[tx], y=[ex["exit_px"]], mode="markers", marker=dict(symbol="x", size=13, color=xc, line=dict(color=xc, width=3)),
                             name="выход", legendgroup="выход", showlegend=show_leg, hovertemplate="выход %{x|%H:%M}, %{y:.0f}<extra></extra>"), **kw)
    return td, te, tx, tts, U, L


def o8_ex_text(ex):
    net = ex["gross_ret"] - O8_COST
    side = "покупка" if ex["side"] == 1 else "продажа"
    why = "сработал защитный стоп" if ex["reason"] == "stop" else "тайм-стоп через 60 минут"
    return side, why, ex["gross_ret"], net


def o8_schematic(ex):
    fig = make_subplots(rows=1, cols=1)
    td, te, tx, tts, U, L = o8_trade_traces(fig, ex, 1, 1, True)
    side, why, gross, net = o8_ex_text(ex)
    m, sgm = ex["m"], ex["sigma"]
    A = lambda **k: fig.add_annotation(xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor=INK2, font=dict(size=12, color=INK), bgcolor=T.rgba(PAPER, 0.85),
                                       bordercolor=AXIS, borderwidth=1, borderpad=3, align="left", **k)
    A(x=td, y=m, text=f"<b>1. Решение {td:%H:%M}</b><br>якорь m = {px(m)}<br>σ̂ (прогноз L1, 60 минут) = {ru(sgm * 100, 2)} %", ax=-30, ay=70, xanchor="right")
    A(x=td + pd.Timedelta(minutes=7), y=U, text=f"<b>2. Покупка-стоп</b> U = m(1 + 1,5σ̂)<br>= {px(U)}", ax=-120, ay=-30, xanchor="right")
    A(x=td + pd.Timedelta(minutes=7), y=L, text=f"<b>2. Продажа-стоп</b> L = m(1 − 1,5σ̂)<br>= {px(L)}", ax=-120, ay=30, xanchor="right")
    A(x=te, y=ex["entry_px"], text=f"<b>3. Вход {te:%H:%M}</b><br>цена дошла до уровня: {side} по {px(ex['entry_px'])};<br>второй ордер снят", ax=-30, ay=-110, xanchor="right")
    A(x=te + pd.Timedelta(minutes=20), y=ex["stop_px"], text=f"<b>4. Защитный стоп</b> на 1σ̂ от входа = {px(ex['stop_px'])}<br>" + ("не тронут" if ex["reason"] == "time" else "сработал"),
      ax=30, ay=60, xanchor="left")
    A(x=tx, y=ex["exit_px"], text=f"<b>5. Выход {tx:%H:%M}</b><br>{why}<br>брутто {pct(gross)} %, нетто (2 б.п.) {pct(net)} %", ax=-70, ay=-60, xanchor="right")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="цена BTC/USDT")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return F(fig, "o8", "o8_schematic", "Одна сделка на реальных барах BTC", 640,
             subtitle=f"{ex['span']}; {td:%Y-%m-%d}; минутные свечи от {str(pd.DataFrame(ex['bars'])['t'].iloc[0])[11:16]} до {str(pd.DataFrame(ex['bars'])['t'].iloc[-1])[11:16]} UTC")


def o8_examples():
    exs = o8_win["examples"]
    assert len(exs) == 6 and sum(e["span"].startswith("selection") for e in exs) == 3
    titles = []
    for i, ex in enumerate(exs):
        side, why, gross, net = o8_ex_text(ex)
        titles.append(f"{'подбор' if ex['span'].startswith('selection') else 'тест B'} {str(ex['decision_time'])[:10]} · {side} · "
                      f"{'стоп' if ex['reason'] == 'stop' else 'время'} · {pct(gross)} %")
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, vertical_spacing=0.14, horizontal_spacing=0.06)
    for i, ex in enumerate(exs):
        o8_trade_traces(fig, ex, i // 3 + 1, i % 3 + 1, i == 0)
    fig.update_xaxes(rangeslider_visible=False, tickformat="%H:%M")
    fig.update_annotations(font_size=12, font_color=INK2)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    return F(fig, "o8", "o8_examples", "Шесть реальных сделок", 780, lift=24, subtitle="три из подбора (2021-2024), три из теста B (2024-07..2025-07); брутто доходность в заголовке")


def o8_selection():
    S = sorted(o8_sel, key=lambda s: s["t_2bps"])
    fig = go.Figure()
    for fam, col, pat in (("BRK", VIOLET, ""), ("REV", TAN, "/")):
        d = [s for s in S if s["family"] == fam]
        fig.add_trace(go.Bar(y=[s["config"] for s in d], x=[s["t_2bps"] for s in d], orientation="h", name=("пробой (BRK)" if fam == "BRK" else "возврат к якорю (REV)"),
                             marker=dict(color=col, pattern_shape=pat, line=dict(color=SURFACE, width=0.8)),
                             customdata=[[s["mean_m_2bps"] * 100, s["n_trades"], s["months_pos_2bps"], s["n_months"]] for s in d],
                             hovertemplate="%{y}<br>t = %{x:+.2f}; среднее нетто %{customdata[0]:+.2f} %/мес; сделок %{customdata[1]}; месяцев с плюсом %{customdata[2]}/%{customdata[3]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEUT, width=1))
    fig.add_annotation(x=_ch["t_2bps"], y=o8_choice["config"], text=f"выбрано: t = {ru(_ch['t_2bps'], 2, True)}", showarrow=True, ax=70, ay=0, font=dict(size=12, color=INK), arrowcolor=INK2)
    fig.update_yaxes(categoryorder="array", categoryarray=[s["config"] for s in S])
    fig.update_layout(barmode="overlay", xaxis_title="t-статистика среднего месячного нетто (2 б.п.), 42 месяца подбора 2021-01..2024-06", yaxis=dict(autorange=True, dtick=1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    f = F(fig, "o8", "o8_selection", "24 конфигурации на месяцах подбора", 760, subtitle="сверху вниз по убыванию t; рисунок: штриховка - REV, сплошной - BRK; все 12 REV убыточны")
    rows = []
    for s in sorted(o8_sel, key=lambda s: -s["t_2bps"]):
        r = s["reasons"]
        rows.append([(s["config"] + (" ★" if s["config"] == o8_choice["config"] else ""), s["config"]), ("пробой" if s["family"] == "BRK" else "возврат", s["family"]), (ru(s["k"], 1), s["k"]),
                     (str(s["H"]), s["H"]), ("нет" if s["stop"] is None else ru(s["stop"], 1) + "σ̂", s["stop"] or 0), (px(s["n_trades"]), s["n_trades"]),
                     (ru(s["mean_m_0bps"] * 100, 2, True), s["mean_m_0bps"]), (ru(s["mean_m_2bps"] * 100, 2, True), s["mean_m_2bps"]), (ru(s["mean_m_5bps"] * 100, 2, True), s["mean_m_5bps"]),
                     (ru(s["t_2bps"], 2, True), s["t_2bps"]), (f"{s['months_pos_2bps']}/{s['n_months']}", s["months_pos_2bps"]), (ru(s["hit_2bps"] * 100, 1), s["hit_2bps"]),
                     (ru(s["max_dd_2bps"] * 100, 1), s["max_dd_2bps"]), (f"{r.get('target', 0)}/{r.get('stop', 0)}/{r.get('time', 0)}", r.get("target", 0) + r.get("stop", 0) + r.get("time", 0))])
    chosen = [r for r in rows if r[0][1] == o8_choice["config"]][0]
    assert chosen[9][0] == "+3,49" and chosen[7][0] == "+4,04" and chosen[12][0] == "28,2", chosen
    tb = o8_sortable(rows, ["конфигурация", "тип", "k", "H, мин", "стоп", "сделок", "0 б.п., %/мес", "2 б.п., %/мес", "5 б.п., %/мес", "t (2 б.п.)", "месяцев +", "попаданий, %", "макс. просадка, %", "цель/стоп/время"])
    return f, tb


def o8_equity():
    sps = ["B", "C", "A"]
    fig = make_subplots(rows=3, cols=3, shared_xaxes=False, vertical_spacing=0.07, horizontal_spacing=0.07, row_heights=[0.40, 0.28, 0.32],
                        subplot_titles=[dict(o8_spans_title)[s] for s in sps] + [""] * 6)
    bars = pd.read_csv(C.ensure_cut_csv(), usecols=["timestamp", "close"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    C.assert_cut(bars["timestamp"], "O8 buy-and-hold bars")
    for ci, sp in enumerate(sps, start=1):
        tr = pd.read_csv(O8D / f"trades_{sp}.csv.gz", parse_dates=["entry_time", "exit_time"])
        for arm, var, nm, col, dash in O8ARMS:
            d = tr[(tr["arm"] == arm) & (tr["variant"] == var)].sort_values("exit_time")
            net = (d["ret_gross"] - O8_COST).to_numpy()
            eq = np.cumsum(net) * 100
            ddv = (np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq)
            row = O8R[sp, arm, var]
            assert len(d) == row["n_trades"] and abs(eq[-1] - row["total_net"] * 100) < 1e-6 and abs(ddv.max() - row["max_dd"] * 100) < 1e-6, (sp, arm, var)
            show = ci == 1
            fig.add_trace(go.Scatter(x=d["exit_time"], y=eq, mode="lines", line=dict(color=col, width=2.2 if arm == "L1" else 1.5, dash=dash), name=nm, legendgroup=nm, showlegend=show,
                                     hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + nm + ": %{y:+.1f} %<extra></extra>"), row=1, col=ci)
            fig.add_trace(go.Scatter(x=d["exit_time"], y=-ddv, mode="lines", line=dict(color=col, width=2.2 if arm == "L1" else 1.5, dash=dash), name=nm, legendgroup=nm, showlegend=False,
                                     hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + nm + ": просадка %{y:.1f} %<extra></extra>"), row=2, col=ci)
        # buy and hold: only the cut file is read; C is a forward span, so its number comes from the test result
        bh = O8BH[sp]
        if sp in O8_BH_RANGE:
            lo, hi = O8_BH_RANGE[sp]
            b = bars[(bars["timestamp"] >= lo) & (bars["timestamp"] <= hi)]
            lr = np.log(b["close"].to_numpy() / b["close"].iloc[0]) * 100
            day = pd.DataFrame({"t": b["timestamp"].dt.floor("D"), "lr": lr}).groupby("t").last()
            assert abs(day["lr"].iloc[-1] - bh["log_return"] * 100) < 1.0, (sp, day["lr"].iloc[-1], bh["log_return"])
            fig.add_trace(go.Scatter(x=day.index, y=day["lr"], mode="lines", line=dict(color=INK2, width=1.8), name="купить и держать (лог.)", legendgroup="bh", showlegend=ci == 1,
                                     hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f} %<extra>купить и держать</extra>"), row=3, col=ci)
        else:
            fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text", showlegend=False, hoverinfo="skip", textfont=dict(size=13, color=INK2),
                                     text=[f"бары после {C.CUTOFF[:10]} не читались;<br>число из теста: купить и держать<br><b>{pct(bh['log_return'], 1)} %</b> (лог.), макс. просадка по дням <b>{ru(bh['max_dd_daily'] * 100, 1)} %</b>"]), row=3, col=ci)
            fig.update_xaxes(visible=False, range=[0, 1], row=3, col=ci)
            fig.update_yaxes(visible=False, range=[0, 1], row=3, col=ci)
        fig.update_yaxes(title_text="сумма нетто-доходностей позиций, %" if ci == 1 else None, row=1, col=ci)
        fig.update_yaxes(title_text="просадка, п.п." if ci == 1 else None, row=2, col=ci)
        if sp in O8_BH_RANGE:
            fig.update_yaxes(title_text="купить и держать, %" if ci == 1 else None, row=3, col=ci)
        fig.add_hline(y=0, line=dict(color=NEUT, width=1), row=1, col=ci)
    fig.update_annotations(font_size=12, font_color=INK2)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return F(fig, "o8", "o8_equity", "Накопленная доходность и просадка", 1000, lift=24,
             subtitle="2 б.п. за круг; одно плечо, без реинвестирования; пунктир - PESS-заливки; низ - купить и держать (отдельная шкала)")


o8_spans_title = [(s, t) for s, t in O8SPANS]


def o8_monthly_tables():
    out = []
    for sp, ttl in O8SPANS:
        rows_ = []
        r0 = O8R[sp, "L1", "primary"]
        months = r0["months"]
        for arm, var, nm, col, dash in O8ARMS:
            assert O8R[sp, arm, var]["months"] == months
        for i, mo in enumerate(months):
            cells = [mo]
            for arm, var, nm, col, dash in O8ARMS:
                v = O8R[sp, arm, var]["monthly_2bps"][i] * 100
                g = ("&#9650;", T.GOOD) if v > 0 else ("&#9660;", T.CRITICAL)
                cells.append(f'<span style="color:{g[1]}">{g[0]}</span> {ru(v, 1, True)}')
            rows_.append(cells)
        mean = ["<b>среднее</b>"] + [f"<b>{ru(O8R[sp, a, v]['mean_m_2bps'] * 100, 2, True)}</b>" for a, v, *_ in O8ARMS]
        rows_.append(mean)
        pos = ["месяцев +"] + [str(int(sum(x > 0 for x in O8R[sp, a, v]["monthly_2bps"]))) + f"/{len(months)}" for a, v, *_ in O8ARMS]
        rows_.append(pos)
        tb = table(rows_, ["месяц"] + [nm for _, _, nm, _, _ in O8ARMS])
        out.append(f"<div><h4>{ttl}</h4>{tb}</div>")
    # the monthly L1 numbers must equal the committed table text
    for sp in ("B", "C", "A"):
        line = " ".join(f"{m}:{v * 100:+.1f}" for m, v in zip(O8R[sp, "L1", "primary"]["months"], O8R[sp, "L1", "primary"]["monthly_2bps"]))
        assert line in o8_tab, (sp, line)
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px">' + "".join(out) + "</div>"


def o8_span_table():
    rows = []
    for sp, _ in O8SPANS:
        for arm in ("L1", "R0", "TRAIL", "SHUF"):
            for var in ("primary", "PESS"):
                r = O8R[sp, arm, var]
                rs = r["reasons"]
                rows.append([f"<b>{sp}</b>" if (arm, var) == ("L1", "primary") else sp, arm, "основные" if var == "primary" else "PESS", px(r["n_trades"]), ru(r["hit"] * 100, 1),
                             ru(r["mean_trade_bps"], 2, True), ru(r["median_trade_bps"], 2, True), ru(r["mean_m_2bps"] * 100, 2, True), ru(r["mean_m_5bps"] * 100, 2, True), ru(r["t_2bps"], 2, True),
                             ru(r["sharpe_daily"], 2, True), ru(r["max_dd"] * 100, 1), ru(r["calmar"], 2, True), f"{rs.get('stop', 0)}/{rs.get('time', 0)}", badge("bad", "не выполнено")])
    # a few cells tie to REPORT.md
    assert O8R["B", "L1", "primary"]["n_trades"] == 955 and O8R["C", "L1", "primary"]["n_trades"] == 1154 and O8R["A", "L1", "primary"]["n_trades"] == 2015
    assert "B 955 (hit 43.5 %, max DD 11.9 %)" in o8_report and abs(O8R["B", "L1", "primary"]["max_dd"] * 100 - 11.85) < 5e-3
    return table(rows, ["отрезок", "прогноз σ̂", "заливки", "сделок", "попаданий, %", "средняя сделка, б.п.", "медианная, б.п.", "нетто 2 б.п., %/мес", "5 б.п., %/мес", "t", "Шарп (дн.)", "макс. просадка, %", "Кальмар", "стоп/время", "цель владельца"])


def o8_verdict():
    def ci_row(label, key, kind_text):
        c = O8CR[key]
        ok = c["lo"] > 0
        res = badge("good", "выполнен") if ok else badge("bad", "не выполнен")
        if key == "c1_primary":
            res = badge("part", "выполнен, на грани")
        return [label, f"{pct(c['mean'], 3)} % [{pct(c['lo'], 3)}; {pct(c['hi'], 3)}]", f"{c['pos']}/{c['n']}", ru(c["t"], 2, True), res]
    rows = [ci_row("<b>1.</b> L1: средний месячный нетто (2 б.п.), основные заливки", "c1_primary", ""),
            ci_row("<b>1.</b> то же при PESS-заливках (стоп проверяется и в минуте входа)", "c1_PESS", ""),
            ci_row("<b>2.</b> L1 − SHUF (σ̂ перемешана - нулевая гипотеза), парно по месяцам", "c2_L1_minus_SHUF", ""),
            ci_row("<b>3.</b> L1 − R0 (та же стратегия, σ̂ без геометрии), парно по месяцам", "c3_L1_minus_R0", ""),
            ci_row("<span class='note'>справочно: L1 − SHUF при PESS</span>", "c2_L1_minus_SHUF_PESS", ""),
            ci_row("<span class='note'>справочно: L1 − R0 при PESS</span>", "c3_L1_minus_R0_PESS", "")]
    tb = table(rows, ["критерий (B + C вместе, 26 месяцев, 2 б.п.)", "среднее, % в месяц [95% t-интервал]", "месяцев с плюсом", "t", "результат"])
    allrows = list(O8R.values())
    best_hit = max(allrows, key=lambda r: r["hit"])
    best_dd = min(allrows, key=lambda r: r["max_dd"])
    assert not any(r["hit"] > 0.60 and r["max_dd"] < 0.05 for r in allrows) and max(r["hit"] for r in allrows) < 0.60 and min(r["max_dd"] for r in allrows) > 0.05
    tgt = table([["цель владельца: попаданий &gt; 60 %, макс. просадка &lt; 5 %", badge("bad", "не выполнено нигде"),
                  f"лучшее попадание среди 24 строк: {ru(best_hit['hit'] * 100, 1)} % ({best_hit['span']}, {best_hit['arm']}, {best_hit['variant']}); меньшая просадка: {ru(best_dd['max_dd'] * 100, 1)} % ({best_dd['span']}, {best_dd['arm']}, {best_dd['variant']}); "
                  "ни одна строка не выполняет обе части"]], ["", "результат", "числа"])
    v = (f'<div class="verdictline">{badge("bad", "стратегия работает на независимых данных: нет")} &nbsp; {badge("bad", "геометрия добавляет к стратегии: нет")}</div>')
    return v + tb + tgt


def sec_o8():
    parts = []
    fl = ("<ul class='note'>"
          "<li><b>Что прогнозируем.</b> Не знак, а размер: каждые 15 минут (в :00, :15, :30, :45) модель L1 из O5 даёт σ̂ - ожидаемое стандартное отклонение лог-доходности на ближайшие H минут "
          "(из логарифма реализованной волатильности; геометрия индикаторов входит в признаки). Направление не прогнозируется.</li>"
          "<li><b>Якорь и полоса.</b> Якорь m - цена открытия минуты решения. Ставятся два стоп-ордера: покупка выше <b>m(1 + k·σ̂)</b> и продажа ниже <b>m(1 − k·σ̂)</b> (в выбранной настройке k = 1,5). "
          "Они живут 15 минут; сработавший первым побеждает, второй снимается; одна позиция одновременно.</li>"
          "<li><b>Вход и выход.</b> Вход по цене уровня (при гэпе - по открытию, то есть по худшей цене). От цены входа ставится защитный стоп на 1·σ̂ против позиции. "
          "Выход - стоп или тайм-стоп через H = 60 минут, что раньше. Стоимость 2 б.п. за круг (D-044: по умолчанию 0; это выбор владельца для этого исследования); 0 и 5 б.п. тоже посчитаны.</li>"
          "<li><b>Идея.</b> Если σ̂ хорошо оценивает размер колебаний, полоса стоит там, где обычное колебание кончается: выход за неё чаще означает начало движения, которое стоит поймать (BRK). "
          "Вторая идея - возврат к якорю (REV). Какая из двух работает и с какими k, H, стопом - решили 24 конфигурации на месяцах подбора 2021-01..2024-06 (по t-статистике), "
          "до того как был посчитан хоть один тестовый месяц. Тесты: B (2024-07..2025-07-20), C (2025-10-05..2026-10-12, заморожено), A (2018-07..2020-12, отдельно).</li>"
          "<li><b>Две проверки.</b> «Не удача»: та же стратегия с перемешанной σ̂ (SHUF). «Геометрия нужна»: та же стратегия с σ̂ без геометрии (R0). "
          "PESS (добавлено до тестов): стоп проверяется и в минуте входа - худший порядок внутри минуты.</li></ul>")
    parts.append(card("1. Как это работает", fl + o8_schematic(o8_win["examples"][0]),
                      "Свечи - реальные минутные бары из файла цен до 2025-07-20. Серые пунктиры: якорь; золотые: два стоп-ордера (живут 15 минут); красный: защитный стоп; фиолетовый пунктир: тайм-стоп."))
    ex = o8_examples()
    fw = o8_win["forward_numbers_only"]
    assert len(fw) == 2 and all(pd.Timestamp(f["entry_time"]) > pd.Timestamp(C.CUTOFF) for f in fw)
    frows = [[str(f["entry_time"])[:16], str(f["exit_time"])[:16], "покупка" if f["side"] == 1 else "продажа", px(f["entry_px"]), px(f["exit_px"]), "стоп" if f["reason"] == "stop" else "время",
              pct(f["ret_gross"], 2), ru(f["sigma"] * 100, 2)] for f in fw]
    parts.append(card("Шесть реальных сделок", ex + "<h4>Две сделки на прогнозном файле (после 2025-07-20) - только числа, без баров</h4>" +
                      table(frows, ["вход", "выход", "сторона", "цена входа", "цена выхода", "выход по", "брутто, %", "σ̂, %"]),
                      "Три сделки из месяцев подбора и три из теста B. Свечи со сделок прогнозного периода не рисуются: бары после отсечки не читаются."))
    f, tb = o8_selection()
    parts.append(card("2. Подбор: 24 конфигурации на 42 месяцах (2021-01..2024-06)", f + "<h4>Таблица (щёлкните по заголовку, чтобы отсортировать)</h4>" + tb,
                      f"Выбор: максимум t при 2 б.п. - BRK k=1,5, H=60, стоп 1σ̂: {pct(_ch['mean_m_2bps'], 2)} % в месяц, t = {ru(_ch['t_2bps'], 2, True)}, {px(_ch['n_trades'])} сделок. "
                      "Все 12 конфигураций возврата к якорю (REV) убыточны; прибыльны только пробои (BRK). Колонки «0 / 2 / 5 б.п.» - среднее нетто в месяц при этой стоимости круга; «цель/стоп/время» - число выходов по причинам."))
    parts.append(card("3. Независимые тесты: доходность и просадка", o8_equity() + "<h4>Месячные доходности (2 б.п., основные заливки, если не указано иное)</h4>" + o8_monthly_tables(),
                      "Кривая - сумма нетто-доходностей позиций, в процентах капитала при плече 1; просадка - от максимума этой суммы, в порядке сделок. "
                      "Основная кривая L1 растёт на B и A, на C она плоская; при PESS кривая уходит вниз на C и A, а на B остаётся слабо положительной."))
    parts.append(card("Все числа по отрезкам и вариантам заливок", o8_span_table(),
                      "Попадание - доля позиций с положительным нетто. «PESS» - худший порядок внутри минуты входа. TRAIL не приведён к масштабу L1 (только описательно)."))
    parts.append(card("4. Табло вердиктов", o8_verdict(),
                      "Вердикт «стратегия работает» требует критериев 1 (при обоих вариантах заливок) и 2; «геометрия добавляет» - критерия 3. Нижняя граница интервала должна быть выше нуля. Единица - месяц; месяцы не независимы."))
    rd = ("<ul class='note'>"
          "<li><b>Прибыль исчезает при честных заливках.</b> С основными заливками стратегия положительна на B, около нуля на C (самый свежий, полностью замороженный период) и положительна на A. "
          "Если стоп проверяется и внутри минуты входа (PESS), среднее по B + C отрицательно (−1,06 % в месяц), на C и A убыток, на B +1,18 % с интервалом, включающим ноль. Подбор заработал в основном на том, что игнорировал разворот внутри минуты пробоя: это артефакт модели заливок, а не преимущество.</li>"
          "<li><b>Прогноз размера не движет результатом.</b> L1 − R0 по B + C: −0,38 % [−1,41; +0,66] (R0 на B даже лучше). SHUF (σ̂ перемешана, время потеряно) на B заметно лучше L1 (+7,36 против +3,68 % в месяц, при 2,6 раза большем числе сделок); по B + C L1 − SHUF = −1,24 % [−3,40; +0,92]. "
          "Доход - от импульса после пробоя полосы в трендовые периоды (B, A), а не от того, КОГДА прогноз говорит, что движения будут большими.</li>"
          "<li><b>В согласии со всем проектом.</b> Размер движений прогнозируем (O1, O5, H53, H83), но на споте одного размера мало: нужно направление, а минутное направление выдохлось (H51, H52). "
          "T02 (экспозиция по волатильности) и H78/H79 (направление, масштабированное волатильностью) закончились так же.</li></ul>")
    cav = ("<ul class='note'><li>PESS - граница (худший порядок внутри минуты), а не модель заливок: истина между основными и PESS. Стоп-маркет-заливки без проскальзывания сверх 2 б.п. - оптимистично для пробоев.</li>"
           "<li>Месяцы не независимы; три месяца неполные: 2025-07 (до 20-го), 2025-10 (с 5-го), 2026-10 (12 дней). Месяцы C читали ранние тактические исследования; цены на них не использовались для подбора стратегии, 12 дней 2026-10 не читал никто.</li>"
           "<li>SHUF - не чистая нулевая гипотеза: перемешивание меняет число сделок. Гэп-страж: решения, у которых следующие 75 баров пересекают 19-часовой разрыв файла (до 2025-03-15), ордеров не ставят.</li></ul>")
    parts.append(card("5. Чтение руководителя", rd + "<h4>Оговорки</h4>" + cav, "Из REPORT.md O8."))
    return "".join(parts)


def sec_pending():
    body = ("<ul class='note'>"
            "<li><b>O4, часть B:</b> score_B.py (O4_bank_confirm) оценивает BANK−R0 на первых 3 полных месяцах с 2026-10 (2026-10..12), критерий A при H=60, 3 из 3 месяцев с плюсом. "
            f"Артефакт заморожен: sha256 <code>{sha4[:16]}...</code>.</li>"
            "<li><b>O5, часть B:</b> score_B.py (O5_geom_bestbase) оценивает L1−L0 на тех же месяцах, один раз. "
            f"sha256 <code>{sha5[:16]}...</code>.</li>"
            f"<li><b>O6, часть B:</b> артефакт заморожен (sha256 <code>{sha6[:16]}...</code>), score_B.py не запускался; после провала на H=240 он почти не нужен.</li>"
            "<li><b>O7, часть B:</b> score_B.py сравнит замороженную L1 с L0: калибровка на 2026-10, оценка 2026-11..2027-01, один раз; единственная непотраченная проверка интервалов.</li>"
            "<li><b>O2:</b> результат есть (секция O2); для более решающего ответа нужно больше срезов или зёрен.</li>"
            "<li>Пересборка: <code>python runs/tactical/report/build_outlook_report.py</code></li></ul>")
    return card("Что ещё не готово", body, "До появления полных месяцев 2026-10.. часть B не оценивается и не обсуждается.")


# ======================================================================================================== assemble
def get_css():
    src = (HERE / "build_report.py").read_text(encoding="utf-8")
    m = re.search(r'CSS = """(.*?)""" % dict', src, re.S)
    css = m.group(1) % dict(PAPER=PAPER, SURFACE=SURFACE, INK=INK, INK2=INK2, MUTED=MUTED, GRID=GRID, AXIS=AXIS, PINK=PINK, GOLD=GOLD, FONT=T.FONT)
    return css + ".lead{max-width:1500px}ul.note{padding-left:20px}td ul{margin:0}.verdictline{margin:6px 0 14px;font-size:15px}table.sortable th{cursor:pointer}"


def build():
    d = load_bars()
    assert d["timestamp"].max() <= pd.Timestamp(C.CUTOFF)
    secs = [("o0", "Обзор", sec1), ("o1", "O1. Геометрия", sec_o1), ("o2", "O2. Физика", sec_o2), ("o3", "O3. Синтетические индикаторы", sec_o3),
            ("o4", "O4. Подтверждение BANK", sec_o4), ("o5", "O5. Геометрия на лучшей базе", sec_o5),
            ("o6", "O6. Часовая геометрия", sec_o6), ("o7", "O7. Геометрия как интервалы", sec_o7), ("o8", "O8. Стратегия: пробой полосы", sec_o8), ("pend", "Что ещё не готово", sec_pending)]
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
    parts.append(f"<footer>Сгенерировано runs/tactical/report/build_outlook_report.py (коммит основы {sha}); данные: runs/tactical/outlook/O1..O8 (results, tables, interp, kernels, windows, summary, REPORT). "
                 f"Цены - только до {C.CUTOFF}; числа прогнозного года вперёд (2025-10..2026-09) - результаты, не бары. Исследовательская работа, одно зерно, не влияет на умолчания MVP (D-062).</footer></main>")
    parts.append("""<script>(function(){const q=new URLSearchParams(location.search).get('only');
if(q){document.body.classList.add('only');document.querySelectorAll('main > section').forEach(s=>{if(s.id!==q)s.style.display='none'});}
addEventListener('load',()=>setTimeout(()=>window.dispatchEvent(new Event('resize')),500));
const links=[...document.querySelectorAll('nav a')];
function onscroll(){let cur=null;document.querySelectorAll('main > section').forEach(s=>{if(s.getBoundingClientRect().top<140)cur=s.id});links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+cur))}
addEventListener('scroll',onscroll);
document.querySelectorAll('table.sortable').forEach(t=>{t.querySelectorAll('th').forEach((th,i)=>{th.title='сортировать';th.addEventListener('click',()=>{const tb=t.tBodies[0];const rows=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';rows.sort((a,b)=>{const x=a.cells[i].dataset.v,y=b.cells[i].dataset.v;const nx=parseFloat(x),ny=parseFloat(y);const c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;return asc?c:-c});rows.forEach(r=>tb.appendChild(r))})})});})();</script></body></html>""")
    out = HERE / "outlook_report.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} {out.stat().st_size / 1e6:.1f} MB, {len(FIGS)} figures, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    build()
