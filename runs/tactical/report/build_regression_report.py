"""Standalone HTML report 'Что находит регрессия' = T09 (tasks/T09_regression_anatomy) + literature research H91.

Reuses the visual system of build_report.py (the CSS is read out of that file, not imported) and rcommon.py / theme.py.
CPU only. Nothing is recomputed: the T09 figures are the committed plotly JSON (re-labelled in Russian), the tables are parsed
from tables.txt, the text numbers are copied from REPORT.md / H91. Two extra figures are drawn here from small data:
the lag scan (regdata/lag_scan.csv, the H91 scratch) and a real bounce window (the cut file, 2018).
usage: python build_regression_report.py   ->   regression_report.html
"""
import html as H
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as po
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcommon import (T, C, TAC, UP, DOWN, PINK, GOLD, TAN, VIOLET, SALMON, INK, INK2, MUTED, GRID, SURFACE, PAPER, AXIS,  # noqa: E402
                     table, card, ru)

HERE = Path(__file__).resolve().parent
T09 = TAC / "tasks" / "T09_regression_anatomy"
H91 = TAC / "research" / "2026-10-13-regression-signal" / "README.md"
FIGS = []


# ======================================================================================================== CSS (from build_report.py)
def css():
    src = (HERE / "build_report.py").read_text(encoding="utf-8")
    a = src.index('CSS = """') + len('CSS = """')
    b = src.index('""" % dict(PAPER')
    base = src[a:b] % dict(PAPER=PAPER, SURFACE=SURFACE, INK=INK, INK2=INK2, MUTED=MUTED, GRID=GRID, AXIS=AXIS, PINK=PINK, GOLD=GOLD, FONT=T.FONT)
    extra = """
.tenl{margin:8px 0 14px;padding-left:22px;max-width:1300px;color:var(--ink)}.tenl li{margin:7px 0}.tenl li b{color:var(--ink)}
details{margin:6px 0 12px}summary{cursor:pointer;color:var(--ink);font-size:14px;padding:4px 0}
ul.refs{font-size:12.5px;padding-left:20px;max-width:1300px}ul.refs li{margin:3px 0}ul.refs a{color:var(--gold);word-break:break-all}
.para{max-width:1300px;margin:6px 0 12px}.para p{margin:6px 0}
.gloss{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin:8px 0}.gloss div{background:#1f1f1e;border:1px solid var(--axis);border-radius:6px;padding:8px 12px;font-size:13px}
.gloss b{color:var(--ink)}table.rt td.num{text-align:right}
"""
    return base + extra


# ======================================================================================================== tables.txt parsing
TXT = (T09 / "tables.txt").read_text(encoding="utf-8").splitlines()


def block(title):
    """Content lines (blank lines dropped) of the first block whose title line starts with `title`."""
    for i, ln in enumerate(TXT):
        if ln.startswith(title) and i + 1 < len(TXT) and TXT[i + 1].startswith("===="):
            out = []
            for x in TXT[i + 2:]:
                if x.startswith("===="):
                    break
                out.append(x)
            return [x for x in out if x.strip()]
    raise KeyError(title)


def pm(s, nd=3):
    """'0.5681±0.0207' -> '0,568 ± 0,021' (value rounded to nd; the half-width to nd)."""
    if "±" in s:
        a, b = s.split("±")
        return f"{ru(float(a), nd)} ± {ru(float(b), nd)}"
    return ru(float(s), nd)


def fixed_rows(lines, years=True):
    rows = []
    for ln in lines:
        p = ln.split()
        rows.append(p)
    return rows


YEARS = [str(y) for y in range(2017, 2026)]


def grid_table(title, labels, cell=lambda s: pm(s, 3), hdr=YEARS, first="вход"):
    """A 'label  v1 v2 ...' block as an HTML table (rows kept in file order, only the labels given)."""
    ls = block(title)
    rows = []
    for ln in ls[1:]:
        p = ln.split()
        if p[0] in labels:
            rows.append([labels[p[0]]] + [cell(x) for x in p[1:]])
    return table(rows, [first] + hdr)


# ======================================================================================================== figures
FTR = {  # exact-string translations of annotations, trace names and axis titles inside the T09 figures
    "close entry (H51)": "вход по закрытию (H51)", "next-open entry (H59)": "вход по следующему открытию (H59)",
    "close entry": "вход по закрытию", "next-open entry": "вход по следующему открытию", "mid of next bar": "середина следующего бара",
    "regression AUC": "AUC регрессии", "best simple fade rule (k chosen in the year)": "лучшее простое правило «против последних k» (k выбран в году)",
    "target horizon": "горизонт цели", "fade the last k hours": "против последних k часов", "fade the last k min": "против последних k мин", "bps": "б.п.",
    "mid of the next bar": "середина следующего бара", "Roll spread (bounce)": "спред Ролла (отскок)",
    "AUC of a model using ONLY the family": "AUC модели ТОЛЬКО на этом семействе", "Family permutation AUC loss": "Потеря AUC при перемешивании семейства",
    "Hourly rich logistic (H62): AUC by year, 4-hour horizon": "Часовая логистика (H62): AUC по годам, горизонт 4 часа",
    "Impulse AUC (next-open), 2019": "Импульсный отклик AUC (след. открытие), 2019",
    "AUC by high-low range tercile and weekend": "AUC по терцилю размаха high-low и по выходным",
    "AUC of a model using ONLY the group": "AUC модели ТОЛЬКО на этой группе",
    "Impulse AUC, next-open entry (forward months)": "Импульсный отклик AUC, вход по след. открытию (месяцы вперёд)",
    "Monthly AUC, forward year 2025-11..2026-09": "AUC по месяцам, год вперёд 2025-11..2026-09",
    "Permutation AUC loss by feature (close entry, mean over months)": "Потеря AUC при перемешивании признака (вход по закрытию, среднее по месяцам)",
    "Top-10% gross bps per trade, by month": "Валовые б.п. на сделку, топ-10%, по месяцам",
    "Close entry (H51 label): standardised coefficient of the monthly logistic regression, feature x month": "Вход по закрытию (метка H51): стандартизованный коэффициент месячной логистики, признак x месяц",
    "Next-open entry (H59 label)": "Вход по следующему открытию (метка H59)",
    "AUC of a model using ONLY the group (full model: row 'all 13')": "AUC модели ТОЛЬКО на группе (полная модель: строка «all 13»)",
    "Drop-one AUC loss (refit without the feature)": "Потеря AUC при удалении признака (переобучение без него)",
    "Grouped permutation loss, by group": "Потеря при перемешивании группы",
    "Next-open entry: single-feature permutation loss": "Вход по след. открытию: перемешивание одного признака",
    "Single-feature permutation AUC loss (fitted model)": "Потеря AUC при перемешивании одного признака (обученная модель)",
    "Univariate AUC - 0.5 of the raw feature (negative = reversal)": "AUC одного признака − 0,5 (отрицательно = разворот)",
    "|feature| decile (10 = most stretched)": "децил |признака| (10 = самое растянутое)",
    "share of moves in the 'fade' direction": "доля ходов в направлении «против признака»",
    "1-min high-low range tercile (spread proxy)": "терциль размаха high-low за минуту (прокси спреда)",
    "Corwin-Schultz spread (zero | low | high)": "спред Корвина-Шульца (ноль | низкий | высокий)",
    "hour of day, UTC (file hours converted: minus 4 h in US daylight saving, minus 5 h otherwise, until the 2024-09-13 splice)":
        "час суток, UTC (часы файла пересчитаны: минус 4 ч летом США, минус 5 ч иначе, до склейки 2024-09-13)",
    "size of the last 1-min move (bps): quantile bins": "размер последнего минутного хода (б.п.): квантильные корзины",
    "size of the last 1-min move (in window sigmas): quantile bins": "размер последнего минутного хода (в сигмах окна): квантильные корзины",
    "volatility (window sigma) tercile": "терциль волатильности (сигма окна)", "volume tercile": "терциль объёма", "weekday vs weekend": "будни и выходные",
    "1,160-min gap": "пропуск 1160 мин", "file splice": "склейка файла", "tick $0.01 to $1": "тик $0,01 → $1",
    "Decay vs 2019, close entry: where the AUC went": "Падение относительно 2019, вход по закрытию: куда ушёл AUC",
    "Decay vs 2019, next-open entry": "Падение относительно 2019, вход по следующему открытию",
    "Market quality, monthly: autocorrelation and variance ratio": "Качество рынка по месяцам: автокорреляция и коэффициент дисперсий",
    "Monthly AUC against the lag-1 return autocorrelation (close entry; colour = period)": "Месячный AUC против автокорреляции доходности на лаге 1 (вход по закрытию; цвет = период)",
    "Monthly AUC of the regression, by entry": "Месячный AUC регрессии по способу входа",
    "Spread and bounce proxies, monthly (bps)": "Прокси спреда и отскока по месяцам (б.п.)",
    "Corwin-Schultz median": "Корвин-Шульц, медиана", "Roll spread": "спред Ролла", "coefficient adaptation": "подстройка коэффициентов",
    "condition mix (volatility x volume cells)": "состав условий (клетки волатильность x объём)",
    "feature strength (same coefficients, same condition mix)": "сила признаков (те же коэффициенты, тот же состав)",
    "lag-1 autocorrelation": "автокорреляция на лаге 1", "median 1-min high-low range": "медианный размах high-low за минуту",
    "total change vs 2019": "полное изменение против 2019", "variance ratio VR(1,15) - 1": "коэффициент дисперсий VR(1,15) − 1",
    "lag-1 autocorrelation of 1-minute returns": "автокорреляция минутной доходности на лаге 1", "monthly AUC": "месячный AUC", "AUC change vs 2019": "изменение AUC против 2019",
    "Gross bps per trade of the top-10% signals = break-even round-trip cost": "Валовые б.п. на сделку топ-10% сигналов = безубыточная стоимость круга",
    "What it costs: spread proxies of the traded bars (bps)": "Во что это обходится: прокси спреда торгуемых баров (б.п.)",
    "Corwin-Schultz of the traded bars": "Корвин-Шульц торгуемых баров", "high-low range of the signal bars (traded)": "размах high-low баров-сигналов (торгуемых)",
    "high-low range, all bars": "размах high-low, все бары",
    "Bitstamp BTC/USD 1-minute close, the regression's top-10% signals as trades (enter next open, hold 15 minutes)":
        "Минутное закрытие Bitstamp BTC/USD; топ-10% сигналов регрессии как сделки (вход по следующему открытию, удержание 15 минут)",
    "P(up) - 0.5 (dashed: the top-10% threshold of the month)": "P(рост) − 0,5 (пунктир: порог топ-10% месяца)",
    "What it reads: bb20 (position in the 20-minute band) and the 30-minute return in sigmas": "Что она читает: bb20 (положение в 20-минутной полосе) и 30-минутная доходность в сигмах",
    "triangle = entry, circle = winning exit, x = losing exit; purple line = long, amber = short": "треугольник = вход, кружок = выигрышный выход, x = проигрышный; фиолетовая линия = лонг, янтарная = шорт",
    "30-minute return in sigmas / 10": "30-минутная доходность в сигмах / 10", "P(up) - 0.5": "P(рост) − 0,5",
    "bb20 (band position; > 0.5 = above the band)": "bb20 (положение в полосе; > 0,5 = выше полосы)", "close": "закрытие",
    "long (regression says up)": "лонг (регрессия: вверх)", "short (regression says down)": "шорт (регрессия: вниз)",
    "AUC by high-low range tercile (spread proxy), close vs next-open entry": "AUC по терцилю размаха high-low (прокси спреда): вход по закрытию и по след. открытию",
    "Close-entry minus next-open AUC against the Roll spread, monthly": "AUC (закрытие) − AUC (след. открытие) против спреда Ролла, по месяцам",
    "Regression AUC by entry convention, by year": "AUC регрессии по способу входа, по годам", "Roll spread (bps)": "спред Ролла (б.п.)",
    "AUC(close) - AUC(next open)": "AUC(закрытие) − AUC(след. открытие)", "Signal bars are the wide-range bars (median high-low range, bps)": "Бары-сигналы — бары с широким размахом (медиана high-low, б.п.)",
    "all bars": "все бары", "traded (top-10%) signal bars": "торгуемые бары-сигналы (топ-10%)",
    "close-entry model scored on next-open labels": "модель входа по закрытию на метках следующего открытия",
    "empirical P(up) (20 quantile bins, 95% band)": "эмпирическая P(рост) (20 квантильных корзин, 95% полоса)",
    "model partial dependence (conditional)": "частная зависимость модели (условная)",
}
FTR_RE = [(re.compile(r"^R2 line ([\d.]+), slope lo/hi (.+)$"), lambda m: f"R² линии {m.group(1).replace('.', ',')}, наклон низ/верх {m.group(2).replace('.', ',')}"),
          (re.compile(r"^Spearman (-?[\d.]+); detrended Pearson (-?[\d.]+) \(n = (\d+)\)$"),
           lambda m: f"Спирмен {m.group(1).replace('.', ',')}; Пирсон без тренда {m.group(2).replace('.', ',')} (n = {m.group(3)})")]


def tr(s):
    if s is None:
        return s
    if s in FTR:
        return FTR[s]
    for rx, f in FTR_RE:
        m = rx.match(s)
        if m:
            return f(m)
    return s


def wrap_sub(sub, width=215):
    """Subtitle as <br>-separated lines of at most `width` characters (plotly does not wrap titles)."""
    import textwrap
    text = re.sub(r"\s*<br>\s*", " ", sub)
    return textwrap.wrap(text, width=width, break_long_words=False)


def rus_fig(name, title, sub, key, sec):
    """Load the committed T09 plotly JSON, translate labels, replace the title (the finding) and subtitle."""
    fig = pio.from_json((T09 / "figs" / f"{name}.json").read_text(encoding="utf-8"))
    for a in fig.layout.annotations or ():
        a.text = tr(a.text)
    for t in fig.data:
        if getattr(t, "name", None):
            t.name = tr(t.name)
    for k in list(fig.layout):
        if k[1:5] == "axis" and fig.layout[k].title is not None and fig.layout[k].title.text:
            fig.layout[k].title.text = tr(fig.layout[k].title.text)
    n_old = fig.layout.title.text.count("<br>")
    lines = wrap_sub(sub)
    mt = fig.layout.margin.t
    fig.update_layout(title_text=f"<b>{title}</b><br><span style='font-size:12px;color:{MUTED}'>{'<br>'.join(lines)}</span>")
    if mt:                                                     # T09's own rule: t = 62 + 17 lines + 84, the legend 36 px above the first subplot title
        h = fig.layout.height or 900
        bm = fig.layout.margin.b or 56
        t = 62 + 17 * len(lines) + 84
        fig.update_layout(margin=dict(t=t), legend=dict(orientation="h", yanchor="bottom", y=1 + 36 / max(h - t - bm, 200), xanchor="left", x=0.0))
    FIGS.append((sec, key, title))
    return fig


def html_fig(fig, key):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"f_{key}",
                       config={"displaylogo": False, "responsive": True, "scrollZoom": False})


SRC_NOTE = " Данные: Bitstamp BTC/USD, минутная история (cut-файл; не Binance)."


# ======================================================================================================== extra figures
def own_title(fig, title, sub, height, extra_t=0):
    """extra_t > 0 for figures with subplot titles: the legend is lifted above the first one."""
    lines = wrap_sub(sub)
    T.apply(fig, title=title, subtitle="<br>".join(lines), height=height)
    t = 62 + 17 * len(lines) + 50 + extra_t
    fig.update_layout(margin=dict(t=t))
    if extra_t:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1 + 34 / max(height * 0.62, 200), xanchor="left", x=0.0))


def fig_lag_scan():
    d = pd.read_csv(HERE / "regdata" / "lag_scan.csv", parse_dates=["date"])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.34, 0.26, 0.40],
                        subplot_titles=("Лучший сдвиг файла относительно Binance, минут (240 = UTC+4, 300 = UTC+5, 0 = UTC)",
                                        "Корреляция минутных доходностей Binance и файла при лучшем сдвиге",
                                        "Автокорреляция минутной доходности на лаге 1: Binance и файл (окно в 1000 минут раз в месяц)"))
    fig.add_trace(go.Scatter(x=d.date, y=d.lag_min, mode="lines+markers", line=dict(color=GOLD, width=2, shape="hv"), marker=dict(size=6, symbol="square"),
                             name="лучший сдвиг", hovertemplate="%{x|%Y-%m}: %{y} мин<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=d.date, y=d["corr"], marker=dict(color=TAN, line=dict(color=SURFACE, width=0.5)), name="корреляция",
                         hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=d.date, y=d.ac1_binance, mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=5), name="Binance",
                             hovertemplate="Binance %{x|%Y-%m}: %{y:.3f}<extra></extra>"), 3, 1)
    fig.add_trace(go.Scatter(x=d.date, y=d.ac1_file, mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=5, symbol="diamond"), name="файл (Bitstamp)",
                             hovertemplate="файл %{x|%Y-%m}: %{y:.3f}<extra></extra>"), 3, 1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=3, col=1)
    for x, lab in (("2022-08-10", "тик $0,01 → $1"), ("2024-09-13", "склейка файла"), ("2025-03-15", "пропуск 1160 мин")):
        for r in (1, 2, 3):
            fig.add_vline(x=pd.Timestamp(x).timestamp() * 1000, line=dict(color=MUTED, width=1, dash="dashdot"), row=r, col=1)
        fig.add_annotation(x=pd.Timestamp(x), y=1.0, yref="y domain", text=lab, showarrow=False, textangle=-90, xanchor="right", yanchor="top",
                           font=dict(color=MUTED, size=10), row=1, col=1)
    fig.update_yaxes(title_text="минут", row=1, col=1, tickvals=[0, 240, 300])
    fig.update_yaxes(title_text="r", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="AC(1)", row=3, col=1)
    fig.update_layout(height=820, showlegend=True)
    own_title(fig, "Откуда файл: сдвиг часов, который исчезает в 2024-09, и отскок, которого на Binance заметно меньше",
              "Для 96 дат (15-е число месяца, 1000 минут) подобран сдвиг, при котором минутные доходности Binance BTCUSDT лучше всего совпадают с файлом.<br>"
                     "Весной-летом сдвиг +4 ч, осенью-зимой +5 ч, после склейки 2024-09-13 ноль: файл — Bitstamp BTC/USD в местном времени.<br>"
                     "Средняя за год AC(1): файл −0,11…−0,17 в 2018-20 против −0,02…−0,05 у Binance на тех же минутах; с 2021 обе около нуля (оценки по 12 окнам в год, ±0,03).", 820, extra_t=30)
    FIGS.append(("s1", "lag_scan", "Сдвиг часов файла и автокорреляция: Binance против файла"))
    return fig


def fig_seams():
    d = pd.read_csv("D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv", usecols=["timestamp", "close"], parse_dates=["timestamp"]).set_index("timestamp")
    C.assert_cut(d.index, "seam figure")
    r = (np.log(d["close"]).diff() * 1e4)
    rows = []
    for y in range(2017, 2025):
        m = pd.Timestamp(y, 3, 1)
        sun2 = m + pd.Timedelta(days=(6 - m.weekday()) % 7 + 7)
        t = sun2 + pd.Timedelta(hours=8)
        day = r[sun2:sun2 + pd.Timedelta(days=1)].abs()
        rows.append((f"{y}-{t.month:02d}-{t.day:02d} 08:00", float(r[t]), int((day > abs(r[t])).sum()) + 1, float(r[str(y)].abs().quantile(0.99))))
    t = pd.Timestamp("2024-09-13 08:01")
    rows.append(("2024-09-13 08:01 (склейка)", float(r[t]), None, float(r["2024"].abs().quantile(0.99))))
    fig = go.Figure()
    xs = [x[0] for x in rows]
    vals = [abs(x[1]) for x in rows]
    cols = [SALMON if (x[2] == 1 or x[2] is None) else TAN for x in rows]
    fig.add_trace(go.Bar(x=xs, y=vals, marker=dict(color=cols, line=dict(color=SURFACE, width=1)), name="|минутная доходность| в минуте шва",
                         text=[f"{v:.0f}" for v in vals], textposition="outside", textfont=dict(color=INK2, size=11),
                         customdata=[[x[2] if x[2] else "-", x[3]] for x in rows],
                         hovertemplate="%{x}<br>|r| = %{y:.1f} б.п.<br>ранг минуты в тот день: %{customdata[0]}<br>99-й процентиль |r| года: %{customdata[1]:.1f} б.п.<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[x[3] for x in rows], mode="lines+markers", line=dict(color=GOLD, width=2, dash="dot"), marker=dict(size=7, symbol="diamond"),
                             name="99-й процентиль |r| того же года"))
    fig.update_xaxes(type="category")
    fig.update_layout(yaxis_title="б.п.", xaxis_title="минута 08:00 по времени файла в воскресенье перехода на летнее время США (весна), и минута склейки 2024",
                      height=470, margin=dict(b=90))
    own_title(fig, "Швы данных: на 3 из 8 весенних переходов минута 08:00 — самая большая за день, на склейке 2024-09 скачок 51 б.п.",
              "Минутная доходность закрытия в 08:00 файла в день перехода. Лососевые столбцы — минута была самым большим минутным ходом этого дня (2018: 109 б.п., 2021: 62, 2024: 48);<br>"
                     "тан — нет. Правило выбора минуты — моё и фиксированное (08:00 весной); осенние швы лежат в разные минуты и здесь не показаны (см. таблицу).", 470, extra_t=0)
    FIGS.append(("s1", "seams", "Скачки доходности на швах данных"))
    return fig, rows


def bounce_window():
    """Rule: in 2018, among bars of the 10% least-traded hours (excluding the two DST-change days and zero-volume bars) whose own close-to-close move was at least 5 bps,
    the bar with the largest REVERSAL at the next open, rev = -sign(r_t) * (open_{t+1} / close_t - 1). Returns the +-20-minute window around it."""
    d = pd.read_csv("D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"]).set_index("timestamp")
    C.assert_cut(d.index, "bounce window")
    y = d["2018-01-01":"2018-12-31"].copy()
    y["r"] = np.log(y.close).diff() * 1e4
    y["gap"] = (y.open.shift(-1) / y.close - 1) * 1e4
    y["rev"] = -np.sign(y.r) * y.gap
    hv = y.volume.resample("h").sum()
    low = hv[hv <= hv.quantile(0.10)].index                      # the 10% least-traded hours of 2018
    dst_days = [pd.Timestamp("2018-03-11"), pd.Timestamp("2018-11-04")]
    ok = y.index.floor("h").isin(low) & ~y.index.normalize().isin(dst_days) & (y.volume > 0) & (y.r.abs() >= 5)
    t0 = y[ok].rev.idxmax()
    w = d[t0 - pd.Timedelta(minutes=20):t0 + pd.Timedelta(minutes=20)].copy()
    w["r"] = np.log(w.close).diff() * 1e4
    w["gap"] = (w.open.shift(-1) / w.close - 1) * 1e4
    return t0, w, float(hv.quantile(0.10)), len(low), float(y.loc[t0, "r"])


def fig_bounce(t0, w, hvq, nlow, r0):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.68, 0.32],
                        subplot_titles=("Минутные свечи; кружок = закрытие бара t, ромб = открытие бара t+1 (отрезок между ними = «зазор»)", "Зазор от закрытия t до открытия t+1, б.п."))
    fig.add_trace(go.Candlestick(x=w.index, open=w.open, high=w.high, low=w.low, close=w.close, name="1-минутные свечи",
                                 increasing=dict(line=dict(color=UP), fillcolor=UP), decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN), whiskerwidth=0.4), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=w.close, mode="markers", marker=dict(size=8, color=INK, symbol="circle", line=dict(color=PAPER, width=1)), name="закрытие бара t",
                             hovertemplate="%{x|%H:%M}: закрытие %{y:,.2f}<extra></extra>"), 1, 1)
    nxt = w.index + pd.Timedelta(minutes=1)
    fig.add_trace(go.Scatter(x=nxt[:-1], y=w.open.shift(-1)[:-1], mode="markers", marker=dict(size=9, color=GOLD, symbol="diamond", line=dict(color=PAPER, width=1)),
                             name="открытие бара t+1", hovertemplate="%{x|%H:%M}: открытие %{y:,.2f}<extra></extra>"), 1, 1)
    for i in range(len(w) - 1):                                                          # the gap as a short segment from close_t to open_{t+1}
        fig.add_trace(go.Scatter(x=[w.index[i], nxt[i]], y=[w.close.iloc[i], w.open.iloc[i + 1]], mode="lines",
                                 line=dict(color=SALMON if w.gap.iloc[i] < 0 else TAN, width=2), showlegend=False, hoverinfo="skip"), 1, 1)
    g = w.gap.iloc[:-1]
    fig.add_trace(go.Bar(x=w.index[:-1], y=g, marker=dict(color=[SALMON if v < 0 else TAN for v in g], line=dict(color=SURFACE, width=0.5)), name="зазор, б.п.",
                         hovertemplate="%{x|%H:%M}: %{y:+.1f} б.п.<extra></extra>"), 2, 1)
    for xa, ya in (("x", "y domain"), ("x2", "y2 domain")):
        fig.add_shape(type="line", x0=t0, x1=t0, y0=0, y1=1, xref=xa, yref=ya, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_xaxes(rangeslider_visible=False, type="date")
    fig.update_yaxes(title_text="цена, USD", row=1, col=1)
    fig.update_yaxes(title_text="б.п.", row=2, col=1)
    fig.update_layout(height=720)
    gv = g.values
    alt = float(np.mean(np.sign(gv[1:]) != np.sign(gv[:-1])))
    cc = float(w[["r", "gap"]].dropna().corr().iloc[0, 1])
    own_title(fig, f"Отскок bid-ask на настоящих барах Bitstamp: {t0:%Y-%m-%d %H:%M} по времени файла, объём бара {ru(w.volume[t0], 2)} BTC",
              f"Окно ±20 минут вокруг бара 2018 года, у которого следующее открытие сильнее всего развернуло ход самого бара (бар сдвинулся на {ru(r0, 1, True)} б.п., следующее открытие вернуло {ru(w.gap[t0], 1, True)} б.п.). "
              f"В этом окне зазор имеет противоположный знак предыдущей минутной доходности (корреляция {ru(cc, 2, True)} по {len(g) - 1} барам) и меняет знак на {alt * 100:.0f}% переходов; "
              f"типичная |минутная доходность| в окне {ru(w.r.abs().median(), 1)} б.п. Закрытие напечатано то на одной стороне книги, то на другой, следующее открытие возвращает цену. "
              f"Это и забирает вход «по закрытию»; вход по следующему открытию его не получает. Одно окно — иллюстрация, а не оценка.", 720, extra_t=30)
    FIGS.append(("s5", "bounce_window", "Реальное окно отскока bid-ask"))
    return fig


# ======================================================================================================== section builders
def para(*ps):
    return '<div class="para">' + "".join(f"<p>{p}</p>" for p in ps) + "</div>"


def ten_lines():
    items = [
        "<b>Что это.</b> Регрессия играет против цены, растянутой за последние 20-60 минут: продаёт после пробега выше 20-минутной полосы и покупает после падения ниже неё.",
        "<b>Один фактор.</b> Все признаки положения цены, недавние доходности и RSI говорят одно и то же. Любая одна группа даёт почти весь AUC, признаки волатильности не добавляют ничего.",
        "<b>Веса нестабильны, результат стабилен.</b> Веса прыгают между коллинеарными признаками от месяца к месяцу (косинус к весам 2019 падает до 0,01 к 2025). Замороженная модель 2019 даёт тот же AUC; переобучение добавляет не больше 0,008.",
        "<b>Две части.</b> (1) Отскок между bid и ask или «залежавшееся» закрытие на тонком рынке: 0,02-0,04 AUC от преимущества входа по закрытию в 2017-20, с 2021 его нет. (2) Настоящий разворот на 15-60 минутах — он и переживает вход по следующему открытию.",
        "<b>Живёт в хвостах.</b> Самый растянутый дециль выигрывает в 56-61% случаев (2017-19) и в 53-55% (2024-25). Средние децили — монета.",
        "<b>Простое правило делает ту же работу.</b> «Играй против последних 30-60 минут» воспроизводит 90-100% преимущества регрессии при входе по следующему открытию в каждый год. С 2022 оно не хуже регрессии.",
        "<b>Почему затухло.</b> Ослаб сам разворот: те же веса и те же условия теперь предсказывают меньше (−0,044…−0,075 из падения −0,045…−0,078). Он следует за рынком: автокорреляция на лаге 1 была −0,09/−0,11 (2017-20), стала около 0 (с 2023); VR(1,15) вырос с 0,70-0,81 до 0,95-0,99; спред Ролла упал с 5,6-10 до 0,1-1,6 б.п.",
        "<b>Экономика.</b> Валовое преимущество топ-10% при входе по следующему открытию: 1,7-3,6 б.п. в 2018-21. С 2022 — 0-1 б.п., внутри собственного шума и ниже любых реальных издержек.",
        "<b>Сегодня на Binance, 2025-11..2026-09:</b> AUC 0,535-0,538, отскока нет. Валовое преимущество 0,2-0,7 б.п. на сделку топ-10%, от нуля не отличить.",
        "<b>Часовой двойник (H62) — тоже разворот</b> (RSI, VWAP и расстояние до EMA на 1-24 часах). Он затух меньше: 0,58-0,59 в 2019-20, 0,54 в 2024-25.",
    ]
    return '<ol class="tenl">' + "".join(f"<li>{x}</li>" for x in items) + "</ol>"


def sec0():
    tiles = [
        ("0,599 → 0,521", "AUC направления, вход по закрытию", "годовое среднее 2019 → 2025 (±0,01); при входе по следующему открытию 0,561 → 0,518"),
        ("0,02-0,04", "AUC — доля отскока bid-ask в 2017-20", "в преимуществе входа по закрытию; с 2021 нуль. Это около трети преимущества 2018-20 (H91)"),
        ("90-100%", "преимущества регрессии даёт простое правило", "«против последних 30-60 минут», вход по след. открытию, 2018-2024; в 2025 правило даже выше (1,44)"),
        ("0,01", "косинус весов 2025 к весам 2019", "веса нестабильны, а замороженная модель 2019 даёт тот же AUC; переобучение даёт до +0,008"),
        ("0 … 1 б.п.", "валовое преимущество топ-10% с 2022", "вход по след. открытию: 0,0; −0,1; 1,0; 0,8 (2022-2025); было 1,7-3,6 в 2018-21; внутри шума"),
        ("−0,51", "связь месячного AUC с автокорреляцией лага 1", "корреляция без тренда, p < 1e-6 (вход по закрытию): затухание следует за исчезновением отскока"),
        ("0,535-0,538", "AUC сегодня на Binance (2025-11..2026-09)", "отскока нет; валовое 0,2-0,7 б.п. на сделку топ-10%, неотличимо от нуля"),
        ("Bitstamp, не Binance", "что на самом деле лежит в файле", "BTC/USD, часы UTC+4/+5 до склейки 2024-09-13; тик $0,01 → $1 в 2022-08-10"),
    ]
    th = "".join(f'<div class="tile"><div class="big">{a}</div><div class="lab">{b}</div><div class="sub">{c}</div></div>' for a, b, c in tiles)
    lead = ('<div class="lead"><b>Ответ в одном абзаце.</b> Логистическая регрессия на 13 признаках находит один простой эффект: цена, растянутая за последние 20-60 минут, '
            'в среднем возвращается. До 2020 года в этом эффекте была заметная доля отскока bid-ask на тонкой бирже (Bitstamp), и вход по закрытию его «подбирал». '
            'Настоящий разворот остался, но ослаб вместе с рынком: спреды сузились, автокорреляция исчезла. Сегодня преимущество не превышает 1 б.п. на сделку и не отличается от нуля. '
            'Простое правило «против последних 30-60 минут» делает ту же работу, что и регрессия, поэтому регрессия — честная база для сравнения, а не источник преимущества.</div>')
    gl = ('<div class="gloss">'
          '<div><b>AUC</b> — вероятность, что модель поставит выше случайный «рост», чем случайное «падение». 0,5 = монета; 0,55 уже заметно для минутных данных.</div>'
          '<div><b>Вход по закрытию / по следующему открытию</b> — по какой цене открывается сделка: по закрытию бара сигнала (нереалистично, там сидит отскок) или по открытию следующего бара (достижимо).</div>'
          '<div><b>б.п.</b> — базисный пункт, 0,01% цены. Валовое — до издержек (по D-044 издержки в проекте 0, но здесь важна сама величина).</div>'
          '<div><b>Отскок bid-ask</b> — цена закрытия случайно напечатана на стороне покупки или продажи; следующий бар «возвращает» её. Корреляция соседних минут отрицательна без всякого прогноза.</div>'
          '<div><b>Fade</b> — играть против последнего движения: продавать после роста, покупать после падения.</div>'
          '<div><b>Цвета:</b> янтарный/фиолетовый — знак эффекта (разворот/продолжение), цвета горизонтов h0-h2 не используются (оси горизонтов здесь нет).</div></div>')
    f8a = rus_fig("fig8a_window_2019", "Реальное окно: 2019-06-17 08:00-12:00 по времени файла (04:00 UTC), регрессия июня 2019 и её сделки",
                  "6 непересекающихся сделок топ-10%, 4 выиграно, среднее +18,3 б.п. валовых в этом окне (иллюстрация, не оценка: весь месяц топ-10% при входе по следующему открытию дал в среднем −1,99 б.п., доля выигрышей 58%).<br>"
                  "Модель играет против растянутых движений: продаёт после пробега выше полосы и покупает после падения ниже неё." + SRC_NOTE, "fig8a", "s0")
    f8b = rus_fig("fig8b_window_2025", "Реальное окно: 2025-05-13 12:00-16:00 по времени файла (12:00 UTC), регрессия мая 2025 и её сделки",
                  "6 непересекающихся сделок топ-10%, 4 выиграно, среднее +2,0 б.п. валовых в этом окне (иллюстрация, не оценка: весь месяц топ-10% при входе по следующему открытию дал в среднем +0,72 б.п., доля выигрышей 54%).<br>"
                  "Та же логика, шесть лет спустя: сделки есть, а средний результат в окне +2,0 б.п. против +18,3 в окне 2019." + SRC_NOTE, "fig8b", "s0")
    body = (f'<div class="tiles">{th}</div>{lead}<h3>Ответ в десяти строках</h3>{ten_lines()}'
            + card("Словарь", gl)
            + card("Что регрессия делает на настоящей цене BTC (2019 и 2025)", html_fig(f8a, "fig8a") + html_fig(f8b, "fig8b"),
                   "На графике: линия закрытия, сигналы топ-10% как сделки и два признака, которые регрессия читает. Окна выбраны в T09, это иллюстрация, а не статистика.")
            + para("<b>Метод в двух фразах.</b> Для каждого месяца 2017-04..2025-07 логистическая регрессия на 13 признаках (EMA20, RSI14, MACD, BB20, ATR14, Stoch14, CCI20, доходности 1/5/10/15/20/30 минут), "
                   "обученная на 60 днях перед месяцем, предсказывает знак хода на 15 минут вперёд (мёртвая зона 5 б.п.); два варианта метки: вход по закрытию бара сигнала (H51) и по следующему открытию (H59). "
                   "Контроль: воспроизведение H51 и H59 до 6e-7 AUC (T09, sanity). Все числа — разведочные, только dev-данные, без SPEC; данные после 2025-07-20 не читались."))
    return body


def sec1():
    fl = fig_lag_scan()
    fs, rows = fig_seams()
    tbl = table([[r[0], ru(r[1], 1, True), (str(r[2]) if r[2] else "-"), ru(r[3], 1)] for r in rows],
                ["минута шва (время файла)", "доходность закрытия, б.п.", "ранг среди минут дня", "99-й процентиль |r| года, б.п."])
    ev = table([
        ["2018-03-11", "Переход на летнее время в сдвинутом файле: пропущен один настоящий час, скачок 109 б.п. за минуту", "совпадает по дате, но слишком мал, чтобы сдвинуть месячный AUC"],
        ["каждую весну и осень до 2024", "Швы перехода: часы файла сдвигаются на 1 час (пример: 2019-11-03, час перепутан)", "выбросы признаков дважды в год; 0,57% якорей, исключены в Q2, Q3 и часовом разрезе"],
        ["2022-08-10", "Тик Bitstamp BTC/USD с $0,01 до $1 (файл переключается внутри часа 12 по времени файла = 08-09 UTC). Доля минут с нулевой доходностью: 3-4% → 12% (окт. 2022) → 31% (дек. 2022)",
         "годовые столбцы 2022 смешаны с этим изменением"],
        ["2024-09-13 08:00 (время файла)", "Склейка: сдвиг UTC+4 → UTC, около 4 часов настоящих данных нет, меняется построение свечей; минут с нулевым объёмом 3,7% → 12,0%, доля баров с open = прошлый close 0,20 → 0,26",
         "2024 содержит склейку; в разбивке по часам до неё часы сдвинуты на 4-5 ч"],
        ["2025-03-15", "Пропуск 1160 минут; после него минут с нулевым объёмом 0,6-0,8%, корреляция зазора около 0", "внутри режима после склейки, вторая точка перелома"],
    ], ["дата", "что произошло", "как учтено в T09"])
    body = (para("<b>Находка о происхождении данных.</b> Файл <code>Bitcoin_BTCUSDT.csv</code> (и cut-версия, на которой работает вся тактическая сессия) — это <b>не Binance BTC/USDT</b>, а Bitstamp BTC/USD. "
                 "Метки времени идут в UTC+4 в американское летнее время и в UTC+5 в остальное, вплоть до склейки 2024-09-13 08:00 по времени файла; после неё это UTC. "
                 "Все механизмы дальше — про Bitstamp, более тонкую долларовую биржу. Этот вывод — самый последствиеёмкий в работе (H91, раздел 2).",
                 "<b>Как установлено.</b> API Bitstamp с этой машины недоступно, прямого сопоставления свечей нет. Площадка определена по дате смены тика (совпадает с блогом Bitstamp до часа), по исключению "
                 "(файл начинается до появления Binance) и по уровню объёма. Уверенность высокая, но это не доказательство. По Binance сверено 96 окон по 1000 минут (рисунок ниже): совпадение подтверждает сдвиг часов и его исчезновение в 2024-09.")
            + card("Сдвиг часов файла и автокорреляция: Binance против файла", html_fig(fl, "lag_scan"),
                   "Источник: runs/tactical/report/regdata/lag_scan.csv (сканер из рабочей папки H91, скопирован как есть; запрашивал публичное API Binance; в T09 не используется). "
                   "Среднее по году корреляции при лучшем сдвиге: 0,62-0,79 в 2018-20, 0,85-0,91 с 2021 (в 2017 корреляция низкая, 0,17: причина не выяснена). Это не ошибка сдвига: у тонкого файла своя микроструктура. 2025-03-15 (корреляция 0,21): окно, вероятно, задело пропуск 1160 минут.")
            + card("Швы данных в самом файле", html_fig(fs, "seams"),
                   "Считано напрямую из cut-файла (до 2025-07-20). Таблица — те же значения. Осенью самые большие минутные ходы дня приходятся на разные минуты (2017: 07:00, 145 б.п.; 2019: 06:00, 66; 2020: 04:00, 56), фиксированного правила для шва нет, и эти минуты не показаны.")
            + card("Шовные точки в числах", tbl)
            + card("События данных, которые надо учитывать (H91, раздел 2)", ev,
                   "Это не рыночные события, а изменения построения данных. Складки на 2022 и 2024 в годовых столбцах T09 смешаны с этими изменениями."))
    return body


def sec2():
    f1 = rus_fig("fig1_coef_heatmap", "В1: коэффициенты регрессии по месяцам",
                 "13 z-нормированных признаков, C = 0,1, обучение на 60 днях до месяца. Фиолетовый = отрицательный вес (играть против), янтарный = положительный. Вес переходит между коллинеарными признаками<br>"
                 "(положение: ema20, bb20, stoch14, cci20 против гистограммы MACD), а получаемая оценка остаётся похожей. Штрих-пунктир: 2022-08-10 (тик $0,01 → $1) и 2024-09-13 (склейка файла)." + SRC_NOTE, "fig1", "s2")
    f2 = rus_fig("fig2_importance_by_year", "В1: какие признаки несут сигнал? (годовые средние по месяцам, вход по закрытию, если не оговорено)",
                 "Каждый признак — версия вопроса «насколько растянуты последние 20-60 минут»; признаки коллинеарны, поэтому потеря при удалении одного ≈ 0 (остальные подменяют его), а перемешивание признака или группы ломает оценку.<br>"
                 "Больше всего несут признаки положения; волатильность (atr14) не несёт ничего." + SRC_NOTE, "fig2", "s2")
    # the subsets block: label lines contain spaces ('top1: macd ...'), parse by hand
    ls = block("Q1 subsets")
    rws = []
    names = {"top1:": "1 признак (macd)", "top2:": "2 признака (macd + bb20)", "top3:": "3 признака (+ cci20)"}
    for ln in ls:
        if ln.startswith("top") and "±" in ln:
            lab = ln.split(":")[0] + ":"
            rws.append([names[lab]] + [pm(x, 3) for x in re.findall(r"[\d.]+±[\d.]+", ln)])
        elif ln.startswith("all 13"):
            rws.append(["все 13 признаков"] + [pm(x, 3) for x in re.findall(r"[\d.]+±[\d.]+", ln)])
    sh = []
    start = [i for i, ln in enumerate(ls) if ln.startswith("share of")][0]
    for ln in ls[start + 2:]:
        p = ln.split()
        sh.append([{"top1": "1 признак", "top2": "2 признака", "top3": "3 признака"}[p[0]]] + [ru(float(x), 2) for x in p[1:]])
    gp = table(rws, ["модель"] + YEARS)
    gs = table(sh, ["доля преимущества полной модели"] + YEARS)
    body = (para("<b>Вопрос:</b> что в 13 признаках на самом деле несёт сигнал? <b>Ответ:</b> один фактор — «насколько цена растянута против недавнего диапазона». "
                 "Любая из групп (позиция, доходности, осцилляторы) сама по себе даёт почти весь AUC (в 2025-26 на Binance: 0,532-0,538, у волатильности 0,501-0,504), а <i>веса</i> между коллинеарными признаками скачут по месяцам. "
                 "Косинус к весам 2019 падает с 0,89 (2019) до 0,10 (2024) и 0,01 (2025) — а замороженная модель 2019 даёт тот же AUC.")
            + card("Коэффициенты по месяцам", html_fig(f1, "fig1"), "Веса нестабильны, оценка стабильна: нельзя «читать» регрессию по весам.")
            + card("Важность признаков и групп", html_fig(f2, "fig2"))
            + card("Сколько преимущества дают 1, 2, 3 признака (AUC по годам, ±95% t)", gp, "Три признака — macd, bb20, cci20, выбранные задним числом по потере при удалении (описательно).")
            + card("Доля преимущества полной модели", gs, "(AUC − 0,5) / (AUC полной − 0,5). Два признака уже дают 0,75-1,15 от полной модели.")
            )
    return body


def sec3():
    f3 = rus_fig("fig3_partial_dependence", "В2: форма. Маргинальная P(рост) почти прямая линия по каждому главному признаку; хвосты чуть круче",
                 "Точки = доля ростов в 20 квантильных корзинах пула эпохи (95% полоса Уилсона, n_eff = n x 10/15); пунктир = средняя P(рост) эпохи. Линия модели — УСЛОВНАЯ зависимость:<br>"
                 "у macd и cci20 знак противоположен маргинальным точкам (подавители в коллинеарном наборе)." + SRC_NOTE, "fig3", "s3")
    f3b = rus_fig("fig3b_tail_edge", "В2: преимущество сидит в хвосте. Игра против самого растянутого дециля выигрывает 56-61% ходов в 2017-19 и 53-55% в 2024-25; средние децили — монета",
                  "Доля попаданий правила «предсказать ход против знака признака» по децилям |признака| (маскировка мёртвой зоны, вход по закрытию). Пунктир = 0,5. Симметрия: тот же fade работает выше и ниже<br>"
                  "(преимущество трёх нижних корзин против трёх верхних — в tables.txt)." + SRC_NOTE, "fig3b", "s3")
    body = (para("<b>Вопрос:</b> линейная регрессия — что именно она ловит, плавный наклон или хвосты? <b>Ответ:</b> наклон слабый и почти прямой, а прибыльная часть — хвост. Самый растянутый дециль выигрывает 56-61% (2017-19) и 53-55% (2024-25), "
                 "средние децили — монета.",
                 "<b>Зависимость от размера хода</b> (H91, поправка к более раннему утверждению «от размера не зависит»): для малых ходов разворота нет, для умеренных и крупных есть, а для самых больших дальше не растёт. "
                 "Наклон по квинтилям |z|: Q2 +0,046, Q3 −0,052, Q4 −0,098, Q5 −0,079; корреляция сильнее всего в Q5 (−0,122). У De Nicola (2021) та же форма на 1-минутных данных Bitstamp: −0,156 для всех ходов, −0,200 для ходов от 1σ, падение до −0,045 на 6σ. "
                 "Утверждение H52 «крупные ходы не более предсказуемы» совместимо: AUC меряет знак, а не размер разворота.")
            + card("Частная зависимость", html_fig(f3, "fig3"), "Линии модели и точки намеренно не совпадают по знаку у macd и cci20: это не противоречие, а коллинеарность (несколько признаков выражают одно и то же).")
            + card("Преимущество по децилям растянутости", html_fig(f3b, "fig3b")))
    return body


def best_rule_table(kind):
    ls = block(f"Q3 [{kind} entry] the best simple rule")
    rows = []
    for ln in ls[1:]:
        p = ln.split()
        rows.append([p[0], ru(float(p[1]), 3), str(int(float(p[2]))), ru(float(p[3]), 3), ru(float(p[4]), 2),
                     (str(ru(float(p[8]), 3)) if p[8] != "NaN" else "-"), (ru(float(p[9]), 2) if p[9] != "NaN" else "-")])
    return table(rows, ["год", "AUC регрессии", "лучшее k, мин (в году)", "AUC правила", "доля преимущества регрессии", "AUC при k прошлого года", "доля (k прошлого года)"])


def sec4():
    f4a = rus_fig("fig4a_impulse_auc_close", "В3 импульсный отклик, вход по закрытию (H51): AUC x 100 правила «против последних k минут»",
                  "Строки: лаг k возвращаемой доходности; столбцы: горизонт h предсказываемого хода. Годовое среднее месячных значений; у строки k = 1 95% полуширина в среднем 0,010 AUC на клетку<br>"
                  "(ширины всех клеток — в tables.txt). Янтарный = fade платит (AUC выше 0,5, положительные б.п.), фиолетовый = проигрывает." + SRC_NOTE, "fig4a", "s4")
    f4b = rus_fig("fig4b_impulse_auc_nopen", "В3 импульсный отклик, вход по следующему открытию (H59): AUC x 100 правила «против последних k минут»",
                  "Строки: лаг k; столбцы: горизонт h. Годовое среднее месячных значений; у строки k = 1 95% полуширина в среднем 0,009 AUC на клетку (ширины всех клеток — в tables.txt).<br>"
                  "Янтарный = fade платит, фиолетовый = проигрывает." + SRC_NOTE, "fig4b", "s4")
    f4c = rus_fig("fig4c_impulse_bps_nopen", "В3 импульсный отклик, вход по следующему открытию (H59): средняя знаковая доходность правила «против последних k минут» (б.п. на сигнал, без издержек)",
                  "Строки: лаг k; столбцы: горизонт h. Годовое среднее месячных значений. Янтарный = fade платит (положительные б.п.), фиолетовый = проигрывает." + SRC_NOTE, "fig4c", "s4")
    body = (para("<b>Вопрос:</b> на каком лаге и горизонте разворот живёт и какое простое правило его заменяет? <b>Ответ:</b> с 2021 лаг k = 1 (последняя минута) почти ничего не даёт; работают лаги 15-60 минут на горизонтах 15-60 минут. "
                 "Правило «против последних 30-60 минут» воспроизводит 90-100% преимущества регрессии при входе по следующему открытию в 2018-2024 и не хуже регрессии с 2022. "
                 "В 2025 доля даже выше 1 (1,44): регрессия на этом хвосте шумит больше простого правила. «Выбор k по прошлому году» (causal) сохраняет 0,77-1,03.")
            + card("Вход по закрытию (H51)", html_fig(f4a, "fig4a"))
            + card("Вход по следующему открытию (H59)", html_fig(f4b, "fig4b"))
            + card("Вход по следующему открытию: б.п. на сигнал", html_fig(f4c, "fig4c"))
            + card("Лучшее простое правило «против последних k минут» при h = 15 против регрессии, вход по следующему открытию", best_rule_table("nopen"),
                   "Доля = (AUC правила − 0,5) / (AUC регрессии − 0,5). «k прошлого года» — честный вариант без заглядывания вперёд.")
            + "<details><summary>То же для входа по закрытию (H51)</summary>" + card("Простое правило против регрессии, вход по закрытию", best_rule_table("close"),
                                                                                      "Здесь доля ниже (0,69-0,89 в 2017-22): часть преимущества входа по закрытию — отскок, которого у простого правила на k ≥ 15 минут нет.") + "</details>")
    return body


def sec5(w):
    t0, win, hvq, nlow, r0 = w
    f9 = rus_fig("fig9_microstructure", "В4: сколько в этом отскока bid-ask? Около 0,02-0,04 AUC в 2017-20, с 2021 ничего",
                 "Вход по закрытию бара сигнала льстит результату отскоком (закрытие с равной вероятностью напечатано на стороне bid или ask); вход по следующему открытию или по середине бара его убирает. Светлые столбцы в левой нижней панели —<br>"
                 "вход по следующему открытию, сплошные — по закрытию; прокси спреда — собственный размах high-low бара сигнала (терцили по обучающему периоду)." + SRC_NOTE, "fig9", "s5")
    fb = fig_bounce(t0, win, hvq, nlow, r0)
    corr = table([
        ["следующая доходность, закрытие → закрытие", "−0,079", "−0,003", "−0,011"],
        ["следующий зазор, закрытие t → открытие t+1", "<b>−0,229</b>", "−0,104", "−0,052"],
        ["следующая доходность, открытие → закрытие", "+0,038", "+0,041", "+0,009"],
    ], ["корреляция r_t с целью", "2017H2-2020", "2021-23", "2024-25"])
    ls = block("Q4 microstructure")
    start = [i for i, ln in enumerate(ls) if ln.startswith("spread / bounce proxies")][0]
    prx = []
    for ln in ls[start + 2:]:
        p = ln.split()
        prx.append([p[0], ru(float(p[1]), 3), ru(float(p[2]), 1), ru(float(p[3]), 1), ru(float(p[4]), 3, True), ru(float(p[5]), 3), ru(float(p[6]), 1)])
    px = table(prx, ["год", "Корвин-Шульц, б.п.", "размах high-low, б.п.", "объём, BTC/бар", "AC(1)", "VR(1,15)", "спред Ролла, б.п."])
    yauc = grid_table("Q4 microstructure", {"close": "вход по закрытию", "nopen": "вход по след. открытию", "mid": "середина след. бара"}, first="способ входа")
    refut = table([
        ["Дискретность тика / относительный тик 1/цена", "опровергнуто", "тик был $0,01 при сильном развороте (относительный тик 0,001-0,03 б.п. при спреде Ролла 6-10 б.п.); разворот исчез ещё до тика $1 (2022-08-10)"],
        ["Транзиентное воздействие после всплесков агрессивных ордеров", "опровергнуто как главный драйвер", "разворот сильнее всего в малообъёмных барах, а посылка предполагала односпредовый Binance"],
        ["Предложение ликвидности на большом объёме (Campbell-Grossman-Wang 1993)", "противоречит", "упорядочение по квинтилям объёма идёт в обратную сторону"],
        ["Кластеры круглых цен (Osler 2003)", "не подтверждено (уверенность средняя)", "разворот слабее у круглых $100 (−0,043 против −0,082); круглые $1000 и направление пересечения не разделены"],
    ], ["кандидат", "вердикт", "основание"])
    body = (para("<b>Вопрос:</b> сколько «разворота» — это просто отскок между bid и ask? <b>Ответ:</b> примерно треть преимущества 2018-20 при входе по закрытию. Весь минутный «разворот» сидит в скачке от закрытия t до открытия t+1; собственный ход следующего бара его не несёт. "
                 "Это достижимо только если входить по закрытию, а на самом деле рынок даёт следующее открытие — этим H59 и H60 и убирают этот кусок. Метка «отскок bid-ask» из H59/H60 подтверждена (H91).",
                 "<b>Концентрация в тонких барах.</b> Наклон r(t+1) на r(t) идёт от −0,242 в нижнем квинтиле относительного объёма до −0,019 в верхнем; для зазора −0,265 … −0,060. Мало сделок: закрытие напечатано с одной стороны книги. "
                 "<b>Спред Ролла</b> (пулом): 10,4; 7,9; 6,4; 6,4 б.п. в 2017, 2018, 2019, 2020. <b>Независимая работа на той же бирже:</b> De Nicola (2021, Bitstamp 1-минутные бары 2015-2018) даёт AC(1) = −0,145 и приписывает автокорреляцию ≤ 15 минут отскоку bid-ask; "
                 "гауссово отображение ρ = −0,14 даёт AUC на 1 бар около 0,437 (симуляция) — совпадает с нашей ячейкой «1 минута вперёд».")
            + card("Корреляция r_t с тремя целями по эпохам (H91, 1a)", corr, "Весь эффект сидит в зазоре от закрытия до следующего открытия; ход «открытие → закрытие» даже слабо продолжается.")
            + card("Микроструктура: отскок по годам и по размаху бара", html_fig(f9, "fig9"))
            + card("Тот же отскок на настоящих барах Bitstamp (2018)", html_fig(fb, "bounce_window"),
                   f"Правило выбора окна (зафиксировано до просмотра окна, считано из cut-файла): внутри 2018 взять 10% самых малоторгуемых часов (сумма объёма за час ≤ {hvq:.0f} BTC; {nlow} часов), "
                   "исключить дни перехода на летнее время (2018-03-11, 2018-11-04), бары с нулевым объёмом и бары с |r_t| < 5 б.п.; выбрать бар с наибольшим разворотом на следующем открытии, "
                   "rev = −знак(r_t) · (open_{t+1} / close_t − 1); показано ±20 минут вокруг него. Первая версия правила («наибольший |зазор|») выбрала плоский бар 2018-12-02 03:47 (0,09 BTC, O=H=L=C) — "
                   "это залежавшаяся печать, после неё цена осталась на новом уровне, а не вернулась; я заменил правило на разворотное и сообщаю об этом.")
            + card("Годовой AUC по способу входа (±95% t по месяцам)", yauc)
            + card("Прокси спреда и отскока по годам (средние месячных значений)", px, "Оценка Корвина-Шульца и Ролла цензурированы нулём: нулевые годы — нижние границы.")
            + card("Что отвергнуто как механизм (H91)", refut))
    return body


def sec6():
    f5a = rus_fig("fig5a_conditions_close", "В5: когда это работает? AUC по условию и периоду, вход по закрытию (H51)",
                  "Корзины из квантилей обучающего периода (без заглядывания вперёд); столбцы = среднее месячных AUC с 95% t-интервалом по месяцам; пунктир = 0,5. В 2024-25 большинство корзин около 0,52-0,53;<br>"
                  "выходные сохраняют 0,547 и (в широком шуме) самые большие последние ходы 0,55-0,57." + SRC_NOTE, "fig5a", "s6")
    f5b = rus_fig("fig5b_conditions_nopen", "В5: когда это работает? AUC по условию и периоду, вход по следующему открытию (H59)",
                  "Корзины из квантилей обучающего периода (без заглядывания вперёд); столбцы = среднее месячных AUC с 95% t-интервалом по месяцам; пунктир = 0,5. В 2024-25 большинство корзин около 0,52-0,53;<br>"
                  "выходные сохраняют 0,547 и (в широком шуме) самые большие последние ходы 0,55-0,57." + SRC_NOTE, "fig5b", "s6")
    body = (para("<b>Вопрос:</b> есть ли условия, при которых регрессия всё ещё работает? <b>Ответ:</b> в 2024-25 почти нигде выделяющихся нет. Выходные: 0,547 против 0,521 в будни. Самые большие последние ходы: 0,55-0,57, но с широким шумом. Структуры по часам суток нет "
                 "(часы пересчитаны в UTC; до склейки 2024-09 это сдвиг файла на 4-5 часов, дни перехода на летнее время исключены).",
                 "<b>Для зазора (до 2020)</b> эффект сильнее всего в малообъёмных, «плоских», тихих барах (H91, Q5): так и должен выглядеть отскок.")
            + card("Вход по закрытию", html_fig(f5a, "fig5a"))
            + card("Вход по следующему открытию", html_fig(f5b, "fig5b")))
    return body


def decomp_table(kind):
    ls = block("Q6 decay decomposition")
    # find '[close] cells = volatility tercile x volume tercile' / '[nopen] ...'  inside the single block
    i0 = [i for i, ln in enumerate(ls) if ln.startswith(f"[{kind}] cells = volatility tercile x volume")][0]
    rows = []
    for ln in ls[i0 + 2:i0 + 11]:
        p = ln.split()
        rows.append([p[0], ru(float(p[1]), 3), ru(float(p[4]), 3, True), ru(float(p[5]), 3, True), ru(float(p[6]), 3, True), ru(float(p[7]), 3, True), ru(float(p[8]), 2)])
    return table(rows, ["год", "AUC (полная, с переобучением)", "полное изменение против 2019", "из-за силы сигнала", "из-за состава условий", "из-за подстройки коэффициентов", "косинус к весам 2019"])


def sec7():
    f6 = rus_fig("fig6_decay_decomposition", "В6: почему затухло? Сам сигнал стал слабее, а рынок — менее «прыгучим»",
                 "Сверху: изменение AUC против 2019, разложенное на силу признаков / состав условий / подстройку коэффициентов (сумма = итог, ромбы). Переобучение добавляет < 0,01, а состав условий второстепенен;<br>"
                 "остальное — те же коэффициенты, предсказывающие меньше. Снизу: затухание следует за исчезновением минутной отрицательной автокорреляции (отскок bid-ask и разворот тонкой книги). Штрих-пунктир: тик Bitstamp $0,01 → $1<br>"
                 "(2022-08-10), склейка файла (2024-09-13), пропуск 1160 минут (2025-03-15); годовые столбцы 2022 и 2024 содержат смену тика и склейку, так что ступень в них смешана с изменением данных." + SRC_NOTE, "fig6", "s7")
    ls = block("Q6 market-quality proxies")
    corr_rows = []
    for kind in ("close", "nopen"):
        i0 = [i for i, ln in enumerate(ls) if ln.startswith(f"[{kind}]")][0]
        for ln in ls[i0 + 2:i0 + 8]:
            p = ln.split()
            nm = {"cs_median_bps": "Корвин-Шульц (медиана)", "hl_median_bps": "размах high-low (медиана)", "volume_mean": "объём (среднее)", "ac1": "автокорреляция лага 1", "vr_1_15": "VR(1,15)", "roll_bps": "спред Ролла"}[p[0]]
            corr_rows.append([("вход по закрытию" if kind == "close" else "вход по след. открытию"), nm, ru(float(p[1]), 3, True), ru(float(p[2]), 3, True), ("< 0,001" if float(p[3]) < 0.0005 else ru(float(p[3]), 3))])
    ct = table(corr_rows, ["вход", "прокси рынка", "Спирмен", "Пирсон без тренда", "p"])
    rank = table([
        ["1", "Эффективный спред и отскок площадки сузились", "<b>высокая</b>", "Спред Ролла 10,4 (2017), 7,9, 6,4, 6,4, 0 (2021), 2,4, 0 (2023), 1,9, 0,6 (2025); оценка цензурирована нулём. Корреляция r_t со следующим зазором: −0,25…−0,35 (2017), −0,15…−0,30 (2018-20), −0,10…−0,18 (2021), −0,05…−0,12 (2022-23), −0,03…−0,10 (2024), около 0 с 2025-04. Дневной объём Bitstamp упал с ≈11 тыс. BTC (2017) до ≈1,3 тыс. (2025): «ликвидность выросла» значит «эффективный спред сузился», не «объём вырос»."],
        ["2", "Эффективность коротких горизонтов выросла по всему рынку", "средне-высокая", "На Binance разворот тоже сжался: AC(1) −0,004 (2020), −0,015 (2021), +0,020 (2022). Sensoy (2019), Brauneis-Mestel (2018), Tang и др. (2025: показатель Хёрста 0,415 → 0,478), Kim-Hansen (2026, препринт); аналог в акциях — Chordia-Roll-Subrahmanyam (2005, 2008); гипотеза адаптивных рынков допускает падение эпизодами."],
        ["3", "Построение данных смешивает поздние переломы", "средняя", "Тик $1 (2022-08-10), склейка (2024-09-13), пропуск 1160 минут (2025-03-15). Перелом AUC около 2024-09 совпадает со склейкой в грубой датировке H72; он смешан, но артефактом не доказан."],
        ["4", "Ценообразование могло уйти в бессрочные фьючерсы", "низкая", "Медленный механизм, согласуется с постепенным затуханием после 2021, но ни одно событие не совпадает с переломом. Цифры о доле деривативов (CCData ≈ 70% в окт. 2024, FOW ≈ 48% в 2021) не проверены."],
    ], ["#", "причина", "уверенность", "основание"])
    not_expl = para("<b>Не объяснение:</b> события только Binance не могут сдвинуть ряд Bitstamp (период нулевых комиссий 2022-07-08..2023-03-22, ступени скидки BNB, запуск бессрочных фьючерсов Binance); халвинги, банкротство FTX и запуск спотовых ETF в США "
                    "не совпадают ни с одним переломом.")
    ev1 = table([
        ["2017-12-10 / 12-17", "Запуск фьючерсов BTC на Cboe и CME", "рядом, не точно; максимум на уровне режима", "-", "-", "общие знания, не перепроверено"],
        ["янв-февр 2018", "Конец пузыря 2017; рекордный объём Bitstamp (≈15-21 тыс. BTC/день в файле)", "правдоподобная смена режима, конкретного события нет", "-", "-", "данные файла"],
        ["~2018-07 / 2019-07-14", "Ступени скидки BNB на комиссии Binance (50% → 25%)", "нет (только Binance)", "нет", "нет", "Decrypt; Binance.US"],
        ["сент 2019", "Бессрочные BTC/USDT-фьючерсы Binance", "нет", "нет", "нет", "Decrypt 2019"],
        ["2020-05-11", "Халвинг", "нет", "нет", "нет", "общеизвестно"],
        ["2020-09-10", "Bitstamp переходит на матчинговый движок Nasdaq (скорее всего единственная миграция)", "-", "<b>кандидат</b>: статистика отскока в файле спадает постепенно через конец 2020 - начало 2021; причинная связь не доказана", "-", "Forklog 2020-09-11; Finextra 2020"],
        ["ноя 2020 - март 2021", "Рост цены $16 тыс. → $47-57 тыс.; приток институционалов", "-", "правдоподобно, рыночное", "-", "данные цены"],
        ["2021-01", "Разбиение периодов Tang и др. (Хёрст 0,415 → 0,478); разбиение выбрано авторами, не оценено", "-", "слабое подтверждение", "-", "arXiv 2402.11930"],
        ["2022-02-15", "Тик бессрочного BTCUSDT Binance 0,01 → 0,1 (фьючерсы, не спот)", "-", "-", "нет", "MultiCharts forum"],
        ["2022-07-08..2023-03-22", "Нулевые комиссии на споте BTC Binance", "нет", "нет", "нет (и только Binance)", "The Block; FXStreet; Bloomberg"],
        ["2022-11-11", "Банкротство FTX", "нет", "нет", "нет", "The Block"],
        ["2024-01-10/11", "Одобрение и начало торгов спотовых ETF в США", "-", "-", "нет (за 8 месяцев)", "общеизвестно"],
        ["2024-04-20", "Халвинг", "-", "-", "нет", "общеизвестно"],
        ["2025-06-02", "Robinhood завершает покупку Bitstamp", "-", "-", "нет", "Robinhood newsroom"],
    ], ["дата", "событие", "перелом 2018-03", "перелом 2021-03", "перелом 2024-09", "источник"])
    summ = table([
        ["2018-03", "конкретного события биржи нет: скорее смена режима или волатильности после пузыря, либо чисто статистическая точка"],
        ["2021-03", "правдоподобная, но недоказанная комбинация: движок Nasdaq на Bitstamp, ралли, сузившее относительный тик, и рыночная эффективность"],
        ["2024-09", "смешан со склейкой данных"],
    ], ["перелом", "вывод H91"])
    lit = table([
        ["De Nicola 2021, <i>Ledger</i> 6:58-80", "2015-03..2018-06", "<b>Bitstamp</b> BTC/USD, 1 мин", "AC(1): 1м −0,145; 5м −0,102; 15м −0,058; 30м −0,044; 1ч −0,056; 2ч −0,086; 4ч −0,056 (все p < 1e-5); 1д −0,007 (н.з.). На 1м разворот −0,200 после ходов ≥ 1σ; автор приписывает ≤ 15м отскоку", "высокая; та же площадка"],
        ["Jaquart, Dann, Weinhardt 2021, <i>JFDS</i> 7:45-66", "2019-03..2019-12", "Bloomberg BTC/USD, 1 мин", "Точность логистики на 1/5/15/60 мин: 0,511/0,518/0,520/0,539; GRU 0,518/0,525/0,536/0,557. Предсказуемость растёт с горизонтом; после издержек 30 б.п. отрицательно", "высокая"],
        ["Tang и др. 2025, <i>Fractal Fract.</i> 9(10):635", "2019-04..2022-05", "Reuters BTC/USD, 10 мин", "Хёрст короткой шкалы 0,415 ± 0,006 (2019-20) → 0,478 ± 0,004 (2021-22): возврат к среднему затухает", "высокая по числам; слабая как датировка"],
        ["Sensoy 2019, <i>FRL</i> 28:68-73", "~2013-2018", "BTCUSD, BTCEUR, внутри дня", "Эффективнее внутри дня с 2016; ликвидность повышает эффективность, волатильность снижает", "высокая по направлению"],
        ["Takaishi, Adachi 2020, <i>APFM</i> 27:145-154", "ранние годы..~2019", "BTC", "Хёрст < 0,5 в начале, около 0,5 после ~2013; позже нарушается эпизодами", "средняя"],
        ["Zargar, Kumar 2019, <i>RIBF</i> 47:344-353", "(не проверено)", "BTC, 15/30/60/120 мин, 1д", "Неэффективен на высокой частоте, без памяти на дневной; эффективность меняется во времени", "средне-высокая (аннотация)"],
        ["Catania, Sandholdt 2019, <i>JRFM</i> 12(1):36", "(не проверено)", "BTC до 1д", "Предсказуемость доходности до 6 ч, на 1 д нет; предсказуемость реализованной волатильности растёт со временем", "средне-высокая (аннотация)"],
        ["Aslan, Sensoy 2020, <i>FRL</i> 35:101298", "(не проверено)", "крипто, внутри дня", "U-образная связь эффективности с частотой выборки", "средняя"],
        ["Brauneis, Mestel 2018, <i>Econ. Letters</i> 165:58-61", "(не проверено)", "много монет, дневные", "Менее предсказуемы при росте ликвидности («BTC наименее предсказуем» не проверено)", "средняя"],
        ["Shen, Urquhart, Wang 2022, <i>Fin. Review</i> 57(2)", "(не проверено)", "BTC, получасовые сессии", "Первая половина часа предсказывает последнюю (моментум); сильнее при большом объёме и волатильности; приписывается предложению ликвидности", "высокая по знаку"],
        ["Wen, Bouri, Xu, Zhao 2022, <i>NAJEF</i> 62:101733", "2013-03..2020-05", "BTC и др., высокая частота", "И внутридневной моментум, и разворот в зависимости от скачков, FOMC, ликвидности, COVID", "низкая (платная, аннотация)"],
        ["Miralles-Quirós 2022, <i>J. Appl. Econ.</i> 25(1):1294-1313", "(не проверено)", "BTC, часовые, 6-24ч", "Перереакция после отрицательных часовых шоков, затем разворот", "средняя"],
        ["Borgards 2021, <i>NAJEF</i> 57:101428", "(не проверено)", "20 монет", "Моментум держится дольше, чем у акций США; метод ex post, не тест прогноза", "средняя"],
        ["Kim, Hansen 2026, arXiv 2607.09426 (препринт)", "2021-01..2024-10", "<b>Binance бессрочные</b> (BTC и 5 др.)", "Мало 1-минутной корреляции в большинстве минут; сильная отрицательная AC на отметках четверти часа; внеобучающий R² 3,37% для 10-секундной доходности после границы", "средняя (не рецензировано)"],
        ["Chu, Zhang, Chan 2019, <i>IRFA</i> 64:221-231", "(не проверено)", "BTC, ETH, высокая частота", "Эффективность меняется во времени (гипотеза адаптивных рынков); типы новостей и событий могут быть незначимы", "средняя"],
        ["Akyildirim, Goncu, Sensoy 2021, <i>Ann. Oper. Res.</i> 297:3-36", "(не проверено)", "12 монет, минуты-дни", "Точность 55-65%, SVM лучший", "низко-средняя (вероятно оптимистично)"],
    ], ["работа", "период", "площадка / данные", "результат", "уверенность"])
    body = (para("<b>Вопрос:</b> почему преимущество затухло и совпадает ли это с датированными событиями? <b>Ответ:</b> сильнее всего затух сам сигнал (те же веса и условия теперь предсказывают меньше), а не состав условий. "
                 "Он следует за исчезновением отскока: корреляция месячного AUC с автокорреляцией лага 1 — −0,51 после удаления линейного тренда (p < 1e-6, 100 месяцев; месяцы автокоррелированы, так что p завышена). "
                 "Для входа по следующему открытию связь с автокорреляцией лага 1 пропадает (+0,04, p = 0,68), зато остаётся связь с размахом high-low (−0,43) и спредом Ролла (−0,45), оба p < 0,001: то, что переживает следующее открытие, следует за общим сужением спредов, а не за самим отскоком.")
            + card("Разложение затухания и качество рынка", html_fig(f6, "fig6"))
            + card("Разложение падения AUC против 2019, вход по закрытию (клетки волатильность x объём по терцилям 2019)", decomp_table("close"),
                   "Для 2021: полное −0,049 = сила −0,069 + состав +0,012 + подстройка +0,008; для 2024: −0,068 = −0,075 + 0,006 + 0,001 (округление).")
            + card("То же, вход по следующему открытию", decomp_table("nopen"), "Вариант с клетками «волатильность x размах high-low» лежит в tables.txt; в 2025 состав условий у входа по следующему открытию даёт −0,039 (малая выборка условий).")
            + card("Связь месячного AUC с прокси рынка (100 месяцев)", ct, "Пирсон — после удаления линейного тренда из обоих рядов. Месяцы автокоррелированы; p оптимистичны.")
            + card("Причины затухания по убыванию уверенности (H91, 1b)", rank) + not_expl
            + card("Датированные события против переломов (H91, раздел 2)", ev1,
                   "Переломы взяты из H72: плато направленного AUC около 0,57, 0,545 и 0,521, sup-Chow p = 0,004; плотно датирован только перелом 2018-03. Ни одно рыночное событие не совпадает ни с одним переломом.")
            + card("Итог по переломам", summ)
            + "<details><summary>Опубликованные работы о минутной автокорреляции в криптовалютах и её тренде (H91, раздел 3)</summary>"
            + card("Работы", lit, "Пробел в литературе: нет опубликованных оценок дат перелома в минутной автокорреляции BTC и нет годовых 1-минутных рядов на споте за 2022-25 — наши измерения здесь новые, а не повтор.") + "</details>")
    return body


def sec8():
    f7 = rus_fig("fig7_breakeven_cost", "В7: экономика. Валовое преимущество было около 4 б.п. при закрытии и 1,7-3,6 при следующем открытии в 2018-21, а с 2022 — 0-1 б.п.",
                 "Топ-10% сигналов каждого месяца (порог из обучающих предсказаний), непересечение не обеспечено, без издержек. Столбцы: среднее по месяцам с 95% t-интервалом. Безубыточная стоимость = высота столбца;<br>"
                 "с 2022 она внутри собственной полосы шума и ниже любых реальных издержек." + SRC_NOTE, "fig7", "s8")
    ls = block("Q7 economic size")
    d = {}
    for ln in ls:
        p = ln.split()
        if p and p[0] in ("bps_close", "bps_nopen", "bps_mid", "bps_close_model_at_nopen", "hit_close", "hit_nopen"):
            d[p[0]] = p[1:]
    nm = {"bps_close": "вход по закрытию, б.п.", "bps_nopen": "вход по след. открытию, б.п.", "bps_mid": "середина след. бара, б.п.", "bps_close_model_at_nopen": "модель закрытия, но вход по след. открытию, б.п.",
          "hit_close": "доля выигрышей, закрытие", "hit_nopen": "доля выигрышей, след. открытие"}
    rows = [[nm[k]] + [(pm(x, 3) if k.startswith("hit") else pm(x, 2)) for x in v] for k, v in d.items()]
    t = table(rows, ["топ-10%, валовое"] + YEARS)
    tr_ = [l for l in ls if l.startswith("trades per month")]
    body = (para("<b>Вопрос:</b> сколько это стоит в б.п.? <b>Ответ:</b> безубыточная стоимость сделки равна среднему валовому доходу сигнала. При входе по следующему открытию: 6,0 (2017), 2,5; 1,7; 2,8; 3,6 (2021), затем 0,0; −0,1; 1,0; 0,8 (2022-2025), при полуширине интервала ±0,8-1,5 б.п. "
                 "С 2022 преимущество внутри шума. Издержки в проекте по D-044 ноль, но и при нулевых издержках валовое ≈ 0 внутри шума.")
            + card("Безубыточная стоимость сделки", html_fig(f7, "fig7"))
            + card("Валовое преимущество топ-10% по годам (±95% t по месяцам)", t, "В среднем 403-456 сигналов топ-10% на месяц; сделки не непересекающиеся, поэтому n_eff меньше. Вход по середине бара даёт близкий результат к следующему открытию."))
    return body


def sec9():
    f10 = rus_fig("fig10_hourly", "В8: часовой двойник — тоже разворот. Строки k = лаг, против которого играем, столбцы h = горизонт",
                  "Игра против последних 1-6 часов предсказывает следующие 1-8 часов (AUC 0,53-0,58 в 2019 и 2022), игра против последних 24 часов — следующие 8-24 часа (0,55-0,60). К 2025 короткие fade на 0,50-0,53, держится только 24-часовой (0,54-0,56).<br>"
                  "Часовая регрессия упала с 0,58-0,59 (2019-20) до 0,54 (2024-25); её лучшее одиночное правило даёт 0,76-1,6 преимущества регрессии." + SRC_NOTE, "fig10", "s9")
    ls = block("Q8 the hourly cousin")
    i0 = [i for i, ln in enumerate(ls) if ln.startswith("yearly AUC")][0]
    rows = []
    for ln in ls[i0 + 1:i0 + 8]:
        y, v = ln.split()
        rows.append([y, pm(v, 3)])
    t = table(rows, ["год", "AUC часовой регрессии H62 (горизонт 4 ч, вход по след. открытию)"])
    body = (para("<b>Вопрос:</b> а часовая модель H62 — то же самое? <b>Ответ:</b> да, это тоже разворот (RSI, расстояние до VWAP и EMA на 1-24 часах), но затухший меньше: 0,584 и 0,593 в 2019-20 против 0,546 и 0,540 в 2024-25. "
                 "Ряд неровный: 0,544 в 2021 и 0,566 и 0,578 в 2022-23; полуширина в 2025 ±0,057 (6 месяцев). Часовой разворот — не 1-барный эффект, и отскок bid-ask его не объясняет: отскок — эффект одного бара (ожидание H91, Q8; прямого теста здесь нет). "
                 "Часы суток здесь пересчитаны в UTC (сдвиг файла 4-5 часов до склейки).")
            + card("Часовой двойник", html_fig(f10, "fig10"))
            + card("Часовая регрессия по годам", t, "Перекрёстная проверка с recent/wf.py walk_forward на тех же входах: 78 общих месяцев, максимум расхождения 0,0."))
    return body


def sec10():
    f11 = rus_fig("fig11_forward_year", "В9: сегодня. Сигнал разворота с AUC 0,535-0,538 остаётся в 2025-11..2026-09, стоит 0,2-0,7 б.п. валовых на сделку топ-10%",
                  "Файл вперёд на Binance, только описательно: 11 оценённых месяцев, первый обучен на 30 днях. Сигнал тот же fade (ema20 и признаки положения), чуть лучше 2024 и 2025 (0,531 и 0,521 на старом файле) и без отскока<br>"
                  "(входы по закрытию и по след. открытию согласуются). Валовое среднее топ-10% не отличается от нуля по 95% интервалу. Данные: Binance BTCUSDT вперёд (другая площадка, не Bitstamp-история).", "fig11", "s10")
    ls = block("Q9 the present")
    i0 = [i for i, ln in enumerate(ls) if ln.startswith("monthly AUC")][0]
    hdr = ls[i0 + 1].split()
    rows = []
    for ln in ls[i0 + 2:i0 + 5]:
        p = ln.split()
        rows.append([{"close": "вход по закрытию", "nopen": "вход по след. открытию", "mid": "середина след. бара"}[p[0]]] + [ru(float(x), 3) for x in p[1:]])
    t = table(rows, ["вход"] + hdr)
    cond = table([
        ["Волатильность: низкая / средняя / высокая", "0,538 / 0,539 / 0,533", "-"],
        ["Объём: низкий / средний / высокий", "0,533 / 0,543 / 0,532", "-"],
        ["Размах high-low: низкий / средний / высокий", "0,529 / 0,542 / 0,554", "-"],
        ["Будни / выходные", "0,533 / 0,558", "выходные лучше и здесь (как в 2024-25 на старом файле)"],
        ["Самые большие последние ходы (кв. 97-100)", "0,623 (±0,067; 7 месяцев)", "широкий шум, описательно"],
    ], ["условие (вход по след. открытию)", "AUC", "комментарий"])
    body = (para("<b>Вопрос:</b> что осталось сегодня, на чистых данных Binance? <b>Ответ:</b> тот же разворот, AUC 0,535 (закрытие) и 0,538 (след. открытие) ± 0,010 по 11 месяцам, и входы по закрытию и по следующему открытию согласуются: отскока нет. "
                 "Валовое среднее топ-10%: +0,25 ± 1,09 б.п. (закрытие) и +0,65 ± 1,23 б.п. (след. открытие, 95% интервал по 11 месяцам), то есть от нуля не отличить. "
                 "Связь месячного AUC с прокси рынка на 11 месяцах незначима (все |r| < 0,4, p ≥ 0,25) — мало данных. Это не доказательство торговой пригодности: обучение с 2025-10, первый месяц на 30 днях.")
            + card("Binance, год вперёд", html_fig(f11, "fig11"))
            + card("AUC по месяцам", t, "Среднее: закрытие 0,5352 ± 0,0095; следующее открытие 0,5383 ± 0,0101; середина 0,5351 ± 0,0100; модель закрытия на метках следующего открытия 0,5360 ± 0,0091.")
            + card("По условиям (весь год вперёд)", cond))
    return body


def sec11(refs):
    body = (para("<b>Торговля: преимущества после издержек нет.</b> С 2022 валовое преимущество топ-10% при достижимом входе равно 0-1 б.п. внутри собственного шума и ниже любых реальных издержек; "
                 "на Binance сегодня 0,2-0,7 б.п., от нуля не отличается. Даже при нулевых издержках по D-044 оснований считать этот сигнал прибыльным нет. Часовой двойник держится лучше (AUC 0,54 в 2024-25), но это не торговый результат.")
            + para("<b>MVP: регрессия — честная база, а исправление происхождения данных — работа к выполнению.</b> Логистика на этих признаках — разумный базовый уровень для сравнения: она ловит один фактор, и простое правило даёт то же. "
                   "Но сам файл — Bitstamp BTC/USD со сдвигом часов, тиком $1 с 2022-08-10, склейкой 2024-09-13 и пропуском 2025-03-15. Нужно перевести время в UTC, исключить шовные минуты и считать эти три даты точками изменения данных. "
                   "Фолды (в том числе судейские по D-046), чьи блоки пересекают эти даты, подвержены смене источника; это пока вывод по аналогии (H91, раздел 5, п. 12), тест перелома Q6 с учётом склейки не выполнен. "
                   "Для бэклога MVP-лида; повторный прогон на истории Binance потребовал бы одобрения владельцем источника данных (D-050, H82), здесь он не предлагается.")
            + para("<b>Исследование: простое правило «против последних 30-60 минут» — честная база.</b> Любая новая модель направления, которая не лучше него (его преимущество — 90-100% от регрессии), не показывает ничего, кроме переупаковки одного и того же разворота. "
                   "Что осталось неизвестным — в списке ниже."))
    unk = ("<details><summary>Что осталось неизвестным (H91, раздел 5)</summary><div class='para'><ul class='refs'>"
           "<li>Прямого сопоставления свечей с Bitstamp нет (API недоступно); площадка определена косвенно.</li>"
           "<li>Сравнение с Binance тонкое: одно окно в 1000 минут в месяц (8-12 в год); относиться к годовым значениям как к ±0,03.</li>"
           "<li>Квинтильные беты — пулом, без стандартных ошибок; месячные беты с HAC — в планах.</li>"
           "<li>Механизм остаточного разворота сверх зазора (AUC на 1 бар 0,476-0,490 на 10-15 минутах, 2018-20) не определён; причина перелома 2018-03 неизвестна; атрибуция 2021-03 не доказана; перелом 2024-09 смешан со склейкой.</li>"
           "<li>Рост предсказуемости HAR может быть артефактом устаревших цен при тике $1; требует HAR на устойчивой к шуму RV и на истории Binance.</li>"
           "<li>Не проверены цитаты: «шум снизился со временем» у Hansen-Lunde, содержание Wen и др., падение объёма Binance на ≈90% после нулевых комиссий, доли деривативов CCData и FOW, «BTC наименее предсказуем» у Brauneis-Mestel.</li>"
           "<li>Недостаёт данных потока ордеров (знак сделок, taker-buy): они чисто разделили бы отскок, залежавшуюся печать и инвентарь, но D-050 исключает новые источники. Любая атрибуция по OHLCV предварительна.</li>"
           "</ul></div></details>")
    deviations = ("<details><summary>Отклонения T09 от H51/H59 и оговорки</summary><div class='para'><ul class='refs'>"
                  "<li>Чистка (purge) строже, чем в H51/H59, эффект 6e-7 AUC.</li>"
                  "<li>Прогнозы вперёд обучаются только на Binance с 2025-10, так что в первый месяц около 30 дней обучения.</li>"
                  "<li>Подмножества и частные зависимости выбраны задним числом (только описательно).</li>"
                  "<li>Состав условий Q6 использует фиксированные клетки 2019; вариант по размаху — в tables.txt.</li>"
                  "<li>Тяжёлые pickle-файлы лежат вне git, в D:/nt/nt_t09_cache.</li></ul></div></details>")
    return body + unk + deviations + refs


def refs_html():
    txt = H91.read_text(encoding="utf-8")
    sec = txt[txt.index("## 6. References"):]
    out = []
    drop = ("Scripts from this study", "provenance:", "first CPU tests:", "verification:")
    ru_h = {"Project and scripts": "Проект", "Market microstructure theory": "Теория микроструктуры рынка", "Noise, volatility and statistical methods": "Шум, волатильность и статистические методы",
            "Spread estimators": "Оценки спреда", "Crypto studies": "Работы по криптовалютам", "Exchange events and press": "События бирж и пресса"}
    for ln in sec.splitlines()[1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("**"):
            h = s.strip("*")
            if out:
                out.append("</ul>")
            out.append(f"<h4>{ru_h.get(h, h)}</h4><ul class='refs'>")
            continue
        if s.startswith("- "):
            s = s[2:]
        if any(k in s for k in drop) or s.startswith("`"):
            continue
        if s.startswith("Project evidence"):
            s = "Доказательства проекта: LOG.md, строки H51-H54, H59, H60, H72, H76, H82, H83; runs/tactical/decay/report.log; runs/tactical/research/2026-10-13-regression-signal/README.md"
        esc = H.escape(s)
        esc = re.sub(r"(https?://[^\s;]+?)(?=[.,]?(?:\s|;|$))", lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{m.group(1)}</a>', esc)
        esc = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", esc)
        out.append(f"<li>{esc}</li>")
    out.append("</ul>")
    return "<h3>Литература и источники (H91, раздел 6)</h3>" + "".join(out)


JS = """
(function(){
const q=new URLSearchParams(location.search).get('only');
if(q){document.body.classList.add('only');document.querySelectorAll('main > section').forEach(s=>{if(s.id!==q)s.style.display='none'});}
addEventListener('load',()=>setTimeout(()=>window.dispatchEvent(new Event('resize')),500));
const links=[...document.querySelectorAll('nav a')];
function onscroll(){let cur=null;document.querySelectorAll('main > section').forEach(s=>{if(s.getBoundingClientRect().top<140)cur=s.id});links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+cur))}
addEventListener('scroll',onscroll);
})();
"""


def build():
    w = bounce_window()
    secs = [("s0", "0. Итог", sec0()), ("s1", "1. Откуда данные", sec1()), ("s2", "2. Один фактор", sec2()), ("s3", "3. Форма и хвосты", sec3()),
            ("s4", "4. Разворот по лагу и горизонту", sec4()), ("s5", "5. Отскок bid-ask", sec5(w)), ("s6", "6. Когда это работает", sec6()),
            ("s7", "7. Почему затухло", sec7()), ("s8", "8. Экономика", sec8()), ("s9", "9. Часовой двойник", sec9()), ("s10", "10. Сегодня на Binance", sec10()),
            ("s11", "11. Что это значит для нас", sec11(refs_html()))]
    nav = "".join(f'<a href="#{s}">{t}</a>' for s, t, _ in secs)
    parts = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Что находит регрессия</title><style>" + css() + "</style><script>" + po.get_plotlyjs() + "</script></head><body>",
             f"<nav><h1>Что находит регрессия</h1><div class='sub'>neural_trade · тактическая сессия<br>T09 + литература H91 · 2026-10-14<br>данные до {C.CUTOFF[:10]}</div>{nav}</nav><main>"]
    for s, t, b in secs:
        parts.append(f'<section id="{s}"><h2>{t}</h2>{b}</section>')
    parts.append("<footer>Сгенерировано runs/tactical/report/build_regression_report.py. Источники: runs/tactical/tasks/T09_regression_anatomy (REPORT.md, tables.txt, figs/*.json), "
                 "runs/tactical/research/2026-10-13-regression-signal/README.md (H91), runs/tactical/report/regdata/lag_scan.csv; цены только до " + C.CUTOFF +
                 ". Исследовательская работа тактической сессии: одно зерно, разведочно, только dev-данные, без SPEC; не влияет на умолчания MVP (D-062).</footer></main>")
    parts.append("<script>" + JS + "</script></body></html>")
    out = HERE / "regression_report.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} {out.stat().st_size / 1e6:.2f} MB, {len(FIGS)} figures")


if __name__ == "__main__":
    build()
