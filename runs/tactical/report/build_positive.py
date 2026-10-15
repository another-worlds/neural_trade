"""Builds runs/tactical/report/positive_outlook.html: the POSITIVE OUTLOOK of the volatility session (branch nt-tactical) and the parallel outlook
session (branch nt-tactical-outlook) in ONE self-contained animated page (no CDN, no plotly: the animations are SVG drawn from the data of the reused figures).
Owner, 2026-10-15: show only what worked and where the breakthroughs are, energetic tone, animations, geometry. Every number is the value in the
committed result files (the file is named next to it in the page's data-src attributes and in the builder's SOURCES table); nothing is rounded up.

Inputs (all read-only):
  * this worktree (branch nt-positive, from origin/nt-tactical): runs/tactical/{LOG.md, tasks/T0*..T12, lab/decay_vol.json, report/vol_windows.json}
  * the outlook branch, a read-only worktree, passed as --outlook (default D:/nt/nt_outlook_ro): runs/tactical/outlook/O*/ (results, frozen shas, interp.json)
  * price: the Binance forward file D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv (hero, barrier day). The cut file is not needed.
    (Bitcoin_BTCUSDT.csv of the MVP is never touched.)
usage: python runs/tactical/report/build_positive.py [--outlook D:/nt/nt_outlook_ro]   -> positive_outlook.html (CPU, seconds)
Screenshots and the console check: python runs/tactical/report/shoot_positive.py"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rcommon import T, TAC  # noqa: E402  (theme, tactical folder)

ap = argparse.ArgumentParser()
ap.add_argument("--outlook", default="D:/nt/nt_outlook_ro")
ap.add_argument("--fwd", default="D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv")
args = ap.parse_args()
OUTL = Path(args.outlook) / "runs" / "tactical"
ASSETS = HERE / "positive_assets"

# colour roles (theme.py): horizon colours are not used here; up = violet, down = gold (candle roles)
UPC, DNC = T.UP_COLOR, T.DOWN_COLOR
OS = T.OTHER_SERIES
GOLD, PINK, TAN, VIOLET, SALMON = OS[0], OS[1], OS[2], OS[3], OS[4]
PALETTE = dict(paper=T.PAPER, surface=T.SURFACE, ink=T.INK, ink2=T.INK_2, muted=T.MUTED, grid=T.GRID, axis=T.AXIS, neutral=T.NEUTRAL,
               up=UPC, down=DNC, gold=GOLD, pink=PINK, tan=TAN, violet=VIOLET, salmon=SALMON, good=T.GOOD)


def jf(p):
    return json.load(open(p, encoding="utf-8"))


def dec(v):
    """plotly JSON typed array {'dtype','bdata'} -> list"""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]).astype(float).tolist()
    return v


SRC = {}   # tag -> source file (relative), listed in the page's data table and checked by the builder


# ----------------------------------------------------------------------------------------------- price data (Binance forward file)
def load_fwd():
    d = pd.read_csv(args.fwd, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    d = d[(d.timestamp >= "2025-11-10") & (d.timestamp < "2025-11-17")].set_index("timestamp")
    assert d.index.is_monotonic_increasing and len(d) == 7 * 1440, len(d)
    return d


def ema_first(x, p):
    """geomlib.ema: alpha = 2/(p+1), initialised with the first value"""
    a = 2.0 / (p + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def hero_data(fw):
    day0 = pd.Timestamp("2025-11-15")
    c = fw["close"].to_numpy(float)
    e20 = ema_first(c, 20)
    e60 = ema_first(c, 60)
    i0 = int((day0 - fw.index[0]) / pd.Timedelta(minutes=1))
    n = 1440
    # 15-minute closes (the minute 14, 29, ... of each quarter-hour), EMA20 / EMA40 warmed up from 2025-11-10
    q = np.arange(14, len(c), 15)
    c15 = c[q]
    E20, E40 = ema_first(c15, 20), ema_first(c15, 40)
    j0 = i0 // 15
    nodes = []
    for j in range(j0, j0 + 96):
        nodes.append({"m": int(q[j] - i0), "e20": float(E20[j]), "e40": float(E40[j]),
                      "d3": float(E20[j] - E20[j - 3]), "d2": float((E20[j] - E20[j - 3]) - (E20[j - 3] - E20[j - 6]))})
    return {"date": "2025-11-15", "c": np.round(c[i0:i0 + n], 2).tolist(), "e20": np.round(e20[i0:i0 + n], 2).tolist(),
            "e60": np.round(e60[i0:i0 + n], 2).tolist(), "nodes": nodes}


def barrier_day(fw):
    """T10's one real day (2025-11-15, chosen by the median-hit-count rule): close, 24 anchors with the +-50 bp box, H3's P(hit) and the outcome"""
    f = jf(TAC / "tasks/T10_barrier_hit/figs/btc_window.json")
    SRC["barrier_window"] = "tasks/T10_barrier_hit/figs/btc_window.json"
    H3 = [t for t in f["data"] if t.get("name") == "H3"][0]
    P = dec(H3["y"])
    shapes = f["layout"]["shapes"]
    assert len(shapes) == 24 and len(P) == 24
    day0 = pd.Timestamp("2025-11-15")
    i0 = int((day0 - fw.index[0]) / pd.Timedelta(minutes=1))
    c, h, l = fw["close"].to_numpy(float), fw["high"].to_numpy(float), fw["low"].to_numpy(float)
    close = c[i0:i0 + 1500]
    boxes = []
    for k, s in enumerate(shapes):
        t = pd.Timestamp(s["x0"])
        a = int((t - fw.index[0]) / pd.Timedelta(minutes=1))
        c0 = c[a - 1]
        assert abs(s["y0"] - c0 * np.exp(-0.005)) < 1e-6 * c0 and abs(s["y1"] - c0 * np.exp(0.005)) < 1e-6 * c0
        hi, lo = c0 * np.exp(0.005), c0 * np.exp(-0.005)
        mm = np.maximum(np.log(h[a:a + 60] / c0), np.log(c0 / l[a:a + 60]))
        label = bool(mm.max() >= 0.005)
        first = int(np.argmax(mm >= 0.005)) if label else -1          # minute inside the window of the first touch
        side = 0
        if label:
            side = 1 if np.log(h[a + first] / c0) >= 0.005 else -1       # the up (+1) or down (-1) bound touched first (up if both)
        # the figure's outcome flag: orange fill = hit (SERIES[1])
        hit_fig = s["fillcolor"].lower() == T.SERIES[1].lower()
        assert hit_fig == label, (k, hit_fig, label)
        boxes.append({"m": a - i0, "c0": float(c0), "lo": float(lo), "hi": float(hi), "p": float(P[k]), "hit": label, "first": first, "side": side})
    return {"date": "2025-11-15", "c": np.round(close, 2).tolist(), "boxes": boxes,
            "n_hit": int(sum(b["hit"] for b in boxes)), "mean_p_hit": float(np.mean([b["p"] for b in boxes if b["hit"]])),
            "mean_p_miss": float(np.mean([b["p"] for b in boxes if not b["hit"]]))}


# ----------------------------------------------------------------------------------------------- the other data blocks
def clock_data():
    s = jf(TAC / "tasks/T11_utc_seasonality/seasonality.json")
    SRC["clock"] = "tasks/T11_utc_seasonality/seasonality.json"
    out = {}
    for key, nm in (("before 2021-11", "before"), ("after 2021-11", "after"), ("Binance 2025-10..2026-09", "binance")):
        out[nm] = {"slot96": s[key]["slot96"]["rel"], "hour": s[key]["utc_hour"]["rel"], "n_days": s[key]["utc_hour"]["n_days"]}
    return out


def har_days():
    v = jf(TAC / "report/vol_windows.json")
    SRC["har_days"] = "report/vol_windows.json (made by report/vol_windows.py from T03's frozen fit)"
    out = []
    for d in v["days"]:
        if d["date"] not in ("2025-12-29", "2026-06-04"):
            continue
        base = pd.Timestamp(d["t"][0])
        ts = [(pd.Timestamp(x) - base) / pd.Timedelta(minutes=1) for x in d["t"]]
        out.append({"date": d["date"], "label": {"2025-12-29": "Обычный день", "2026-06-04": "Штормовой день"}[d["date"]],
                    "sd": d["sd_bps_1m"], "w_har": d["width_bps"]["HAR"], "w_trail": d["width_bps"]["trail"], "n": d["n_windows"],
                    "c": d["close"], "m": [int(t) for t in ts],
                    "win": [{"m": int((pd.Timestamp(w["t0"]) - base) / pd.Timedelta(minutes=1)), "c0": w["c0"], "end": w["end"],
                             "hlo": w["HAR"]["lo"], "hhi": w["HAR"]["hi"], "tlo": w["trail"]["lo"], "thi": w["trail"]["hi"]} for w in d["windows"]]})
    assert len(out) == 2
    return out


def geom_windows():
    ip = jf(OUTL / "outlook/O1_geom_vol/interp.json")
    SRC["geom_windows"] = "outlook/O1_geom_vol/interp.json (real 60-bar 15-minute windows, selection span)"
    res = []
    for ex in ip["examples"]:
        w = max(ex["windows"], key=lambda z: abs(z["value"]))
        c = np.array([r[3] for r in w["ohlcv"]], float)
        res.append({"feature": ex["feature"], "coef": ex["coef"], "value": w["value"], "anchor": w["anchor_time"], "c": c.tolist(),
                    "e20": ema_first(c, 20).tolist(), "e40": ema_first(c, 40).tolist()})
    return res


def o5_months():
    R = [json.loads(x) for x in open(OUTL / "outlook/O5_geom_bestbase/results.jsonl")]
    SRC["o5_months"] = "outlook/O5_geom_bestbase/results.jsonl"
    out = {}
    for H in (15, 60):
        sp = {(r["month"], r["model"]): r["spearman"] for r in R if r["span"] == "judge" and r["H"] == H}
        months = sorted({m for m, _ in sp})
        diff = [sp[(m, "L1")] - sp[(m, "L0")] for m in months]
        out[H] = {"months": months, "diff": diff}
    assert len(out[15]["diff"]) == 30 and abs(np.mean(out[15]["diff"]) - 0.0122) < 5e-5 and min(out[15]["diff"]) > 0
    assert abs(np.mean(out[60]["diff"]) - 0.0056) < 5e-5 and sum(d > 0 for d in out[60]["diff"]) == 23
    return out


def o11a_months():
    R = [json.loads(x) for x in open(OUTL / "outlook/O11_accuracy_batch/part_a/results.jsonl")]
    SRC["o11a"] = "outlook/O11_accuracy_batch/part_a/results.jsonl"
    sk = {(r["month"], r["model"]): r["skill"] for r in R if r["span"] in ("B", "C") and r["tg"] == "A"}
    months = sorted({m for m, _ in sk})
    diff = [sk[(m, "cand")] - sk[(m, "base")] for m in months]
    assert len(diff) == 25 and abs(np.mean(diff) - 0.0447) < 5e-5 and min(diff) > 0, (len(diff), np.mean(diff))
    return {"months": months, "diff": diff}


def t01_months():
    d = jf(TAC / "tasks/T01_vol_forward/results.json")
    SRC["t01"] = "tasks/T01_vol_forward/results.json"
    rows = [r for r in d["rows"] if r["arm"] == "F" and r["target"] == "A" and r["model"] == "S3"]
    assert len(rows) == 12 and abs(np.mean([r["skill_ar"] for r in rows]) - 0.1419) < 5e-5
    return {"months": [r["month"] for r in rows], "v": [r["skill_ar"] for r in rows]}


def decay_years():
    d = jf(TAC / "lab/decay_vol.json")
    SRC["decay_vol"] = "lab/decay_vol.json"
    ys = {}
    for r in d:
        ys.setdefault(r["month"][:4], []).append(r["har"])
    out = {y: float(np.mean(v)) for y, v in ys.items() if "2018" <= y <= "2025"}
    assert abs(out["2018"] - 0.286) < 5e-4 and abs(out["2023"] - 0.411) < 5e-4
    return out


def decile_data():
    f = jf(TAC / "tasks/T09_regression_anatomy/figs/fig3b_tail_edge.json")
    SRC["deciles"] = "tasks/T09_regression_anatomy/figs/fig3b_tail_edge.json (panel bb20)"
    ann = [a["text"] for a in f["layout"]["annotations"]]
    k = ann.index("bb20")
    tr = [t for t in f["data"] if t.get("xaxis") == ("x" if k == 0 else f"x{k + 1}")]
    assert len(tr) == 3
    return {t["name"]: dec(t["y"]) for t in tr}


def brier_data():
    """T10's Brier-skill-by-month figure (its own JSON), the primary cell (X = 50 bps, T = 60 min), the chosen model H3: months of Look 1 and Look 2"""
    f = jf(TAC / "tasks/T10_barrier_hit/figs/brier_skill_by_month.json")
    SRC["brier"] = "tasks/T10_barrier_hit/figs/brier_skill_by_month.json (panel X = 50 bps, T = 60 min, trace H3)"
    ann = [a["text"] for a in f["layout"]["annotations"]]
    k = ann.index("X = 50 bps, T = 60 min")
    ax = "x" if k == 0 else f"x{k + 1}"
    t = [t for t in f["data"] if t.get("xaxis") == ax and t.get("name") == "H3 (chosen)"][0]
    y = dec(t["y"])
    v = [None if q is None else float(q) for q in y]
    assert len(v) == 25 and min(q for q in v if q is not None) > 0
    return {"months": t["x"], "v": v}


def timeline_data():
    """frozen prospective bets: sha256 prefix as written in each study's own report (or its .sha256 file)"""
    rows = [
        ("T04-B", "15-минутный календарь для прогноза изменения волатильности", "61749ad6", "tasks/T04_vol_calendar/REPORT.md", "2026-10-01", "2027-01-01",
         "3 полных месяца с 2026-10"),
        ("T05-B", "Интервалы 60 мин: HAR, EWMA и квантильная регрессия", "e9c38d61", "tasks/T05_interval_models/frozen_B.sha256", "2026-10-01", "2027-01-01",
         "3 полных месяца с 2026-10"),
        ("T11", "UTC-календарь V3 (96 слотов): V3 - V0 не меньше +0,01", "6d62faaf", "tasks/T11_utc_seasonality/REPORT.md", "2026-10-02", "2027-01-01",
         "оценка по 3 и по 6 месяцам (Pocock)"),
        ("T12", "Барьерная вероятность H3 + изотоника месяца", "382555ce", "tasks/T12_barrier_calibrated/frozen_B.sha256", "2026-10-02", "2027-01-01",
         "3 полных месяца с 2026-10-02"),
        ("O4", "BANK: плотный HAR против базы, H = 60", "83dd3ad7", "outlook/O4_bank_confirm/frozen_B.sha256", "2026-10-01", "2027-01-01",
         "3 полных месяца с 2026-10"),
        ("O5", "Геометрия L1 над лучшей базой L0, H = 60", "79f7f25d", "outlook/O5_geom_bestbase/frozen_B.sha256", "2026-10-01", "2027-01-01",
         "3 из 3 месяцев, прирост не меньше +0,003"),
        ("O7", "Интервалы с геометрией на заморозке O5, калибровка 2026-10", "79f7f25d", "outlook/O7_geom_intervals/REPORT.md (на заморозке O5)", "2026-11-01", "2027-02-01",
         "оценка 2026-11..2027-01"),
        ("O9", "Кандидат ROBUST: пробой + регрессионный гейт", "9c5259a6", "outlook/O9_ensemble_strategy/frozen_P.sha256", "2026-11-01", "2027-02-01",
         "оценка 2026-11..2027-01, пессимистичные заливки"),
        ("O10", "Выходы по геометрии и волатильности", "3f87ef09", "outlook/O10_geomvol_exits/frozen_P.sha256", "2026-11-01", "2027-02-01",
         "оценка 2026-11..2027-01"),
        ("O11a", "Геометрия + банк для изменения волатильности", "e2c7ac48", "outlook/O11_accuracy_batch/part_a/frozen_P_a.sha256", "2026-11-01", "2027-02-01",
         "прирост не меньше +0,010, 3 из 3"),
        ("O11b", "Помесячное переобучение для уровня волатильности", "7219a428", "outlook/O11_accuracy_batch/part_b/frozen_P_b.sha256", "2026-11-01", "2027-02-01",
         "прирост не меньше +0,003, 3 из 3"),
        ("T13", "Линии поддержки и сопротивления: A1, B1, C1", "8fec9bf3", "tasks/T13_support_lines/frozen_B.sha256", "2026-10-02", "2027-01-01",
         "критерии по частям, 3 полных месяца с 2026-10-02"),
    ]
    out = []
    for tid, what, sha, src, a, b, crit in rows:
        # verify the prefix against the file named (a .sha256 file, or the report text)
        p = (TAC if not src.startswith("outlook/") else OUTL) / src.split(" ")[0]
        txt = open(p, encoding="utf-8").read()
        if tid == "O7":
            assert "79f7f25d" in open(OUTL / "outlook/O5_geom_bestbase/REPORT.md", encoding="utf-8").read()
        elif tid == "T04-B":
            assert sha in txt
        else:
            assert txt.strip().startswith(sha) or sha in txt, (tid, sha)
        out.append({"id": tid, "what": what, "sha": sha, "src": src, "a": a, "b": b, "crit": crit})
    SRC["timeline"] = "the .sha256 / REPORT files named in each row"
    return out


# ----------------------------------------------------------------------------------------------- T13: support and resistance lines
T13 = TAC / "tasks/T13_support_lines"


def rus(v, d=4, sign=True):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("\u2212" if v < 0 else ("+" if sign else "")) + s


def ci_txt(m, lo, hi, d=4):
    return f"{rus(m, d)} [{rus(lo, d)}; {rus(hi, d)}]"


def lines_data():
    """windows.json (three real Binance windows drawn from make_windows.py, rule in the file) and the T13 results (results.json), numbers as measured"""
    w = jf(T13 / "windows.json")
    r = jf(T13 / "results.json")
    SRC["lines_windows"] = "tasks/T13_support_lines/windows.json (make_windows.py: the rule is in the file)"
    SRC["lines_results"] = "tasks/T13_support_lines/results.json, tables.txt"
    assert len(w["windows"]) == 3 and all(len(x["bars"]["c"]) == 240 and len(x["fut"]["c"]) == 60 for x in w["windows"])
    return {"windows": w["windows"], "rule": w["rule"], "pair": w["pair"], "n_anchors": w["n_anchors"], "model": w["model"]}, r


def lines_results_html(r):
    """The measured T13 numbers as the page's cards. A part that passed (both looks, Holm) is shown prominently; a part that did not is shown with its
    measured POSITIVE effect sizes only (the negative ones are in tables.txt), never rounded up."""
    nm = {"A": "A. Вероятность касания ±50 б.п. за 60 минут", "B": "B. Размер следующего часа (Спирмен)", "C": "C. Изменение волатильности следующего часа"}
    pairtxt = lambda k: {"1": "1-минутные", "15": "15-минутные"}[k.split("x")[0]] + " бары, " + k.split("x")[1] + " баров назад"
    eff = {}
    for p in "ABC":
        e = []
        for look, lname in (("J1", "J1 Bitstamp 2017-10..2018-12" if p != "B" else "J1 Bitstamp 2018-07..2018-12"), ("J2", "J2 Binance 2025-11..2026-09")):
            if p == "A":
                s = r["A"][look]["summary"]["A1_vs_H3"]
                e.append((lname, s["rel"] * 100, s["ci"][0] * 100, s["ci"][1] * 100, "%", f"{s['rel'] * 100:+.2f}", None))
            else:
                d = r[p][look]
                e.append((lname, d["mean"], d["ci"][0], d["ci"][1], "", None, f"{d['pos']} из {d['n']} месяцев"))
        eff[p] = e
    cards = []
    for p in "CBA":
        c = r[p]["criterion"]; passed = bool(c["PASS"])
        if p == "A":
            what = "прирост Брайера A1 против H3, % (порог +1% с интервалом выше 0; A1 должна обыграть и перемешанный контроль AS)"
        elif p == "B":
            what = "прирост Спирмена B1 − L1 по месяцам (порог +0,003 с интервалом выше 0)"
        else:
            what = "прирост скилла C1 − V3 над AR по месяцам (порог +0,005 с интервалом выше 0)"
        if passed:
            kp = []
            for (lname, m, lo, hi, unit, _, note) in eff[p]:
                dd = 2 if unit == "%" else 4
                kp.append(f'<div class="kpi"><div class="kv">{rus(m, dd)}{unit}</div><div class="kl">{lname}: [{rus(lo, dd)}; {rus(hi, dd)}]' + (f", {note}" if note else "") + "</div></div>")
            cards.append(f'<div class="card cand" id="t13{p.lower()}"><div class="tagrow"><span class="chip">часть {p} · критерий пройден на обоих периодах · Holm p = {rus(c["holm_p"], 4, False)}</span>'
                         f'<span class="chip dim">выбрано в отборе 2019-01..2024-06: {pairtxt(r[p]["pair"])}</span></div><h3>{nm[p]}: линии добавляют к базе</h3>'
                         f'<div class="kpis">{"".join(kp)}</div><p class="note">{what}. Это второй взгляд на периоды, которые базы уже видели: пара (размер бара, окно) выбрана до взгляда, на отборе, и на J1 и J2 ничего не подбиралось.</p></div>')
        else:
            pos = [x for x in eff[p] if x[1] > 0]
            if pos:
                lns = "".join(f"<li>{x[0]}: <b>{rus(x[1], 2 if x[4] else 4)}{x[4]}</b> [{rus(x[2], 2 if x[4] else 4)}; {rus(x[3], 2 if x[4] else 4)}]" + (f", {x[6]}" if x[6] else "") + "</li>" for x in pos)
            else:
                lns = "<li>положительного эффекта на этих периодах нет</li>"
            cards.append(f'<div class="card" id="t13{p.lower()}"><div class="tagrow"><span class="chip dim">часть {p} · критерий не пройден</span>'
                         f'<span class="chip dim">выбрано в отборе: {pairtxt(r[p]["pair"])}</span></div><h3>{nm[p]}</h3><ul class="pts">{lns}</ul>'
                         f'<p class="note">{what}. Порог не достигнут, поэтому здесь показаны только измеренные положительные эффекты; все числа — в tasks/T13_support_lines/tables.txt.</p></div>')
    sha = jf(T13 / "frozen_B.json")["sha256"][:8]
    assert open(T13 / "frozen_B.sha256", encoding="utf-8").read().strip().startswith(sha)
    SRC["lines_freeze"] = "tasks/T13_support_lines/frozen_B.sha256"
    intro = ("<h3 style=\"margin-top:18px\">Что измерено (T13, критерии записаны до расчёта)</h3>"
             "<p class=\"note\">J1 — Bitstamp 2017–18, J2 — Binance 2025-11..2026-09. Одна пара (размер бара, окно) на часть выбрана на 2019-01..2024-06, затем записана в selection.json и закоммичена до J1 и J2. "
             "Критерий проходит только если выполнен на ОБОИХ периодах; Holm по трём частям.</p>")
    outro = (f'<p class="note">Проспективная проверка: A1, B1 и C1 заморожены на Binance до 2026-09-30, sha256 {sha}…; оценка по трём полным месяцам с 2026-10 в <code>score_B.py</code>.</p>'
             '<p class="src">Файлы: tasks/T13_support_lines/results.json, tables.txt, selection.json, frozen_B.sha256, windows.json.</p>')
    return intro + '<div class="cards">' + "".join(cards) + "</div>" + outro


def main():
    fw = load_fwd()
    LN, LR = lines_data()
    D = {"hero": hero_data(fw), "barrier": barrier_day(fw), "clock": clock_data(), "har": har_days(), "geom": geom_windows(), "o5": o5_months(),
         "o11a": o11a_months(), "t01": t01_months(), "decay": decay_years(), "dec": decile_data(), "timeline": timeline_data(), "brier": brier_data(), "lines": LN, "pal": PALETTE}
    data_js = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    css = (ASSETS / "pos.css").read_text(encoding="utf-8")
    js = (ASSETS / "pos.js").read_text(encoding="utf-8")
    body = (ASSETS / "body.html").read_text(encoding="utf-8").replace("{{T13_RESULTS}}", lines_results_html(LR))
    for k, v in PALETTE.items():
        css = css.replace("{{" + k + "}}", v)
    css = css.replace("{{font}}", T.FONT)
    sha = "n/a"
    html = ("<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Позитивный аутлук: волатильность, геометрия, вероятность</title><style>" + css + "</style></head><body>"
            + body + "<script id=\"data\" type=\"application/json\">" + data_js + "</script><script>" + js + "</script></body></html>")
    out = HERE / "positive_outlook.html"
    out.write_text(html, encoding="utf-8")
    print("written", out, f"{out.stat().st_size / 1e6:.2f} MB; sources:", json.dumps(SRC, ensure_ascii=False))


if __name__ == "__main__":
    main()
