"""Second export for the web dashboard (2026-10-13): the decay story, the decay map, the research verdict board, the geometry
forest plot and the drawdown/leverage frontier. All numbers come from committed result files (recut versions, cutoff 2025-07-20)
or, for the verdict board and the geometry plot, from the journal entries they cite (H-numbers in each row).
usage: python export2.py -> runs/tactical/webdash/out/{decay_monthly,regimes,decay_map,verdicts,geometry,frontier}.json"""
import json, os
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); OUT = "runs/tactical/webdash/out"; os.makedirs(OUT, exist_ok=True)
L = lambda p: json.load(open(p, encoding="utf-8"))


def dump(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(name, len(rows))


# 1. monthly decay: direction AUC at close entry and next-open entry, volatility predictability (HAR Spearman)
close = {r["month"]: r for r in L("runs/tactical/recut/lab/decay.json")}
nxt = {r["month"]: r for r in L("runs/tactical/recut/lab/decay_nextopen.json")}
vol = {r["month"]: r for r in L("runs/tactical/recut/lab/decay_vol.json")}
rows = []
for m in sorted(set(close) | set(nxt) | set(vol)):
    rows.append({"month": m, "auc_close": close.get(m, {}).get("auc"), "auc_nextopen": nxt.get(m, {}).get("auc"),
                 "hit10_nextopen": nxt.get(m, {}).get("top10_hit"), "har_spearman": vol.get(m, {}).get("har")})
dump("decay_monthly", rows)

# 2. regimes of the direction AUC (Bai-Perron style breaks, H72)
res = [r for r in L("runs/tactical/gaps/study1_results.json")["results"] if r["series"] == "S_auc" and not r["ex_july"]][0]
dump("regimes", [{"from": s["from"], "to": s["to"], "mean": s["mean"], "n": s["n"]} for s in res["segments"]])

# 3. decay map: bar size x horizon x year, mean monthly AUC (H52, recut)
q = L("runs/tactical/recut/decay/q1.json"); acc = defaultdict(list)
for key, v in q.items():
    bar, hz = key.split("|")[:2]
    for r in v["rows"]:
        acc[(bar, int(hz), r["month"][:4])].append(r["auc"])
dump("decay_map", [{"cell": f"{b} / {h}", "bar": b, "horizon": h, "year": y, "auc": sum(a) / len(a), "months": len(a)}
                   for (b, h, y), a in sorted(acc.items(), key=lambda kv: (["1m", "5m", "15m", "1h"].index(kv[0][0]), kv[0][1], kv[0][2]))])

# 4. research verdict board (H67-H80): effect vs the pre-registered threshold, in the study's own unit
V = [
    ("H67", "#1 Форма пути в своих часах", "Спирмен к базе, T=60 (Z2)", 0.0146, 0.02, "не прошла"),
    ("H68", "#4 Поворот перевеса: адаптация", "AUC к статичной модели", 0.0017, 0.005, "не прошла"),
    ("H69", "#2 Афтершоки Омори", "Скилл log-loss, T=30, %", 0.052, 1.0, "не прошла"),
    ("H70", "#6 VaR/ES FZ0, T=15", "Выигрыш FZ0 к FHS, %", 0.85, 1.0, "не прошла"),
    ("H71", "#5 TD-дисперсия (линейная)", "Выигрыш QLIKE на 60 д, %", 0.24, 1.0, "не прошла"),
    ("H74", "#5 TD-сеть, подтверждение", "Выигрыш к лучшей базе на 60 д, %", -11.1, 2.0, "не прошла"),
    ("H72", "Изменение волатильности на час", "Скилл к AR", 0.163, 0.02, "прошла"),
    ("H72", "Вол-оф-вол на 4 часа", "Скилл к AR", 0.011, 0.02, "не прошла"),
    ("H76", "#3 Стрела времени: затвор", "AUC верх-низ квинтиля", 0.011, 0.015, "не прошла"),
    ("H77", "#7 Физика на оси дисперсии", "Спирмен вкл-выкл", 0.0116, 0.010, "прошла"),
    ("H78", "Размер позиции по волатильности", "Calmar к фиксированному", 4.30 - 5.01, 0.0, "не прошла"),
    ("H79", "Стопы по прогнозу HAR", "Calmar к выходу по времени", 1.38 - 3.48, 0.0, "не прошла"),
    ("H80", "Синтетические индикаторы + геометрия", "AUC к случайным рядам", 0.0016, 0.005, "не прошла"),
]
dump("verdicts", [{"id": f"{h}-{i}", "h": h, "name": n, "unit": u, "effect": e, "threshold": t, "ratio": (e / t if t else (1.0 if e > 0 else -1.0)),
                   "verdict": v} for i, (h, n, u, e, t, v) in enumerate(V)])

# 5. geometry forest: direction AUC gain over the regression baseline (or the stated base), 95% CI, by experiment
G = [
    ("H24", "Геометрия обучаемых индикаторов в сети", 0.0141, -0.0020, 0.0303, "к базовой сети"),
    ("H43", "Геометрия + индикаторы от направления", 0.0122, -0.0080, 0.0325, "к базовой сети"),
    ("H45", "Геометрия 5 учебниковых + регрессия", -0.0008, None, None, "к регрессии"),
    ("H47", "164 признака геометрии, 1 год, C=0,001", 0.0024, -0.0012, 0.0060, "к регрессии"),
    ("H47", "164 признака геометрии, 3 года", 0.0011, -0.0021, 0.0043, "к регрессии"),
    ("H49", "N1 кривизна и кручение", -0.0031, -0.0057, -0.0004, "к регрессии"),
    ("H49", "N2 площади Леви", -0.0076, -0.0103, -0.0049, "к регрессии"),
    ("H49", "N3 фазовый портрет", -0.0021, -0.0048, 0.0007, "к регрессии"),
    ("H49", "N4 выпуклая оболочка", -0.0016, -0.0039, 0.0006, "к регрессии"),
    ("H49", "N5 Вассерштейн в окне", -0.0020, -0.0045, 0.0004, "к регрессии"),
    ("H49", "N6 путь Фишера-Рао", -0.0031, -0.0057, -0.0006, "к регрессии"),
    ("H49", "N7 выпуклость (контроль)", 0.0001, -0.0011, 0.0013, "к регрессии"),
    ("H80", "Синтетические ряды + геометрия (B)", -0.0135, None, None, "к регрессии"),
    ("H80", "Синтетические ряды + геометрия (A)", -0.0240, None, None, "к регрессии"),
]
dump("geometry", [{"id": f"{h}-{i}", "h": h, "name": n, "effect": e, "lo": lo, "hi": hi, "base": b} for i, (h, n, e, lo, hi, b) in enumerate(G)])

# 6. leverage frontier on the look period at 2 bps: cumulative return vs max drawdown by weight (H79)
st = L("runs/tactical/stops/results_look.json")["combos"]; fr = []
NAMES = {"a_1h_rich_l1_365": "Часовой сигнал (a)", "b_h62_4h_rich_logreg_730": "Кандидат H62 (b)"}
for combo, c in st.items():
    sig, ver = combo.split("|")
    for rule in ("X0_time",) + (("X6_best_plus_filter",) if "X6_best_plus_filter" in c["rules"] else ()):
        lev = c["rules"][rule]["levels"].get("2")
        if not lev or "frontier" not in lev:
            continue
        for p in lev["frontier"]["rows"]:
            fr.append({"series": f"{NAMES.get(sig, sig)}, {'порог прошлого месяца' if ver == 'prev' else 'порог того же месяца'}, "
                                 f"{'выход по времени' if rule == 'X0_time' else 'стопы HAR + фильтр'}",
                       "signal": sig, "version": ver, "rule": rule, "w": p["w"], "cum": p["cum_pct"], "dd": p["dd_pct"]})
dump("frontier", fr)
