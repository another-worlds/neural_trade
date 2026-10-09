"""Export the tactical session's results as JSON row files for the web dashboard (claude.ai artifact "Neural Trade Tactical").
usage: python export.py  -> runs/tactical/webdash/out/*.json (journal, models, longtrain, hourly_dev, hourly_final, tasks)"""
import glob, json, os
os.chdir(r"D:\nt\nt_tactical"); OUT = "runs/tactical/webdash/out"; os.makedirs(OUT, exist_ok=True)


def dump(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w", encoding="utf-8"), ensure_ascii=False)


# one line per journal entry: area, what was tested, verdict (effect / partial / none / failed / info), the key number
J = [
    ("H5", "2026-10-06", "Сеть", "Ограничить голову направления от тяжёлой части сети", "none", "эффекта нет"),
    ("H6", "2026-10-07", "Сеть", "Перебор настроек сети, раунды R2/R3", "none", "AUC 0,52-0,55"),
    ("H7", "2026-10-07", "Сеть", "Почему все AUC высокие на срезе 19.04.2022", "info", "режим рынка, не навык"),
    ("H8", "2026-10-07", "Сеть", "Топ-10 высоких AUC на 7-дневном блоке", "none", "не подтвердились"),
    ("H9", "2026-10-07", "Сеть", "Кандидат с AUC 0,805", "info", "только этот срез"),
    ("H10", "2026-10-07", "Скорость", "1 против 3 процессов, батч 256 против 1024", "info", "замер скорости"),
    ("H11", "2026-10-07", "Сеть", "Голова уверенности на 1 дне", "none", "нужно больше данных"),
    ("H12", "2026-10-07", "Сеть", "Перебор на длинном блоке, промежуточно", "none", "без выигрыша"),
    ("H13", "2026-10-08", "Сеть", "Перебор на длинном блоке, финал", "none", "без выигрыша"),
    ("H14", "2026-10-08", "Сеть", "Как связаны 9 выходов", "info", "голова цены - шум"),
    ("H15", "2026-10-08", "Сеть", "Сочетания голов и пороги", "none", "заметного выигрыша нет"),
    ("H16", "2026-10-08", "Сеть", "Какие потери тянут сеть", "info", "направлению ~10% градиента"),
    ("H17", "2026-10-08", "Данные", "Статичная выдержка большого файла", "info", "побитно совпадает"),
    ("H18", "2026-10-08", "Скорость", "Окна только из нужного хвоста", "effect", "в 5 раз быстрее"),
    ("H19", "2026-10-08", "Сеть", "Убрать голову цены; 3 горизонта против 1", "partial", "CRPSS +0,024 (на грани)"),
    ("H20", "2026-10-08", "Сеть", "Честные пороги ансамбля", "partial", "топ-10% 59,6%"),
    ("H21", "2026-10-08", "Индикаторы", "Индикаторы на крошечных прогонах", "none", "без индикаторов лучше"),
    ("H22", "2026-10-08", "Индикаторы", "Победители H21 на длинном блоке", "none", "преимущество исчезло"),
    ("H23", "2026-10-08", "Регрессия", "Прямой перебор периодов + логрегрессия", "effect", "0,570 против 0,521"),
    ("H24", "2026-10-09", "Сеть", "Гипотезы 10-13 (индикаторы от направления, без hd, ворота, геометрия)", "partial", "только калибровка"),
    ("H25", "2026-10-09", "Индикаторы", "G: линейный выход из обучаемых индикаторов", "none", "обучаемые = учебниковые"),
    ("H26", "2026-10-09", "Индикаторы", "Решающий раунд: без, заморожены, тонкое считывание, сети по функциям", "none", "индикаторы ничего не добавляют"),
    ("H27", "2026-10-09", "Регрессия", "Лаборатория логрегрессии", "effect", "+0,036 AUC к сети"),
    ("H28", "2026-10-09", "Пересборка", "Лестница L0-L7 с нуля", "none", "никто не обогнал регрессию"),
    ("H29", "2026-10-09", "Данные", "Поиск суточной версии 70%; крупные бары", "none", "версии нет; 1 д AUC 0,525"),
    ("H30", "2026-10-09", "Сеть", "Находка перебора на отложенных срезах", "failed", "59,6% -> 54,3%"),
    ("H31", "2026-10-09", "Сеть", "Сеть только цены", "none", "хуже прогноза ноль"),
    ("H32", "2026-10-09", "Сеть", "E: индикаторы в конце, форма", "none", "AUC +0,001"),
    ("H33", "2026-10-09", "Торговля", "Тройной барьер + мета-модель", "effect", "1 ч: +7..+19 б.п."),
    ("H34", "2026-10-09", "JEPA", "Минимальный JEPA", "partial", "волатильность +0,053"),
    ("H35", "2026-10-09", "Пересборка", "Лестница при обучении до 3 лет", "none", "картина не меняется"),
    ("H36", "2026-10-09", "Пересборка", "Новая сеть: регрессия + PatchTST-lite", "none", "-0,0002 к регрессии"),
    ("H37", "2026-10-09", "Индикаторы", "Хвост очереди: E путь+форма, MFI, стартовые параметры", "none", "на направление не влияет"),
    ("H38", "2026-10-09", "Торговля", "Часовой тройной барьер: 97 конфигураций + отложенный период", "failed", "+9,8 б.п., интервал по месяцам через 0"),
]
dump("journal", [dict(zip(("id", "date", "area", "test", "verdict", "key"), j)) for j in J])

# direction AUC by model (24 lab slices, mean of 3 horizons) from the ladder, long-train and new-network records
import math
NAMES = {"L0": "Регрессия sklearn", "L1": "Регрессия Keras", "L3": "3 горизонта сразу", "L5": "43 признака", "L6": "+ нелинейная сеть",
         "L7": "+ GRU по окну"}


def ci(v):
    m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5; h = 2.07 * sd / math.sqrt(len(v))
    return m, m - h, m + h


models = []
for l in open("runs/tactical/rebuild/results.jsonl", encoding="utf-8"):
    r = json.loads(l); ps = r.get("per_slice", {})
    if r.get("step") and "auc_h0" in ps:
        a3 = [sum(x) / 3 for x in zip(ps["auc_h0"], ps["auc_h1"], ps["auc_h2"])]; m, lo, hi = ci(a3)
        models.append({"model": NAMES.get(r["step"], r["step"]), "step": r["step"], "auc": m, "lo": lo, "hi": hi, "family": "Пересборка, 11,5 дня"})
for l in open("runs/tactical/newnet/results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["n_slices"] == 24:
        a = r["summary"]["auc3"]; models.append({"model": f"новая сеть {r['arch']} (1 г)", "auc": a[0], "lo": a[1], "hi": a[2], "family": "новая сеть"})
dump("models", models)

lt = []
for l in open("runs/tactical/rebuild/longtrain_results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if "auc3" in r and r.get("step") in ("L0", "L5", "L6", "L7"):
        lt.append({"span": r.get("span"), "step": r["step"], "slice": r.get("slice"), "auc3": r["auc3"]})
agg = {}
for r in lt:
    agg.setdefault((r["span"], r["step"]), []).append(r["auc3"])
dump("longtrain", [{"span": k[0], "step": k[1], "auc": sum(v) / len(v), "n": len(v)} for k, v in agg.items()])

dev = []
for l in open("runs/tactical/hourly/results.jsonl", encoding="utf-8"):
    r = json.loads(l); s = r["summary"]; c = r["config"]
    if "meta_bps" in s:
        dev.append({"tag": r["tag"], "T": c["T"], "tp": c["tp"], "sl": c["sl"], "feat": c["feat"], "primary": c["primary"],
                    "target": c["target"], "meta": c["meta"], "bps": s["meta_bps"][0], "bps_lo": s["meta_bps"][1], "bps_hi": s["meta_bps"][2],
                    "null95": s["meta_null95"][0], "excess": s["meta_excess"][0], "hit": s["meta_hit"][0], "tpd": s["meta_trades_per_day"][0]})
dump("hourly_dev", dev)
fin = []
for i, l in enumerate(open("runs/tactical/hourly/final.jsonl", encoding="utf-8")):
    r = json.loads(l)
    for k in ("meta", "primary", "all"):
        v = r["final"].get(k)
        if v:
            fin.append({"rank": i + 1, "tag": r["tag"], "selection": k, "n": v["n"], "hit": v["hit"], "bps": v["bps"], "null95": v["null95"],
                        "ci_lo": v["monthly_bps_ci"][0], "ci_hi": v["monthly_bps_ci"][1], "net10": v["net10_bps"]})
dump("hourly_final", fin)
dump("tasks", [{k: t.get(k, "") for k in ("task", "who", "status", "since", "next")} for t in json.load(open("runs/tactical/tasks.json", encoding="utf-8"))])
print({f: len(json.load(open(f"{OUT}/{f}", encoding="utf-8"))) for f in os.listdir(OUT)})
