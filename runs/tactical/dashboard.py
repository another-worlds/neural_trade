"""Live dashboard of the tactical session (v4: one tab per experiment; the running one opens by default).
Owner (2026-10-07): every new run gets its own tab, so progress and status are visible without asking.
To add a tab, append an entry to EXPERIMENTS (the lead does this whenever it launches a run).
Writes runs/tactical/dashboard.html (data inlined, plotly from CDN, reloads every 20 s, keeps the chosen tab).
    python runs/tactical/dashboard.py            # write once
    python runs/tactical/dashboard.py --loop 20  # rewrite every 20 s
"""
import re, collections, datetime, glob, json, math, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "dashboard.html")
sys.path.insert(0, ROOT)
from hc2_compare import T
import hc4_metric

H = ("h0", "h1", "h2")
HM = [("delta", "corr", "Цена: корреляция"), ("delta", "skill_vs_zero", "Цена: выигрыш vs «не изменится»"),
      ("direction", "auc", "Направление: AUC"), ("direction", "hit_rate", "Направление: доля угаданных"),
      ("direction", "brier", "Направление: Brier (меньше — лучше)"), ("variance", "crpss", "Уверенность: CRPSS (>0 — лучше константы)"),
      ("variance", "coverage90", "Уверенность: покрытие 90% (идеал 0,90)"), ("variance", "corr_var_err2_spearman", "Уверенность: связь разброса с ошибкой")]

V2 = [("skiponly", "skip_only"), ("ep3", "3 эпохи"), ("lr3e4", "LR 3e-4"), ("dir5", "direction ×5"), ("look20", "окно 20"),
      ("nophys", "без физики"), ("calval", "калибровка value"), ("calgrad", "калибровка gradient")]

# ---- the experiment registry: one tab each, newest last ------------------------------------------------------------
EXPERIMENTS = [
    {"id": "r1", "title": "Раунд 1", "when": "06.10", "kind": "auc",
     "goal": "Ограничить нелинейный путь направления (3 переключателя). 5 срезов × 20 seed'ов, 6-часовой блок, только направление.",
     "specs": [("hc_baseline", "база", 100), ("hc_r1_skip_only", "skip_only", 100), ("hc_r1_shrink1", "shrink 1.0", 100), ("hc_r1_drop05", "dropout 0.5", 100)],
     "compare": [("hc_baseline", "hc_r1_skip_only"), ("hc_baseline", "hc_r1_shrink1"), ("hc_baseline", "hc_r1_drop05")],
     "verdict": "Без эффекта; dropout чуть хуже. shrink принят на 83/100 по решению владельца."},
    {"id": "r23", "title": "Раунды 2–3", "when": "06–07.10", "kind": "auc",
     "goal": "7 Config-вариантов и балансировка лоссов. 40 срезов × 3 seed'а, 6-часовой блок, только направление. Правило: интервал > 0 и эффект ≥ +0,01.",
     "specs": [("hc2_base_c*", "база", 120)] + [(f"hc2_{v}_c*", l, 24 if v == "nophys" else 120) for v, l in V2],
     "compare": [("hc2_base_c*", f"hc2_{v}_c*") for v, _ in V2],
     "verdict": "Ни один вариант не прошёл правило; хуже базы: LR 3e-4 и окно 20."},
    {"id": "slice", "title": "Срез 2022-04-19", "when": "07.10", "kind": "static",
     "goal": "Почему на этом срезе все конфигурации получают высокий AUC.",
     "text": ["Блок проверки: суббота 16.04.2022, 11:01–18:30 UTC (Пасха, биржи США закрыты, BTC ≈ $40 450 в коридоре 0,52%, тонкая ликвидность).",
              "Простое правило «против последних 10 минут» даёт там AUC 0,756 — выше сети (0,67). Автокорреляция 10-минутных доходностей −0,32: штиль, возврат к среднему.",
              "Контраст: 2024-07-09 (тренд) — работает продолжение движения (0,605), сеть 0,446. 2022-10-23 — возврат работает (0,65), но сеть 0,44.",
              "На ~24 независимых точках чистый шум даёт 0,8 примерно в 1% случаев."],
     "verdict": "Режим рынка, а не навык сети.", "specs": [], "compare": []},
    {"id": "hc3", "title": "Топ-10 на 7 днях", "when": "07.10", "kind": "auc",
     "goal": "10 лучших по AUC конфигураций старой кампании на 7-дневном блоке, 4 среза × 1 seed (только направление).",
     "specs": [("hc3_default", "дефолт", 4)] + [(f"hc3_cand{i:02d}", f"кандидат {i}", 4) for i in range(1, 11)],
     "compare": [("hc3_default", "hc3_cand01"), ("hc3_default", "hc3_cand02")],
     "verdict": "Остановлено владельцем после 3 полных конфигураций: прироста против дефолта нет."},
    {"id": "cand", "title": "Кандидат 0,805", "when": "07.10", "kind": "cand",
     "goal": "skip_only, срез 2022-04-19, seed 2, голова h1 = 0,805. C1: тот же блок, 10 новых seed'ов. C2: 7 дней, 6 срезов × 2 seed'а. Все 9 выходов. SPEC: runs/tactical/cand_0805/SPEC.md",
     "specs": [("cand_c1_skip", "C1 skip_only", 10), ("cand_c1_base", "C1 база", 10), ("cand_c2_skip", "C2 skip_only, 7 дней", 12), ("cand_c2_base", "C2 дефолт, 7 дней", 12)],
     "compare": [("cand_c1_base", "cand_c1_skip"), ("cand_c2_base", "cand_c2_skip")],
     "verdict": "Закрыт: C1 и C2 провалены. Попутно: на 7 днях голова уверенности работает, на 6 часах — нет."},
    {"id": "bench", "title": "Скорость, 1 день", "when": "07.10", "kind": "bench",
     "goal": "Сколько прогонов в час даёт 1 процесс против 3 и пачка 256 против 1024 на 1-дневном блоке; качество всех 9 выходов для пачки 1024. SPEC: runs/tactical/bench_1d/make_bench.py",
     "specs": [("bench_bs256_x1", "A: 1 процесс, пачка 256", 12), ("bench_bs256_x3", "B: 3 процесса, пачка 256", 12), ("bench_bs1024_x1", "C: 1 процесс, пачка 1024", 12)],
     "compare": [("bench_bs256_x1", "bench_bs1024_x1")], "verdict": None},
    {"id": "epochs", "title": "Эпохи на 1 дне", "when": "07.10", "kind": "epochs",
     "goal": "Гипотеза: на 1 дне голова уверенности не учится из-за малого числа шагов (48), а не данных. Варианты: пачка 256 × 40 эпох (240 шагов), пачка 64 × 14 эпох (322 шага). SPEC: runs/tactical/epochs_1d/SPEC.md",
     "specs": [("ep1d_bs256_e40", "256 × 40 эпох (240 шагов)", 12), ("ep1d_bs64_e14", "64 × 14 эпох (322 шага)", 12),
               ("bench_bs256_x1", "справка: 1 день, 48 шагов", 12), ("cand_c2_base", "справка: 7 дней, 320 шагов", 12)],
     "compare": [("bench_bs256_x1", "ep1d_bs256_e40"), ("bench_bs256_x1", "ep1d_bs64_e14")], "verdict": None},
    {"id": "hc4r1", "title": "Хиллклаймб, раунд 1", "when": "07.10", "kind": "hc",
     "goal": "Неделя, 6 срезов × 2 seed'а, общая оценка по 3 группам (цена, направление, уверенность) в единицах шума + защита от «ничего не предсказывать». Вариант лучше, если интервал > 0, ни одна группа не ниже −0,5 и не упали AUC и связь разброса с ошибкой. SPEC: runs/tactical/hc4/SPEC.md",
     "base": "cand_c2_base",
     "specs": [("cand_c2_base", "база (дефолт)", 12), ("hc4_calval", "калибровка value", 12), ("hc4_calgrad", "калибровка gradient", 12),
               ("hc4_ep12", "12 эпох", 12), ("hc4_look120", "окно 120", 12), ("hc4_nophys", "без физики", 12), ("hc4_bs1024", "пачка 1024", 12)],
     "compare": [], "verdict": "Победителя нет: калибровка gradient хуже базы, остальные 5 — разницы нет. Config-настройки исчерпаны, следующий раунд требует нового кода."},
    {"id": "plan", "title": "План (идеи владельца)", "when": "08.10", "kind": "plan",
     "goal": "Живой план тактической сессии: ваши идеи и мои гипотезы со статусами. Файл runs/tactical/PLAN.md.",
     "specs": [], "compare": [], "verdict": None},
    {"id": "heads", "title": "Головы: ансамбль и пороги", "when": "08.10", "kind": "thresh",
     "goal": "На сохранённых ответах сети (11 прогонов базы, длинный блок, без переобучения): как связаны 9 голов, что дают комбинации горизонтов и как растёт точность направления с порогом уверенности. Журнал H14-H15.",
     "specs": [], "compare": [],
     "verdict": "Цена не несёт информации; ансамбль горизонтов лучше одной головы; точность растёт с порогом (до ~60% на верхних 2-5% баров при согласии горизонтов и ожидании большого движения), но заработок на сделку пока неотличим от случайного."},
    {"id": "probe", "title": "Пробник градиентов", "when": "08.10", "kind": "probe",
     "goal": "Какое слагаемое лосса сколько тянет ствол, головы и индикаторы, и где лоссы спорят (косинус). Дефолтная сеть, длинный блок, 2 прогона по очереди, PROBE_EVERY=10. Первый запуск упал (ошибка компиляции на GPU), исправлено.",
     "specs": [], "compare": [], "verdict": None},
    {"id": "noprice", "title": "Без цены: 3 горизонта vs 1", "when": "08.10", "kind": "hc",
     "goal": "Ваш эксперимент: голова цены удалена из лоссов и архитектуры (PRICE_HEAD=none). Сеть без цены на 3 горизонтах (ансамбль) против сети без цены на одном горизонте h1 (ACTIVE_HORIZONS=[1]); 6 срезов x 2 seed'а. Статус: implementer пишет переключатели; запуск после проверки.",
     "base": "hc5_noprice3",
     "specs": [("hc5_noprice3", "без цены, 3 горизонта", 12), ("hc5_noprice1", "без цены, 1 горизонт (h1)", 12)],
     "compare": [], "verdict": None},
]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20, cwd=REPO).stdout
    except Exception:
        return ""


_cache = {}


def rows_of(pattern):
    if pattern in _cache:
        return _cache[pattern]
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "screens", pattern, "results*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            a = r.get("direction_auc") or {}
            ok = all(a.get(h) and a[h].get("auc") is not None for h in H)
            ep = (r.get("timings") or {}).get("epoch_s") or []
            out.append({"key": (r["data_end"][:16], r["seed"]), "wall": r.get("wall_s"),
                        "auc": st.mean(a[h]["auc"] for h in H) if ok else None,
                        "h": [a[h]["auc"] if a.get(h) else None for h in H], "hm": r.get("head_metrics"),
                        "epoch": st.median(ep[1:]) if len(ep) > 1 else None})
    _cache[pattern] = out
    return out


def tci(xs):
    if len(xs) < 2:
        return None
    m = st.mean(xs); se = st.stdev(xs) / math.sqrt(len(xs)); t = T.get(len(xs) - 1, 2.0)
    return {"mean": m, "lo": m - t * se, "hi": m + t * se, "n": len(xs)}


def hm_mean(rows, g, k, h=None):
    hs = [h] if h else H
    v = [r["hm"][x][g].get(k) for r in rows if r.get("hm") for x in hs
         if r["hm"].get(x) and r["hm"][x].get(g) and r["hm"][x][g].get(k) is not None]
    return st.mean(v) if v else None


def paired(a_rows, b_rows):
    A = {r["key"]: r["auc"] for r in a_rows if r["auc"] is not None}
    per = collections.defaultdict(list)
    for r in b_rows:
        if r["auc"] is not None and r["key"] in A:
            per[r["key"][0]].append(r["auc"] - A[r["key"]])
    c = tci([st.mean(v) for v in per.values()])
    if c:
        c["slices"] = {s: st.mean(v) for s, v in per.items()}
        c["pairs"] = sum(len(v) for v in per.values())
        c["verdict"] = "лучше" if c["lo"] > 0 else ("хуже" if c["hi"] < 0 else "разницы не видно")
    return c


def main():
    _cache.clear()
    procs_all = sh('wmic process where "name=\'python.exe\'" get CommandLine').splitlines()
    procs = [l for l in procs_all if "cli screen" in l]
    exps = []
    for e in EXPERIMENTS:
        specs = []
        for pat, lab, planned in e["specs"]:
            rows = rows_of(pat)
            stem = pat.replace("*", "")
            run = sum(1 for l in procs if f"configs/tactical/{stem}" in l)
            walls = [r["wall"] for r in rows if r["wall"]]
            aucs = [r["auc"] for r in rows if r["auc"] is not None]
            specs.append({"pat": pat, "label": lab, "planned": planned, "done": len(rows), "running": run,
                          "wall": st.median(walls) if walls else None, "over": sum(w > 120 for w in walls),
                          "auc": st.mean(aucs) if aucs else None, "auc_se": (st.stdev(aucs) / math.sqrt(len(aucs))) if len(aucs) > 1 else None,
                          "epoch": st.median([r["epoch"] for r in rows if r["epoch"]]) if any(r["epoch"] for r in rows) else None,
                          "heads": {f"{g}.{k}": hm_mean(rows, g, k) for g, k, _ in HM} if any(r.get("hm") for r in rows) else None,
                          "perh": {h: {f"{g}.{k}": hm_mean(rows, g, k, h) for g, k, _ in HM} for h in H} if any(r.get("hm") for r in rows) else None})
        comps = []
        for a, b in e["compare"]:
            c = paired(rows_of(a), rows_of(b))
            la = next(s["label"] for s in specs if s["pat"] == a); lb = next(s["label"] for s in specs if s["pat"] == b)
            comps.append({"a": la, "b": lb, "c": c})
        running = sum(s["running"] for s in specs)
        main_specs = [s for s in specs if not s["label"].startswith("справка")]
        complete = bool(main_specs) and all(s["done"] >= s["planned"] for s in main_specs)
        status = "идёт" if running else ("готово" if complete or e.get("verdict") else ("в очереди" if not any(s["done"] for s in main_specs) else "частично"))
        x = {"id": e["id"], "title": e["title"], "when": e["when"], "kind": e["kind"], "goal": e["goal"], "text": e.get("text"),
             "verdict": e.get("verdict"), "specs": specs, "comps": comps, "status": status, "running": running}
        if e["kind"] == "cand":
            x["extra"] = cand_extra()
        if e["kind"] == "bench":
            x["extra"] = bench_extra()
            x["verdict"] = x["verdict"] or (x["extra"].get("verdict") if x["extra"] else None)
        if e["kind"] == "hc":
            rows = []
            for pat, lab, planned in e["specs"]:
                if pat == e["base"]:
                    continue
                try:
                    r = hc4_metric.compare(e["base"], pat)
                except Exception as ex:
                    r = {"error": str(ex)}
                rows.append({"label": lab, "r": {k: v for k, v in r.items() if k != "noise_sd"}})
            x["extra"] = {"hc": rows}
            win = [z for z in rows if z["r"].get("verdict") == "BETTER"]
            if all(sp["done"] >= sp["planned"] for sp in specs):
                x["verdict"] = ("Победитель раунда: " + max(win, key=lambda z: z["r"]["mean"])["label"]) if win else "В раунде нет варианта лучше базы."
        if e["kind"] == "plan":
            pp = os.path.join(ROOT, "PLAN.md")
            x["extra"] = {"md": open(pp, encoding="utf-8").read() if os.path.exists(pp) else ""}
        if e["kind"] == "thresh":
            cur = {}
            for h in (0, 1, 2):
                fp = os.path.join(ROOT, "probe", f"threshold_curves_cand_c2_base_h{h}.json")
                if os.path.exists(fp):
                    cur[f"h{h}"] = json.load(open(fp, encoding="utf-8"))
            hi = os.path.join(ROOT, "probe", "heads_interplay_cand_c2_base.json")
            x["extra"] = {"curves": cur, "interplay": json.load(open(hi, encoding="utf-8")) if os.path.exists(hi) else None}
        if e["kind"] == "probe":
            res = {}
            for fp in sorted(glob.glob(os.path.join(ROOT, "probe", "probe_*.json"))):
                d = json.load(open(fp, encoding="utf-8"))
                res[os.path.basename(fp)] = {k: (v[-1] if v else None) for k, v in d.get("probe", {}).items()}
            runo = os.path.join(ROOT, "probe", "run.out")
            log = open(runo, encoding="utf-8").read()[-400:] if os.path.exists(runo) else ""
            prog = []
            for a in ("2021-10-13T18:00:00_0", "2023-04-29T08:00:00_1"):
                cand = glob.glob(os.path.join(ROOT, "probe", f"log_{a[:10]}*_{a[-1]}.txt"))  # bash writes ':' as a private char
                lp = cand[0] if cand else os.path.join(ROOT, "probe", "missing")
                txt = open(lp, encoding="utf-8", errors="ignore").read() if os.path.exists(lp) else ""
                ep = re.findall(r"PROBE_EPOCH (\d+)/(\d+) (\S+)", txt)
                started = time.strftime("%H:%M", time.localtime(os.path.getmtime(lp))) if os.path.exists(lp) else None  # last write of the log
                prog.append({"run": a, "epochs_done": int(ep[-1][0]) if ep else 0, "epochs": int(ep[-1][1]) if ep else 8,
                             "last": ep[-1][2] if ep else None, "started": started,
                             "done": os.path.exists(os.path.join(ROOT, "probe", f"probe_{a[:10]}_s{a[-1]}.json"))})
            x["extra"] = {"runs": res, "log": log, "progress": prog}
            if any("probe_run.py" in l for l in procs_all):
                x["status"] = "идёт"
            elif res:
                x["status"] = "готово"
        if e["kind"] == "epochs":
            x["extra"] = epochs_extra(specs)
            if x["extra"] and x["extra"].get("final"):
                x["verdict"] = x["extra"]["verdict"]
        exps.append(x)
    active = next((x["id"] for x in reversed(exps) if x["status"] == "идёт"), None) or next((x["id"] for x in exps if x["status"] in ("в очереди", "частично")), None) or exps[-1]["id"]
    state_est = estimates(exps)
    res = []
    rp = os.path.join(ROOT, "hc4", "resources.csv")
    if os.path.exists(rp):
        for l in open(rp, encoding="utf-8").read().splitlines()[1:][-360:]:
            p = l.split(",")
            try:
                res.append([p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), ",".join(p[5:]).strip(",")])
            except Exception:
                pass
    gl = os.path.join(ROOT, "hc4", "guard.log")
    stops = [l for l in open(gl, encoding="utf-8").read().splitlines() if " stop " in l][-5:] if os.path.exists(gl) else []
    procs_all_n = sum(1 for l in procs_all if ("configs/tactical/" in l or "probe_run.py" in l))
    state = {"procs_all_n": procs_all_n, "res": res, "stops": stops, "est": state_est, "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "gpu": sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").strip(),
             "procs": len(procs), "exps": exps, "active": active, "hm": [[g + "." + k, lab] for g, k, lab in HM],
             "log": sh("git log --pretty=format:%h|%ad|%s --date=format:%m-%d %H:%M -12").splitlines()}
    tmp = OUT + ".tmp"
    open(tmp, "w", encoding="utf-8").write(TEMPLATE.replace("__STATE__", json.dumps(state)))
    for _ in range(10):  # the browser may hold the file for a moment while it reloads (WinError 5): retry
        try:
            os.replace(tmp, OUT)
            break
        except PermissionError:
            time.sleep(0.5)


def estimates(exps):
    """Guesstimates per stage (owner, 2026-10-07). Time of the running round is computed live from the measured run
    times; everything else, and every accuracy figure, is the lead's judgement from the evidence so far, labelled so."""
    now = datetime.datetime.now()
    hc = next((x for x in exps if x["id"] == "hc4r1"), None)
    live = None
    if hc:
        walls = [sp["wall"] for sp in hc["specs"] if sp["wall"]]
        per = st.median(walls) if walls else 420.0
        left = sum(max(sp["planned"] - sp["done"], 0) for sp in hc["specs"])
        mins = left * per / 1 / 60  # one process at a time (guard v3)
        live = {"left": left, "per_run_s": round(per), "minutes": round(mins), "at": (now + datetime.timedelta(minutes=mins)).strftime("%H:%M")}
    t = lambda m: (now + datetime.timedelta(minutes=m)).strftime("%d.%m %H:%M")
    base = live["minutes"] if live else 0
    stages = [
        {"stage": "Хиллклаймб, раунд 1 (6 вариантов, неделя)", "status": "идёт" if live and live["left"] else "готово",
         "time": (f"осталось ~{live['minutes']} мин ({live['left']} прогонов × ~{live['per_run_s']} с, 1 процесс), конец ≈ {live['at']}" if live else "—"),
         "acc": "Вероятность, что хоть один вариант «лучше»: ~25–35%. Самый вероятный кандидат — калибровка value (так работает основной конвейер): ждём выигрыш в группе «уверенность» (+0,3…+1 шума), направление без изменений (AUC 0,53–0,56).",
         "conf": "низкая–средняя"},
        {"stage": "Раунд 2 (если в раунде 1 есть победитель): 4–6 вариантов поверх победителя", "status": "план",
         "time": f"~2–2,5 ч, конец ≈ {t(base + 150)}",
         "acc": "Если раунд 1 дал победителя, шанс ещё одного шага вверх ~20–30%: эффекты от Config-настроек обычно убывают.",
         "conf": "низкая"},
        {"stage": "Проверка победителя на 6 отложенных срезах", "status": "план",
         "time": "~30 мин (12 прогонов)",
         "acc": "Шанс, что выигрыш подтвердится: ~50% (на прошлых раундах «лучшие» откатывались на свежих данных на 50–80% отрыва).",
         "conf": "средняя"},
        {"stage": "Если Config исчерпан: новый код — сеть, распознающая режим рынка (штиль ↔ тренд)", "status": "идея",
         "time": "implementer ~1–2 ч + раунд ~30–60 мин на неделе",
         "acc": "Цель — AUC направления 0,56–0,60 там, где сейчас 0,52–0,55. Шанс ~10–20%: разбор среза 2022-04-16 показал, что режим объясняет успехи и провалы, но сеть пока его не ловит.",
         "conf": "низкая"},
        {"stage": "Итог сессии по точности (ориентир)", "status": "—", "time": "—",
         "acc": "Реалистично: уверенность (риск) — покрытие 0,88–0,90, CRPSS +0,01…+0,05; цена — около «не изменится»; направление — 0,53–0,57. Прорыв направления выше 0,60 стабильно — маловероятен (<10%).",
         "conf": "средняя"},
    ]
    return {"live": live, "stages": stages}


def cand_extra():
    s1 = {r["key"]: r for r in rows_of("cand_c1_skip")}; b1 = {r["key"]: r for r in rows_of("cand_c1_base")}
    h1 = [r["h"][1] for r in s1.values() if r["h"][1] is not None]
    d1 = tci([s1[k]["auc"] - b1[k]["auc"] for k in s1 if k in b1 and s1[k]["auc"] is not None and b1[k]["auc"] is not None])
    m1 = [r["auc"] for r in s1.values() if r["auc"] is not None]
    c2 = paired(rows_of("cand_c2_base"), rows_of("cand_c2_skip"))
    gc = (hm_mean(rows_of("cand_c2_skip"), "variance", "crpss") or 0) - (hm_mean(rows_of("cand_c2_base"), "variance", "crpss") or 0)
    gv = (hm_mean(rows_of("cand_c2_skip"), "variance", "coverage90") or 0) - (hm_mean(rows_of("cand_c2_base"), "variance", "coverage90") or 0)
    R = lambda name, val, ok: {"name": name, "val": val, "ok": ok}
    return {"rules": [
        R("C1 (a) h1 AUC в среднем ≥ 0,75", f"{st.mean(h1):.3f}" if h1 else "—", (st.mean(h1) >= 0.75) if h1 else None),
        R("C1 (b) лучше базы, интервал по seed'ам > 0", f"{d1['mean']:+.3f} [{d1['lo']:+.3f}; {d1['hi']:+.3f}]" if d1 else "—", (d1["lo"] > 0) if d1 else None),
        R("C1 (c) среднее h0–h2 ≥ 0,756 (правило «против 10 мин»)", f"{st.mean(m1):.3f}" if m1 else "—", (st.mean(m1) >= 0.756) if m1 else None),
        R("C2 (a) лучше дефолта, интервал по срезам > 0", f"{c2['mean']:+.3f} [{c2['lo']:+.3f}; {c2['hi']:+.3f}], срезов {c2['n']}" if c2 else "—", (c2["lo"] > 0) if c2 else None),
        R("C2 (b) CRPSS и покрытие не хуже −0,05", f"CRPSS {gc:+.3f}, покрытие {gv:+.3f}", gc >= -0.05 and gv >= -0.05)]}


def bench_extra():
    p = os.path.join(ROOT, "bench_1d", "times.txt")
    if not os.path.exists(p):
        return None
    t = {}
    for l in open(p):
        parts = l.split()
        if len(parts) == 3:
            t.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    arms = []
    for arm, pat, steps in (("bs256_x1", "bench_bs256_x1", 6), ("bs256_x3", "bench_bs256_x3", 6), ("bs1024_x1", "bench_bs1024_x1", 2)):
        rows = rows_of(pat); x = t.get(arm, {})
        dur = (x["end"] - x["start"]) if "start" in x and "end" in x else ((time.time() - x["start"]) if "start" in x else None)
        ep = [r["epoch"] for r in rows if r["epoch"]]
        arms.append({"arm": arm, "done": len(rows), "minutes": round(dur / 60, 1) if dur else None, "finished": "end" in x,
                     "per_hour": round(len(rows) / (dur / 3600), 1) if dur and rows else None,
                     "epoch_s": round(st.median(ep), 2) if ep else None, "s_per_step": round(st.median(ep) / steps, 3) if ep else None})
    v = None
    if all(a["finished"] for a in arms):
        best = max(arms, key=lambda a: a["per_hour"] or 0)
        v = f"Быстрее всего: {best['arm']} ({best['per_hour']} прогонов/ч)."
    return {"arms": arms, "verdict": v}


def epochs_extra(specs):
    by = {s["pat"]: s for s in specs}
    out = []
    for pat in ("ep1d_bs256_e40", "ep1d_bs64_e14"):
        s = by[pat]
        if not s["heads"]:
            out.append({"label": s["label"], "cov": None, "crpss": None, "ok": None, "done": s["done"]}); continue
        cov = s["heads"]["variance.coverage90"]; cr = s["heads"]["variance.crpss"]
        out.append({"label": s["label"], "cov": cov, "crpss": cr, "ok": (cov >= 0.80 and cr >= 0), "bad": (cov < 0.75 or cr < -0.10), "done": s["done"]})
    final = all(by[p]["done"] >= by[p]["planned"] for p in ("ep1d_bs256_e40", "ep1d_bs64_e14"))
    verdict = None
    if final:
        verdict = "Подтверждена" if any(o["ok"] for o in out) else ("Опровергнута" if all(o.get("bad") for o in out) else "Частично")
    return {"arms": out, "final": final, "verdict": verdict}


TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Tactical session</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202a;--mut:#667;--line:#e1e5ea;--ok:#199e70;--bad:#c0392b;--acc:#3987e5}
@media(prefers-color-scheme:dark){:root{--bg:#10151b;--card:#18202a;--ink:#e6ebf0;--mut:#93a0ad;--line:#2a3541}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
.w{max-width:1280px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--mut);font-size:12px}.g{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));margin-top:10px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.full{grid-column:1/-1}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;border-bottom:1px solid var(--line);padding-bottom:8px}
.tab{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;font-size:13px}
.tab.on{border-color:var(--acc);box-shadow:inset 0 -3px 0 var(--acc)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:1px}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.row{display:grid;grid-template-columns:230px 1fr 210px;gap:8px;align-items:center;margin:5px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.mut{color:var(--mut)}code{background:var(--line);padding:0 4px;border-radius:3px}
.goal{border-left:4px solid var(--acc);padding:6px 10px;background:var(--card);border-radius:6px;font-size:13px;margin-top:10px}
.verd{font-size:14px;margin-top:8px}
</style></head><body><div class="w">
<h1>Тактическая сессия: прорыв в расчёте сети</h1>
<div class="sub" id="sub"></div>
<div class="goal"><b>Цель:</b> выйти из «ловушки» (нет навыка направления) поиском тактического прорыва в расчёте сети; риск-менеджмент — отдельная ветка. <b>Правила владельца:</b> всегда все 9 выходов; отсев 6 срезов × 2 seed'а; screen ≤ 2 мин на прогон (7-дневные — по разрешению); у каждого прогона своя вкладка.</div>
<div class="c" style="margin-top:12px"><h2>Оценки по этапам: время и точность <span class="mut">(guesstimates — мои оценки, не результаты)</span></h2><table id="est"></table>
<div class="sub">Время текущего раунда считается вживую по реальным прогонам; остальное — оценки по уже полученным данным, будут пересматриваться после каждого раунда.</div></div>
<div class="c" style="margin-top:12px"><h2>Нагрузка машины в реальном времени <span class="mut">(проверка каждые 10 с; правило владельца)</span></h2>
<div id="resnow" class="sub"></div><div id="resplot" style="height:220px"></div>
<div class="sub">Ограничитель: мои процессы с пониженным приоритетом; не больше 3; новый — только при ≥ 12 ГБ свободной памяти и CPU &lt; 80%; самый новый мой процесс останавливается при &lt; 6 ГБ свободной памяти или CPU ≥ 95% дольше минуты.</div><div id="stops" class="sub"></div></div>
<div class="tabs" id="tabs"></div>
<div id="pane"></div>
<div class="g"><div class="c full"><h2>Коммиты nt-tactical</h2><table id="git"></table></div></div>
</div>
<script>
const S=__STATE__,ink=getComputedStyle(document.body).color,grid=getComputedStyle(document.documentElement).getPropertyValue('--line');
const lay=o=>Object.assign({margin:{l:52,r:12,t:8,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:ink,size:12},xaxis:{gridcolor:grid},yaxis:{gridcolor:grid},legend:{orientation:'h',y:-0.25}},o||{});
const cfg={displayModeBar:false,responsive:true},f3=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(3),p3=v=>v==null?'—':v.toFixed(3);
const SC={'идёт':'#199e70','готово':'#8a97a6','в очереди':'#c9a227','частично':'#d95926'};
const mark=p=>p==null?'<span class="mut">ждёт данных</span>':(p?'<span class="ok">✔</span>':'<span class="bad">✘</span>');
document.getElementById('sub').textContent='Обновлено '+S.now+' · перезагрузка каждые 20 с · процессов на GPU от нас: '+S.procs+' · GPU: '+(S.gpu||'н/д');
let cur=S.active;try{const h=location.hash.slice(1);if(h&&S.exps.some(e=>e.id===h))cur=h}catch(e){}
function tabs(){document.getElementById('tabs').innerHTML=S.exps.map(e=>`<button class="tab ${e.id===cur?'on':''}" onclick="show('${e.id}')"><span class="dot" style="background:${SC[e.status]||'#8a97a6'}"></span>${e.title} <span class="mut">· ${e.when} · ${e.status}</span></button>`).join('')}
function show(id){cur=id;try{history.replaceState(null,'','#'+id)}catch(e){}tabs();render()}
function render(){const e=S.exps.find(x=>x.id===cur);let h=`<div class="g"><div class="c full"><h2>${e.title} <span class="mut">(${e.status})</span></h2><div>${e.goal}</div>`+
 (e.verdict?`<div class="verd"><b>Итог:</b> ${e.verdict}</div>`:'<div class="verd mut">Итога пока нет.</div>')+'</div>';
 if(e.text)h+='<div class="c full"><ul>'+e.text.map(t=>'<li>'+t+'</li>').join('')+'</ul></div>';
 if(e.specs.length){h+='<div class="c full"><h2>Прогресс</h2>'+e.specs.map(s=>{const p=Math.min(100,100*s.done/s.planned);
   return `<div class="row"><div><b>${s.label}</b></div><div class="bar"><i style="width:${p}%"></i></div><div class="sub">${s.done}/${s.planned}${s.running?' · идёт ('+s.running+')':''}${s.wall?' · '+Math.round(s.wall)+' с/прогон':''}${s.over?' · >120 с: '+s.over:''}</div></div>`}).join('')+'</div>';}
 if(e.extra&&e.extra.rules)h+='<div class="c full"><h2>Требования (SPEC)</h2><table><tr><th>Требование</th><th>Факт</th><th></th></tr>'+e.extra.rules.map(r=>`<tr><td>${r.name}</td><td>${r.val}</td><td>${mark(r.ok)}</td></tr>`).join('')+'</table></div>';
 if(e.kind==='bench'&&e.extra)h+='<div class="c full"><h2>Скорость</h2><table><tr><th>Вариант</th><th>прогонов</th><th>минут</th><th>прогонов/час</th><th>эпоха, с</th><th>с/шаг</th></tr>'+e.extra.arms.map(a=>`<tr><td>${a.arm}${a.finished?'':' <span class="mut">(идёт)</span>'}</td><td>${a.done}</td><td>${a.minutes??'—'}</td><td><b>${a.per_hour??'—'}</b></td><td>${a.epoch_s??'—'}</td><td>${a.s_per_step??'—'}</td></tr>`).join('')+'</table><div class="sub">Минуты — по часам от начала до конца варианта; включают загрузку данных 6 срезов.</div></div>';
 if(e.kind==='epochs'&&e.extra)h+='<div class="c full"><h2>Правило гипотезы</h2><div class="sub">Подтверждена: покрытие ≥ 0,80 и CRPSS ≥ 0 хотя бы у одного варианта. Опровергнута: у обоих покрытие &lt; 0,75 или CRPSS &lt; −0,10. Справка: 1 день/48 шагов — 0,67 и −0,22; 7 дней — 0,87 и +0,01.</div><table><tr><th>Вариант</th><th>прогонов</th><th>покрытие 90%</th><th>CRPSS</th><th></th></tr>'+e.extra.arms.map(a=>`<tr><td>${a.label}</td><td>${a.done}</td><td>${p3(a.cov)}</td><td>${f3(a.crpss)}</td><td>${mark(a.ok)}</td></tr>`).join('')+'</table></div>';
 if(e.kind==='plan'&&e.extra){const md=e.extra.md.replace(/&/g,'&amp;').replace(/</g,'&lt;');
   h+='<div class="c full">'+md.split('\n').map(l=>l.startsWith('## ')?'<h2 style="margin-top:12px">'+l.slice(3)+'</h2>':(l.startsWith('# ')?'<h2>'+l.slice(2)+'</h2>':(l.startsWith('- ')?'<div>&bull; '+l.slice(2)+'</div>':(l.trim()?'<div>'+l+'</div>':'')))).join('')+'</div>';}
 if(e.kind==='thresh'&&e.extra){const I=e.extra.interplay;
   if(I)h+='<div class="c full"><h2>Как связаны головы (среднее по 11 прогонам)</h2><table><tr><th></th><th>h0</th><th>h1</th><th>h2</th></tr>'+[['AUC головы направления','auc_direction_head'],['AUC знака головы цены','auc_price_head'],['Корреляция цены с фактом','corr_price_y'],['Выигрыш цены при лучшем масштабе','skill_at_best_beta'],['Разброс прогноза цены / реальный','sd_pred_over_sd_true'],['Самоуверенность направления |p-0,5|','mean_abs_p_minus_half'],['Связь разброса с |движением|','spearman_var_absy']].map(([l,k])=>'<tr><td>'+l+'</td>'+['h0','h1','h2'].map(z=>'<td>'+p3(I[z][k])+'</td>').join('')+'</tr>').join('')+'</table></div>';
   h+='<div class="c full"><h2>Точность направления против порога уверенности</h2><div class="sub">Доля угаданных на барах с уверенностью выше порога; по оси X - какая доля баров остаётся (меньше = строже). Зелёный пунктир - 60%. Пороги взяты по тем же данным (небольшое подглядывание); следующий шаг - пороги с калибровочного блока. Горизонты h0/h2 включаются в легенде.</div><div id="thplot" style="height:380px"></div><h2>Среднее движение в нашу сторону на сделку и случайный эталон</h2><div id="thbps" style="height:300px"></div></div>';}
 if(e.kind==='probe'&&e.extra){const R=e.extra.runs,ks=Object.keys(R);
   if(e.extra.progress)h+='<div class="c full"><h2>Прогресс</h2>'+e.extra.progress.map(q=>'<div class="row"><div><b>прогон '+q.run+'</b></div><div class="bar"><i style="width:'+(q.done?100:100*q.epochs_done/q.epochs)+'%"></i></div><div class="sub">'+(q.done?'готово':(q.started?('эпоха '+q.epochs_done+'/'+q.epochs+(q.last?' (последняя '+q.last+')':' (сборка графа, лог обновлён '+q.started+')')):'ждёт'))+'</div></div>').join('')+'</div>';
   if(!ks.length)h+='<div class="c full"><h2>Результатов пока нет</h2><pre class="sub">'+(e.extra.log||'')+'</pre></div>';
   else{const terms=['point','trend','dir','nll','crps','coherence','ife','vol','t_perp','casimir','hd','vac_overflow','inter_reg'];
    ['trunk','head','indicator'].forEach(g=>{h+='<div class="c full"><h2>Доля градиента и косинус с общим градиентом: '+({trunk:'ствол',head:'головы',indicator:'индикаторы'})[g]+'</h2><table><tr><th>слагаемое</th>'+ks.map(k=>'<th>'+k.replace('probe_','').replace('.json','')+': доля</th><th>cos</th>').join('')+'</tr>'+
     terms.map(t=>'<tr><td>'+t+'</td>'+ks.map(k=>'<td>'+p3(R[k]['probe_grad_share_'+t+'_'+g])+'</td><td>'+f3(R[k]['probe_cos_'+t+'_'+g])+'</td>').join('')+'</tr>').join('')+
     '<tr><td><b>конфликт: средний / худший cos пары</b></td>'+ks.map(k=>'<td>'+f3(R[k]['probe_conflict_mean_'+g])+'</td><td>'+f3(R[k]['probe_conflict_min_'+g])+'</td>').join('')+'</tr></table></div>'});}}
 if(e.kind==='hc'&&e.extra){const VV={'BETTER':'<span class="ok">лучше</span>','WORSE':'<span class="bad">хуже</span>','NO DIFFERENCE':'<span class="mut">разницы нет</span>'};
   h+='<div class="c full"><h2>Общая оценка против базы (в единицах шума)</h2><table><tr><th>Вариант</th><th>срезов</th><th>общая оценка</th><th>95% интервал</th><th>цена</th><th>направление</th><th>уверенность</th><th>AUC / ранжирование риска</th><th>вердикт</th></tr>'+
   e.extra.hc.map(z=>{const r=z.r,g=r.groups||{},q=r.resolution||{};return `<tr><td><b>${z.label}</b></td><td>${r.slices??0}</td><td>${f3(r.mean)}</td><td>${r.lo==null?'—':'['+f3(r.lo)+'; '+f3(r.hi)+']'}</td><td>${f3(g.price)}</td><td>${f3(g.direction)}</td><td>${f3(g.confidence)}</td><td>${f3(q['direction.auc'])} / ${f3(q['variance.corr_var_err2_spearman'])}</td><td>${r.verdict?VV[r.verdict]:'<span class="mut">ждёт данных</span>'}</td></tr>`}).join('')+
   '</table><div class="sub">Значения — средний эффект в единицах seed-шума (±1 ≈ разница двух сидов одной сети). Пока срезов меньше 6, вердикт предварительный.</div></div>';
   const ok=e.extra.hc.filter(z=>z.r.mean!=null);if(ok.length)h+='<div class="c full"><h2>Лес общей оценки</h2><div id="hcforest" style="height:'+(90+40*ok.length)+'px"></div></div>';}
 if(e.comps.length)h+='<div class="c full"><h2>Парные сравнения по направлению (Δ AUC, 95% интервал по срезам)</h2><div id="forest" style="height:'+(80+36*e.comps.length)+'px"></div></div>';
 if(e.specs.some(s=>s.heads))h+='<div class="c full"><h2>Все 9 выходов (среднее h0–h2)</h2><table><tr><th>Метрика</th>'+e.specs.filter(s=>s.heads).map(s=>`<th>${s.label}</th>`).join('')+'</tr>'+S.hm.map(([k,l])=>'<tr><td>'+l+'</td>'+e.specs.filter(s=>s.heads).map(s=>`<td>${p3(s.heads[k])}</td>`).join('')+'</tr>').join('')+'</table></div>';
 if(e.specs.some(s=>s.auc!=null))h+='<div class="c"><h2>Средний AUC направления</h2><div id="aucbar" style="height:300px"></div></div><div class="c"><h2>Время прогона</h2><div id="wallbar" style="height:300px"></div></div>';
 h+='</div>';document.getElementById('pane').innerHTML=h;
 if(document.getElementById('thplot')){const C=e.extra.curves,cols={single_h:'#8a97a6',mean3:'#3987e5',agree3:'#d95926',agree3_hivar:'#199e70'},lab={single_h:'одна голова',mean3:'среднее 3 горизонтов',agree3:'3 горизонта согласны',agree3_hivar:'согласны + ждём большое движение'},dash={h0:'dot',h1:'solid',h2:'dash'};
   const tr=[],tb=[];Object.keys(C).forEach(hz=>Object.keys(cols).forEach(sg=>{const R=C[hz].filter(r=>r.signal===sg);if(!R.length)return;
    tr.push({type:'scatter',mode:'lines+markers',name:lab[sg]+' ('+hz+')',x:R.map(r=>r.coverage*100),y:R.map(r=>r.hit*100),line:{color:cols[sg],dash:dash[hz]},visible:hz==='h1'?true:'legendonly'});
    tb.push({type:'scatter',mode:'lines+markers',name:lab[sg]+' ('+hz+')',x:R.map(r=>r.coverage*100),y:R.map(r=>r.gross_bps),line:{color:cols[sg],dash:dash[hz]},visible:hz==='h1'?true:'legendonly'});
    if(sg==='agree3_hivar'&&hz==='h1')tb.push({type:'scatter',mode:'lines',name:'случайный эталон (95%)',x:R.map(r=>r.coverage*100),y:R.map(r=>r.null95_bps),line:{color:'#c0392b',dash:'dot'}});}));
   Plotly.newPlot('thplot',tr,lay({xaxis:{title:'какая доля баров остаётся, %',type:'log',autorange:'reversed',gridcolor:grid},yaxis:{title:'угадано, %',gridcolor:grid},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:60,y1:60,line:{color:'#199e70',dash:'dot'}},{type:'line',xref:'paper',x0:0,x1:1,y0:50,y1:50,line:{color:'#c0392b',dash:'dash',width:1}}]}),cfg);
   Plotly.newPlot('thbps',tb,lay({xaxis:{title:'какая доля баров остаётся, %',type:'log',autorange:'reversed',gridcolor:grid},yaxis:{title:'б.п. на сделку',gridcolor:grid,zeroline:true}}),cfg);}
 if(document.getElementById('hcforest')){const ok=e.extra.hc.filter(z=>z.r.mean!=null);Plotly.newPlot('hcforest',[{type:'scatter',mode:'markers',y:ok.map(z=>z.label),x:ok.map(z=>z.r.mean),marker:{size:12,color:ok.map(z=>z.r.verdict==='BETTER'?'#199e70':(z.r.verdict==='WORSE'?'#c0392b':'#3987e5'))},
  error_x:{type:'data',symmetric:false,array:ok.map(z=>z.r.hi-z.r.mean),arrayminus:ok.map(z=>z.r.mean-z.r.lo),thickness:2,width:6}}],lay({xaxis:{title:'общая оценка (единицы шума)',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'},yaxis:{autorange:'reversed'},margin:{l:170,r:12,t:8,b:40},showlegend:false}),cfg)}
 if(document.getElementById('forest')){const cs=e.comps.filter(x=>x.c);Plotly.newPlot('forest',[{type:'scatter',mode:'markers',y:cs.map(x=>x.b+' vs '+x.a),x:cs.map(x=>x.c.mean),marker:{size:11,color:cs.map(x=>x.c.verdict==='хуже'?'#c0392b':(x.c.verdict==='лучше'?'#199e70':'#3987e5'))},
  error_x:{type:'data',symmetric:false,array:cs.map(x=>x.c.hi-x.c.mean),arrayminus:cs.map(x=>x.c.mean-x.c.lo),thickness:2,width:6},text:cs.map(x=>x.c.n+' срезов, '+x.c.pairs+' пар: '+x.c.verdict),hovertemplate:'%{y}: %{x:>+.3f}<br>%{text}<extra></extra>'}],
  lay({xaxis:{title:'Δ AUC',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'},yaxis:{autorange:'reversed'},margin:{l:330,r:12,t:8,b:40},showlegend:false}),cfg)}
 if(document.getElementById('aucbar')){const ss=e.specs.filter(s=>s.auc!=null);Plotly.newPlot('aucbar',[{type:'bar',x:ss.map(s=>s.label),y:ss.map(s=>s.auc),error_y:{type:'data',array:ss.map(s=>s.auc_se?1.96*s.auc_se:0)},marker:{color:'#3987e5'}}],lay({yaxis:{range:[.4,.75],gridcolor:grid},xaxis:{tickangle:-25},margin:{l:52,r:12,t:8,b:110},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:.5,y1:.5,line:{color:'#c0392b',dash:'dash',width:1}}]}),cfg);
  const ww=e.specs.filter(s=>s.wall);Plotly.newPlot('wallbar',[{type:'bar',x:ww.map(s=>s.label),y:ww.map(s=>s.wall),marker:{color:'#8a97a6'}}],lay({yaxis:{title:'с (медиана)',gridcolor:grid},xaxis:{tickangle:-25},margin:{l:52,r:12,t:8,b:110},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:120,y1:120,line:{color:'#c0392b',dash:'dash'}}]}),cfg)}}
document.getElementById('est').innerHTML='<tr><th>Этап</th><th>Статус</th><th>Время (оценка)</th><th>Точность (оценка)</th><th>Уверенность в оценке</th></tr>'+S.est.stages.map(x=>`<tr><td><b>${x.stage}</b></td><td>${x.status}</td><td>${x.time}</td><td>${x.acc}</td><td>${x.conf}</td></tr>`).join('');
if(S.res.length){const L=S.res[S.res.length-1];document.getElementById('resnow').innerHTML=`Сейчас (${L[0]}): свободно памяти <b>${L[1]} ГБ</b> из 64 · CPU <b>${L[2]}%</b> · GPU <b>${L[3]}%</b>, ${L[4]} МБ · моих процессов обучения: <b>${S.procs}</b>`;
 Plotly.newPlot('resplot',[{type:'scatter',mode:'lines',name:'CPU, %',x:S.res.map(r=>r[0]),y:S.res.map(r=>r[2]),line:{color:'#d95926'}},{type:'scatter',mode:'lines',name:'GPU, %',x:S.res.map(r=>r[0]),y:S.res.map(r=>r[3]),line:{color:'#199e70'}},
 {type:'scatter',mode:'lines',name:'свободно RAM, ГБ',x:S.res.map(r=>r[0]),y:S.res.map(r=>r[1]),yaxis:'y2',line:{color:'#3987e5'}}],
 lay({yaxis:{title:'%',range:[0,100],gridcolor:grid},yaxis2:{title:'ГБ',overlaying:'y',side:'right',range:[0,64]},margin:{l:45,r:45,t:8,b:30},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:95,y1:95,line:{color:'#c0392b',dash:'dot'}}]}),cfg);}
else document.getElementById('resnow').textContent='Данных о нагрузке пока нет.';
document.getElementById('stops').innerHTML=S.stops.length?('Остановки из-за перегрузки: '+S.stops.join(' · ')):'Остановок из-за перегрузки не было.';
tabs();render();
document.getElementById('git').innerHTML=S.log.map(l=>{const[h,t,...m]=l.split('|');return `<tr><td><code>${h}</code></td><td class="sub">${t}</td><td>${m.join('|')}</td></tr>`}).join('');
</script></body></html>"""

if __name__ == "__main__":
    if "--loop" in sys.argv:
        step = int(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            try:
                main()
            except Exception as e:
                print("dashboard error:", e, flush=True)
            time.sleep(step)
    else:
        main()
