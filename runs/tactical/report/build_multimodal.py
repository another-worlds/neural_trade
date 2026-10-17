"""Builds runs/tactical/report/multimodal_report.html: the MORNING REPORT of the multimodal sprint (T14) in ONE self-contained animated page, in Russian
(no CDN, no plotly: the animations are SVG drawn from the data below, the visual system of positive_outlook.html is reused: pos.css, the helper prelude of pos.js).
Owner preferences (2026-10-17): visual first, a short conclusion on top of every section, build-on-what-works tone (every miss comes with its next step),
technical numbers inside a collapsible 'Детали' block, a map of alternative pipelines kept in view, numbers exactly as measured.

Inputs (all read-only, all committed):
  runs/tactical/tasks/T14_multimodal/**/results*.json, tables.txt, figures, weights_sha256.txt, versions_*.txt, smoke.md, M8_module/frozen_P.json, M9_module/*
  runs/tactical/research/2026-10-17-multimodal/README.md (catalogue, new-data concepts)
  M3_kronos results: read from the working tree when merged; otherwise from `git show origin/nt-tactical-t14-m3:...`; when neither exists M3 is drawn as 'в расчёте'
  (the builder is re-runnable: run it again after M3 is merged).
  prospective timeline: the .sha256 / REPORT files of the earlier studies (T04-T13 in this tree; O4-O11 in the read-only outlook worktree --outlook).
The builder ASSERTS the key numbers against the JSON files before anything is embedded and fails the build on a mismatch.
usage: python runs/tactical/report/build_multimodal.py [--outlook D:/nt/nt_outlook_ro]   -> multimodal_report.html, png assets embedded (CPU, seconds)"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rcommon import T, TAC, ROOT  # noqa: E402  (theme, tactical folder)

ap = argparse.ArgumentParser()
ap.add_argument("--outlook", default="D:/nt/nt_outlook_ro")
args = ap.parse_args()
OUTL = Path(args.outlook) / "runs" / "tactical"
TASK = TAC / "tasks" / "T14_multimodal"
RES = TAC / "research" / "2026-10-17-multimodal"
POS = HERE / "positive_assets"
ASSETS = HERE / "mm_assets"
M3_BRANCH = "origin/nt-tactical-t14-m3"
M3_REL = "runs/tactical/tasks/T14_multimodal/M3_kronos"

PALETTE = dict(paper=T.PAPER, surface=T.SURFACE, ink=T.INK, ink2=T.INK_2, muted=T.MUTED, grid=T.GRID, axis=T.AXIS, neutral=T.NEUTRAL,
               up=T.UP_COLOR, down=T.DOWN_COLOR, gold=T.OTHER_SERIES[0], pink=T.OTHER_SERIES[1], tan=T.OTHER_SERIES[2], violet=T.OTHER_SERIES[3],
               salmon=T.OTHER_SERIES[4], good=T.GOOD)

CHECKS = []   # (label, got, expected, tol, ok)


def chk(label, got, exp, tol=5e-5):
    ok = abs(got - exp) <= tol
    CHECKS.append((label, got, exp, tol, ok))
    return got


def chk_true(label, cond):
    CHECKS.append((label, cond, True, 0, bool(cond)))


def jf(p):
    return json.load(open(p, encoding="utf-8"))


def rus(v, d=4, sign=True):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("\u2212" if v < 0 and float(f"{abs(v):.{d}f}") != 0 else ("+" if sign else "")) + s


def pct(v, d=2, sign=True):
    return rus(v * 100, d, sign) + "%"


# ------------------------------------------------------------------------------------------------ results
def gitshow(rel):
    r = subprocess.run(["git", "show", f"{M3_BRANCH}:{M3_REL}/{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def load_results(key, path):
    """-> normalised dict {look1:{A:{...},B:{...}}, look2:..., pass:{A,B,candidate}} or None when the file does not exist yet (M3)"""
    if path is None:
        return None
    d = jf(path) if not isinstance(path, bytes) else json.loads(path.decode("utf-8"))
    out = {"key": key, "pass": d["pass"], "sha": d.get("view_sha256")}
    for lk in ("look1", "look2"):
        L = d["looks"][lk]
        a, b = L["A"], L["B"]
        ai, br = a["increment"], b["rel_improvement"]
        out[lk] = {
            "months": len(L["months"]),
            "A": {"view": ai["view"]["mean"], "lo": ai["view"]["ci"][0], "hi": ai["view"]["ci"][1], "p": ai["view"]["p_one_sided"], "pos": ai["view"]["positive"],
                  "n": ai["view"]["n_months"], "c1": ai["c1"]["mean"], "c2a": ai["c2a"]["mean"], "c2b": ai["c2b"]["mean"] if "c2b" in ai else None,
                  "pass": bool(a["flags"]["pass"])},
            "B": {"view": br["view"]["mean"], "lo": br["view"]["ci"][0], "hi": br["view"]["ci"][1], "p": br["view"]["p_one_sided"], "n": b["n"],
                  "c1": br["c1"]["mean"], "c2a": br["c2a"]["mean"], "pass": bool(b["flags"]["pass"])},
            "V3": a["skill"]["V3"]["mean"], "brier_H3": b["brier"]["H3"],
        }
    return out


def rpath(rel):
    p = TASK / rel
    return p if p.exists() else None


def m3_results():
    p = rpath("M3_kronos/results.json")
    if p is not None:
        return p, "working tree"
    raw = gitshow("results.json")
    if raw:
        return raw, f"git show {M3_BRANCH} (not merged into this tree yet)"
    return None, "pending"


RUNS = {}
for key, rel in [
    ("M1", "M1_chronos2/results.json"), ("M1b", "M1b_chronos2_har/results.json"), ("M2", "M2_ttm/results.json"), ("M2b", "M2b_ttm_ft/results.json"),
    ("M4_img", "M4_regime_card/M4_img/results.json"), ("M4_cap", "M4_regime_card/M4_cap/results.json"), ("M4_both", "M4_regime_card/M4_both/results.json"),
    ("M4c_capnum", "M4_regime_card/M4c_capnum/results.json"), ("M4b_img_gbm", "M4_regime_card/M4b_img_gbm/results.json"),
    ("M5", "M5_visionts/results.json"), ("M5b", "M5b_visionts_seasonal/results.json"),
    ("M6", "M6_faiss/results.json"), ("M6b_k20", "M6b_faiss_resid_k20/results.json"), ("M6b_k50", "M6b_faiss_resid_k50/results.json"),
    ("M6b_k100", "M6b_faiss_resid_k100/results.json"), ("M6x_labels", "M6x_labels/results.json"), ("M6x_keys", "M6x_keys/results.json"),
    ("M6x_dist", "M6x_dist/results.json"),
    ("M7", "M7_gbm/results.json"), ("M7b", "M7b_gbm_plus_v3/results.json"), ("M7c", "M7c_gbm_resid/results.json"), ("M7s", "M7s_gbm/results.json"),
    ("M8_abl_stretch", "M8_abl_stretch/results.json"), ("M8_abl_volpct", "M8_abl_volpct/results.json"), ("M8_abl_hourwk", "M8_abl_hourwk/results.json"),
    ("M8_abl_sr", "M8_abl_sr/results.json"), ("M8_abl_lrv", "M8_abl_lrv/results.json"), ("M8_lin", "M8_lin/results.json"),
    ("M8x_abl_hour", "M8x_abl_hour/results.json"), ("M8x_abl_wkend", "M8x_abl_wkend/results.json"), ("M8x_lin_wk", "M8x_lin_wk/results.json"),
    ("M9_prof", "M9_prof/results.json"), ("M9_prof_str", "M9_prof_str/results.json"), ("M9_prof_str_vp", "M9_prof_str_vp/results.json"),
    ("M9_gbm7", "M9_gbm7/results.json"), ("M9h6", "M9h6_prof_str/results.json"), ("M9h6_vp", "M9h6_prof_str_vp/results.json"),
    ("M9h3", "M9h3_prof_str/results.json"),
]:
    RUNS[key] = load_results(key, rpath(rel))
M3_SRC_PATH, M3_SRC = m3_results()
RUNS["M3"] = load_results("M3", M3_SRC_PATH)
M3_PENDING = RUNS["M3"] is None

# ------------------------------------------------------------------------------------------------ the numbers the brief states: assert them
for k, a1, a2, b1, b2 in [("M7", 0.0273, 0.0124, 2.22, 2.04), ]:
    chk(f"{k} A look1", RUNS[k]["look1"]["A"]["view"], a1)
    chk(f"{k} A look2", RUNS[k]["look2"]["A"]["view"], a2)
    chk(f"{k} B look1 %", RUNS[k]["look1"]["B"]["view"] * 100, b1, 5e-3)
    chk(f"{k} B look2 %", RUNS[k]["look2"]["B"]["view"] * 100, b2, 5e-3)
chk_true("M7 passes A and B on both looks", all(RUNS["M7"][lk][t]["pass"] for lk in ("look1", "look2") for t in "AB") and RUNS["M7"]["pass"]["candidate"])
for k, a1, a2 in [("M1", 0.0074, 0.0026), ("M2", -0.0023, 0.0012), ("M2b", 0.0105, 0.0042), ("M5", 0.0005, 0.0037)]:
    chk(f"{k} A look1", RUNS[k]["look1"]["A"]["view"], a1)
    chk(f"{k} A look2", RUNS[k]["look2"]["A"]["view"], a2)
chk_true("M1, M2, M5 do not pass (pre-registered)", not any(RUNS[k]["pass"]["candidate"] for k in ("M1", "M2", "M5")))
chk_true("no M4 variant passes", not any(RUNS[k]["pass"]["candidate"] for k in ("M4_img", "M4_cap", "M4_both", "M4c_capnum", "M4b_img_gbm")))
c = RUNS["M4_cap"]
chk_true("M4 caption passes look 1 (A and B) and not look 2", c["look1"]["A"]["pass"] and c["look1"]["B"]["pass"] and not c["look2"]["A"]["pass"] and not c["look2"]["B"]["pass"])
chk_true("M6 pre-registered does not pass; M6b k50 passes A and not B", not RUNS["M6"]["pass"]["candidate"] and RUNS["M6b_k50"]["pass"]["A"] and not RUNS["M6b_k50"]["pass"]["B"])
_m7 = jf(TASK / "M7_gbm/results.json")["looks"]
chk_true("M7 is above the kernel control C2b on both targets and both looks",
         all(_m7[lk][t]["flags"]["above_c2b"] for lk in ("look1", "look2") for t in "AB"))
chk_true("M7c passes both targets", RUNS["M7c"]["pass"]["A"] and RUNS["M7c"]["pass"]["B"])
if not M3_PENDING:
    chk_true("M3 does not pass", not RUNS["M3"]["pass"]["candidate"])

R9 = jf(TASK / "M9_module/results_m9.json")
R8 = jf(TASK / "M8_module/results_m8.json")
F8 = jf(TASK / "M8_module/frozen_P.json")
F9 = jf(TASK / "M9_module/frozen_P_M9h6_prof_str.json")
PROF = jf(TASK / "M9_module/profile_M9h6_prof_str.json")
chk_true("M8 froze M7c_gbm_resid", F8["module"] == "M7c_gbm_resid")
chk_true("M9h6 candidate is post hoc with recipe sha e065e6b7", F9["recipe_sha256"].startswith("e065e6b7") and F9["status"] == "post hoc candidate")
chk_true("pre-registered M9 views (prof, prof_str, prof_str_vp) did not pass both looks on either target",
         all(not R9["views"][k][t]["pass_both_looks"] for k in ("M9_prof", "M9_prof_str", "M9_prof_str_vp") for t in "AB"))
chk_true("M9 freeze rule froze nothing", R9["freeze_choice"]["view"] is None)
chk("M9h6 R_A", R9["views"]["M9h6_prof_str"]["A"]["R"], 1.638, 5e-4)
chk("M9h6 R_B", R9["views"]["M9h6_prof_str"]["B"]["R"], 0.891, 5e-4)
chk_true("M8 main carrier is hourwk on both targets", R8["ablation"]["A"]["main_carrier"] == "hourwk" and R8["ablation"]["B"]["main_carrier"] == "hourwk")
chk("M8 hourwk share A", R8["ablation"]["A"]["groups"]["hourwk"]["share"], 0.62, 5e-3)
chk("M8 hourwk share B", R8["ablation"]["B"]["groups"]["hourwk"]["share"], 0.92, 5e-3)
base = jf(TASK / "core/reference/baselines.json")
SENS_RA = chk("M9h6 look-2-trimmed R_A (sensitivity table)", R9["sensitivity"]["M9h6_prof_str"]["A"]["R"], 3.06, 5e-3)
chk("V3 skill look1", base["looks"]["look1"]["A"]["V3_skill_vs_AR"]["mean"], 0.2299, 5e-5)
chk("V3 skill look2", base["looks"]["look2"]["A"]["V3_skill_vs_AR"]["mean"], 0.2384, 5e-5)

# ------------------------------------------------------------------------------------------------ the C2a lesson, computed (not asserted by hand)
FM = ["M1", "M1b", "M2", "M2b", "M5", "M5b", "M4_img", "M4b_img_gbm"] + ([] if M3_PENDING else ["M3"])
cmp_rows = []
n_beat = n_tot = 0
for k in FM:
    for lk in ("look1", "look2"):
        for t in "AB":
            r = RUNS[k][lk][t]
            n_tot += 1
            n_beat += int(r["view"] > r["c2a"])
            cmp_rows.append((k, lk, t, r["view"], r["c2a"]))
C2A_BEATEN = n_tot - n_beat   # comparisons in which C2a is above the view

# ------------------------------------------------------------------------------------------------ images
def small_png(src, maxw=900, colors=256):
    im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    im = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), buf.tell()


IMGS = {}
img_sizes = {}


def add_img(key, rel_or_bytes, w=900):
    if rel_or_bytes is None:
        return
    src = rel_or_bytes if isinstance(rel_or_bytes, bytes) else str(rel_or_bytes)
    IMGS[key], img_sizes[key] = small_png(src, w)


add_img("M1", rpath("M1_chronos2/fig_day.png"))
add_img("M2", rpath("M2_ttm/fig_day.png"))
add_img("M3", rpath("M3_kronos/fig_M3_kronos_example.png") or gitshow("fig_M3_kronos_example.png"))
add_img("M4", rpath("M4_regime_card/fig_regime_cards.png"))
add_img("M5", rpath("M5_visionts/fig_vts.png"))
add_img("M6", rpath("M6_faiss/fig_analogs.png"))
add_img("M7pd", rpath("M7_gbm/fig_pd.png"))
add_img("M8", rpath("M8_module/fig_m8.png"))
add_img("M9", rpath("M9_module/fig_m9.png"))
for sv in ("M1", "M2", "M3", "M4", "M5", "M6"):
    assert sv in IMGS or sv == "M3", sv

# ------------------------------------------------------------------------------------------------ candidate cards
STAT = {"works": "работает", "near": "почти", "nogain": "нет прироста над контролем", "wip": "в очереди", "newdata": "нужны новые данные"}


def run_js(k, label, post):
    r = RUNS[k]
    return {"key": k, "label": label, "post": post, "pass": r["pass"], "look1": r["look1"], "look2": r["look2"]}


def c2a_phrase(k):
    r = RUNS[k]
    return f"C2a {rus(r['look1']['A']['c2a'])} / {rus(r['look2']['A']['c2a'])}"


def av(k, lk="look1"):
    return rus(RUNS[k][lk]["A"]["view"])


def bv(k, lk="look1"):
    return pct(RUNS[k][lk]["B"]["view"])


CANDS = []
CANDS.append(dict(
    id="M1", name="Chronos-2", kind="готовая числовая модель рядов (120 млн параметров), ковариаты", status="nogain",
    img="M1", imgcap="BTC/USDT, день 2026-03-20 (UTC): часовая волатильность — Chronos-2 (медиана и q10–q90), прогноз V3 и реализованное значение.",
    concl=f"Прогноз Chronos-2 добавляет к V3 {av('M1')} / {av('M1','look2')} (цель A), но те же числа S3 с нелинейным преобразованием (C2a) дают {rus(RUNS['M1']['look1']['A']['c2a'])} / {rus(RUNS['M1']['look2']['A']['c2a'])}.",
    done=f"Сделан один шаг улучшения: M1b, прогноз HAR подан как ковариата, {av('M1b')} / {av('M1b','look2')}: контроль C2a всё ещё сильнее.",
    nxt="Следующий шаг (предложение): подать модели как ковариату суточный профиль «час × выходной» — именно он несёт выигрыш M7–M9 (раздел 3); без него ей нечего добавить к V3.",
    runs=[run_js("M1", "как задано", False), run_js("M1b", "шаг улучшения: HAR как ковариата", True)]))
CANDS.append(dict(
    id="M2", name="TTM r2", kind="малая модель IBM (805 тыс. параметров), среднее 50/50 с HAR", status="nogain",
    img="M2", imgcap="BTC/USDT, день 2026-03-20 (UTC): прогноз TTM против V3 и реализованной волатильности.",
    concl=f"Из коробки TTM даёт {av('M2')} / {av('M2','look2')}; после дообучения (M2b) {av('M2b')} / {av('M2b','look2')}: на взгляде 1 это выше C2a, на взгляде 2 нет.",
    done="Шаг улучшения выполнен: M2b, дообучение на первых месяцах (после факта, отмечено).",
    nxt="Следующий шаг (предложение): дообучать TTM только после того, как профиль «час × выходной» добавлен во вход: тогда видно, что остаётся на долю модели.",
    runs=[run_js("M2", "как задано", False), run_js("M2b", "шаг улучшения: дообучение", True)]))
if not M3_PENDING:
    m3 = RUNS["M3"]
    CANDS.append(dict(
        id="M3", name="Kronos", kind="язык свечей: сэмплированные пути OHLC (N = 4, Kronos-small)", status="nogain",
        img="M3" if "M3" in IMGS else None, imgcap="BTC/USDT: пример сэмплированных путей Kronos и реализованная цена.",
        concl=f"Диапазон, волатильность и P(касания) из четырёх путей Kronos дают {av('M3')} / {av('M3','look2')} (цель A) и {bv('M3')} / {bv('M3','look2')} (цель B): прироста над C2a нет.",
        done="Результат предрегистрирован с N = 4 путями; варианта улучшения в этом прогоне нет.",
        nxt="Следующий шаг (предложение): больше путей на якорь (десятки вместо четырёх), чтобы оценка P(касания) перестала быть грубой; запускать после профиля, по стоимости прогона.",
        runs=[run_js("M3", "как задано (N = 4 пути)", False)]))
else:
    CANDS.append(dict(
        id="M3", name="Kronos", kind="язык свечей: сэмплированные пути OHLC", status="wip", img=None, imgcap="",
        concl="Результат Kronos ещё в расчёте: отчёт перестраивается этим же скриптом, когда файлы M3 появятся в ветке.", done="", nxt="", runs=[]))
CANDS.append(dict(
    id="M4", name="Карточка режима: картинка и подпись", kind="свечной график 96×96 → CLIP; шаблонная текстовая подпись режима", status="near",
    img="M4", imgcap="Карточки режима: свечи и объём, к ним шаблонная подпись.",
    concl=f"Картинка сама ничего не добавила ({av('M4_img')} / {av('M4_img','look2')}). Подпись проходит критерий на взгляде 1 на обеих целях (A {av('M4_cap')}, B {bv('M4_cap')}), на взгляде 2 нет (A {av('M4_cap','look2')}, B {bv('M4_cap','look2')}).",
    done=f"Шаг улучшения: M4c, те же числа подписи без текста, дали на взгляде 1 цель A {av('M4c_capnum')}: значит, работает содержание чисел, а не слова.",
    nxt="Следующий шаг (предложение): подавать содержание подписи напрямую как профиль «час × выходной» и |растяжение|, без CLIP.",
    runs=[run_js("M4_img", "картинка", False), run_js("M4_cap", "подпись", False), run_js("M4_both", "картинка + подпись", False),
          run_js("M4c_capnum", "шаг улучшения: числа подписи без текста", True), run_js("M4b_img_gbm", "шаг улучшения: картинка в GBM", True)]))
CANDS.append(dict(
    id="M5", name="VisionTS", kind="сезонный профиль, нарисованный как картинка (MAE, период 288)", status="nogain",
    img="M5", imgcap="BTC/USDT: контекст в виде картинки и прогноз VisionTS.",
    concl=f"Прогноз VisionTS даёт {av('M5')} / {av('M5','look2')} (цель A); вариант M5b с сезонным профилем {av('M5b')} / {av('M5b','look2')}.",
    done="Шаг улучшения выполнен: M5b, сезонный профиль без картинки.",
    nxt="Следующий шаг (предложение): вернуться к VisionTS, когда суточный профиль «час × выходной» станет частью входа; сама картинка профиль не восстанавливает.",
    runs=[run_js("M5", "как задано", False), run_js("M5b", "шаг улучшения: сезонный профиль", True)]))
CANDS.append(dict(
    id="M6", name="Поиск аналогов (FAISS)", kind="среднее остатков HAR у k ближайших прошлых окон", status="works",
    img="M6", imgcap="BTC/USDT: окно и найденные аналоги из прошлого.",
    concl=f"Аналоги дают {av('M6')} / {av('M6','look2')} (цель A): выше C1 и C2a на обоих взглядах — первый вид, который подсказал нелинейность. Предрегистрация не пройдена из-за второго контроля C2b.",
    done=f"Шаг улучшения: M6b, остатки HAR у соседей, k = 50: A {av('M6b_k50')} / {av('M6b_k50','look2')}, цель A проходит на обоих взглядах, выше C2b.",
    nxt="Следующий шаг: передать подсказку M7 (нелинейность на режимных ключах), что и сделано (раздел 3).",
    runs=[run_js("M6", "как задано (k = 50)", False), run_js("M6b_k50", "шаг улучшения: остатки HAR, k = 50", True)]))

# ------------------------------------------------------------------------------------------------ tables for the 'Детали' blocks
def fl(b):
    return "да" if b else "нет"


def run_rows(keys, labels=None):
    rows = []
    for k in keys:
        r = RUNS[k]
        if r is None:
            continue
        lab = (labels or {}).get(k, k)
        for lk, ln in (("look1", "1"), ("look2", "2")):
            A, B = r[lk]["A"], r[lk]["B"]
            rows.append(f"<tr><td>{lab}</td><td>{ln}</td><td>{rus(A['view'])} [{rus(A['lo'])}; {rus(A['hi'])}]</td><td>{A['p']:.4f}</td><td>{rus(A['c1'])}</td><td>{rus(A['c2a'])}</td>"
                        f"<td>{fl(A['pass'])}</td><td>{pct(B['view'])} [{pct(B['lo'])}; {pct(B['hi'])}]</td><td>{B['p']:.4f}</td><td>{pct(B['c1'])}</td><td>{pct(B['c2a'])}</td><td>{fl(B['pass'])}</td></tr>")
    return rows


TH = ("<table class=\"rt\"><tr><th>вариант</th><th>взгляд</th><th>A: прирост к V3 [95% CI]</th><th>p (одност.)</th><th>C1</th><th>C2a</th><th>прошёл</th>"
      "<th>B: Брайер к H3 [95% CI]</th><th>p (одност.)</th><th>C1</th><th>C2a</th><th>прошёл</th></tr>")


def table(keys, labels=None):
    return TH + "".join(run_rows(keys, labels)) + "</table>"


def cand_html(c):
    chip = f'<span class="chip st-{c["status"]}">{STAT[c["status"]]}</span>'
    seg = ""
    if len(c["runs"]) > 1:
        seg = '<div class="seg" data-cand="%s">' % c["id"] + "".join(
            f'<button class="{"on" if i == 0 else ""}" data-i="{i}">{r["label"]}{" (после факта)" if r["post"] else ""}</button>' for i, r in enumerate(c["runs"])) + "</div>"
    img = ""
    if c.get("img") and c["img"] in IMGS:
        img = f'<figure class="shot"><img alt="{c["id"]}" src="{IMGS[c["img"]]}"><figcaption>{c["imgcap"]}</figcaption></figure>'
    det = ""
    if c["runs"]:
        det = (f'<details><summary>Детали: все числа {c["id"]}</summary>' +
               table([r["key"] for r in c["runs"]], {r["key"]: r["label"] for r in c["runs"]}) +
               '<p class="note">A: прирост месячного скилла над V3 (t-интервал по месяцам), B: относительное улучшение Брайера над H3 (интервал по дням-блокам). C1: те же столбцы, перемешанные внутри месяца; '
               'C2a: числа S3 того же окна плюс монотонное нелинейное преобразование. Взгляд 1: 2024-07..2025-07 (разрезанный файл), взгляд 2: 2025-11..2026-09 (файл Binance).</p></details>')
    nx = f'<p class="nx"><b>Что сделано.</b> {c["done"]}</p><p class="nx"><b>{c["nxt"].split(":")[0]}:</b>{c["nxt"][len(c["nxt"].split(":")[0]) + 1:]}</p>' if c["done"] else ""
    return (f'<div class="card vcard" id="v_{c["id"]}"><div class="tagrow"><span class="chip dim">{c["id"]}</span>{chip}</div><h3>{c["name"]}</h3>'
            f'<div class="kind">{c["kind"]}</div><p class="concl">{c["concl"]}</p>'
            f'{seg}<div class="pair"><svg class="vch" id="vch_{c["id"]}A" viewBox="0 0 330 250"></svg><svg class="vch" id="vch_{c["id"]}B" viewBox="0 0 330 250"></svg></div>{nx}{img}{det}</div>')


# ------------------------------------------------------------------------------------------------ concept map
def concept_nodes():
    m = RUNS
    nodes = [
        dict(id="found", label="Готовые модели рядов", sub="Chronos-2 · TTM · Kronos", status="nogain",
             card=f"Chronos-2 {av('M1')} / {av('M1','look2')}; TTM {av('M2')} / {av('M2','look2')} (после дообучения {av('M2b')} / {av('M2b','look2')}); "
                  + (f"Kronos {av('M3')} / {av('M3','look2')}" if not M3_PENDING else "Kronos в расчёте")
                  + f" (прирост к V3, цель A, взгляд 1 / 2). Контроль C2a: {rus(m['M1']['look1']['A']['c2a'])} / {rus(m['M1']['look2']['A']['c2a'])}.",
             nxt="Подать профиль «час × выходной» как вход, затем дообучать.", link="v_M1"),
        dict(id="image", label="Картинка", sub="CLIP · VisionTS", status="nogain",
             card=f"Свечной график как картинка: CLIP {av('M4_img')} / {av('M4_img','look2')}; VisionTS {av('M5')} / {av('M5','look2')}. Картинка — функция тех же OHLCV, новой информации в ней нет.",
             nxt="Картинка пригодится как способ показать профиль, не как источник сигнала.", link="v_M4"),
        dict(id="text", label="Подпись режима", sub="шаблонный текст", status="near",
             card=f"Подпись проходит критерий на взгляде 1 на обеих целях (A {av('M4_cap')}, B {bv('M4_cap')}); на взгляде 2 A {av('M4_cap','look2')}, B {bv('M4_cap','look2')}. Числа без текста (M4c) дают A {av('M4c_capnum')} на взгляде 1.",
             nxt="Подавать содержание подписи напрямую числами.", link="v_M4"),
        dict(id="freq", label="Частота", sub="вейвлеты", status="wip",
             card="Вейвлет-энергии полос как масштабные индикаторы рядом с HAR. В этом спринте не запускались.",
             nxt="Тест с тем же набором контролей: числа той же полосы без вейвлета.", link=None),
        dict(id="graph", label="Граф", sub="visibility-граф окна", status="wip",
             card="Статистики графа видимости 60 баров рядом с линиями поддержки T13. В этом спринте не запускались.",
             nxt="Тест против C2a; граф — функция того же окна, ждём нелинейного выигрыша, не информации.", link=None),
        dict(id="retr", label="Память аналогов", sub="FAISS · M6", status="works",
             card=f"Среднее остатков HAR у k = 50 соседей (M6b): A {av('M6b_k50')} / {av('M6b_k50','look2')}, цель A проходит на обоих взглядах и выше C2b; B не проходит порог 1% на взгляде 2.",
             nxt="Проверить, добавляет ли ретривал что-то к профилю M9.", link="v_M6"),
        dict(id="gbm", label="Табличный GBM", sub="12 ключей · M7", status="works",
             card=f"GBM на 12 режимных ключах проходит обе цели на обоих взглядах: A {av('M7')} / {av('M7','look2')}, B {bv('M7')} / {bv('M7','look2')}.",
             nxt="Заморожен как M7c (раздел 3), проверка вперёд.", link="worked"),
        dict(id="fusion", label="Слияние: модуль профиля", sub="профиль + |растяжение| · M9", status="near",
             card=f"M9h6 (29 столбцов, ridge и логистика) восстанавливает {R9['views']['M9h6_prof_str']['A']['R']:.2f} выигрыша M7 по A и {R9['views']['M9h6_prof_str']['B']['R']:.2f} по B. "
                  "Заранее зарегистрированные виды M9 не прошли оба взгляда; M9h6 — кандидат после факта, его судит только проверка вперёд.",
             nxt="Заморожен (sha e065e6b7…), ждёт 3 и 6 месяцев данных.", link="worked"),
        dict(id="newdata", label="Новые данные", sub="funding · OI · стакан · новости", status="newdata",
             card="Funding и открытый интерес — бесплатные архивы data.binance.vision (пилот 1–3 дня, около 20 ГБ); стакан — владелец должен сказать, считается ли это тиковым стаканом (D-044); новости и ончейн — ограничения API и риск заглядывания вперёд.",
             nxt="Решение владельца по D-050 (варианты A: funding/OI, B: стакан, C: новости).", link="next"),
    ]
    return nodes


# ------------------------------------------------------------------------------------------------ the install inventory
def install_data():
    lines = [ln.split(None, 2) for ln in open(TASK / "weights_sha256.txt", encoding="utf-8") if ln.strip()]
    W = {}
    for sha, size, path in lines:
        W[path] = (sha, int(size))
    mm = open(TASK / "versions_mm_env.txt", encoding="utf-8").read()
    ml = open(TASK / "versions_mm_legacy.txt", encoding="utf-8").read()
    sm = open(TASK / "smoke.md", encoding="utf-8").read()
    for s in ("torch==2.11.0+cu126", "chronos-forecasting==2.3.2", "granite-tsfm==0.3.10", "open_clip_torch==3.3.0", "visionts==1.0.1", "faiss-cpu==1.15.1", "lightgbm==4.7.0"):
        assert s in mm, s
    assert "torch==2.5.1+cu124" in ml
    for s in ("mm_env 5.0 GB, mm_legacy 4.8 GB, mm_cache 6.7 GB", "D:/nt/nt_mm_env", "D:/nt/nt_mm_legacy"):
        assert s in sm, s
    P = "D:/nt/neural_trade/runs/tactical/tasks/T14_multimodal"
    rows = [
        ("M1 Chronos-2", "mm_env", "chronos-forecasting 2.3.2, transformers 5.20.1", "amazon/chronos-2, 478 МБ", "ddcda3c7", f"mm_env/python.exe {P}/M1_chronos2/load.py"),
        ("M2 TTM r2", "mm_env", "granite-tsfm 0.3.10", "ibm-granite/granite-timeseries-ttm-r2, 3,2 МБ (две ветки)", "b3cbaf0a · a706726a", f"mm_env/python.exe {P}/M2_ttm/load.py"),
        ("M3 Kronos-small", "mm_legacy", "torch 2.5.1+cu124, pandas 2.2.2, клон Kronos 67b630e", "NeoQuasar/Kronos-small, 99 МБ + токенайзер 16 МБ", "b082dfcb · 59d85f6a", f"mm_legacy/python.exe {P}/M3_kronos/load.py"),
        ("M4 CLIP", "mm_env", "open_clip_torch 3.3.0, mplfinance 0.12.10b0, lightgbm 4.7.0", "ViT-B-32 OpenAI, 605 МБ", "e6d1bd77", f"mm_env/python.exe {P}/M4_regime_card/load.py"),
        ("M5 VisionTS", "mm_env", "visionts 1.0.1 (+ requests)", "MAE base ImageNet, 448 МБ", "c551c37b", f"mm_env/python.exe {P}/M5_visionts/load.py"),
        ("M6 FAISS", "mm_env", "faiss-cpu 1.15.1", "весов нет (CPU)", "—", f"mm_env/python.exe {P}/M6_faiss/load.py"),
    ]
    # the sha prefixes in the table must be the prefixes of the real hashes
    shas = {v[0][:8] for v in W.values()}
    for r in rows:
        for pfx in re.findall(r"[0-9a-f]{8}", r[4]):
            assert pfx in shas, (r[0], pfx)
    return rows


INSTALL = install_data()

# ------------------------------------------------------------------------------------------------ prospective bets
def timeline_rows():
    rows = [
        ("T04-B", "15-минутный календарь для прогноза изменения волатильности", "61749ad6", "tasks/T04_vol_calendar/REPORT.md", "2026-10-01", "2027-01-01", "3 полных месяца с 2026-10"),
        ("T05-B", "Интервалы 60 мин: HAR, EWMA и квантильная регрессия", "e9c38d61", "tasks/T05_interval_models/frozen_B.sha256", "2026-10-01", "2027-01-01", "3 полных месяца с 2026-10"),
        ("T11", "UTC-календарь V3 (96 слотов): V3 − V0 не меньше +0,01", "6d62faaf", "tasks/T11_utc_seasonality/REPORT.md", "2026-10-02", "2027-01-01", "оценка по 3 и по 6 месяцам (Pocock)"),
        ("T12", "Барьерная вероятность H3 + изотоника месяца", "382555ce", "tasks/T12_barrier_calibrated/frozen_B.sha256", "2026-10-02", "2027-01-01", "3 полных месяца с 2026-10-02"),
        ("T13", "Линии поддержки и сопротивления: A1, B1, C1", "8fec9bf3", "tasks/T13_support_lines/frozen_B.sha256", "2026-10-02", "2027-01-01", "критерии по частям, 3 полных месяца"),
        ("O4", "BANK: плотный HAR против базы, H = 60", "83dd3ad7", "outlook/O4_bank_confirm/frozen_B.sha256", "2026-10-01", "2027-01-01", "3 полных месяца с 2026-10"),
        ("O5", "Геометрия L1 над лучшей базой L0, H = 60", "79f7f25d", "outlook/O5_geom_bestbase/frozen_B.sha256", "2026-10-01", "2027-01-01", "3 из 3 месяцев, прирост не меньше +0,003"),
        ("O7", "Интервалы с геометрией на заморозке O5, калибровка 2026-10", "79f7f25d", "outlook/O7_geom_intervals/REPORT.md", "2026-11-01", "2027-02-01", "оценка 2026-11..2027-01"),
        ("O9", "Кандидат ROBUST: пробой + регрессионный гейт", "9c5259a6", "outlook/O9_ensemble_strategy/frozen_P.sha256", "2026-11-01", "2027-02-01", "оценка 2026-11..2027-01"),
        ("O10", "Выходы по геометрии и волатильности", "3f87ef09", "outlook/O10_geomvol_exits/frozen_P.sha256", "2026-11-01", "2027-02-01", "оценка 2026-11..2027-01"),
        ("O11a", "Геометрия + банк для изменения волатильности", "e2c7ac48", "outlook/O11_accuracy_batch/part_a/frozen_P_a.sha256", "2026-11-01", "2027-02-01", "прирост не меньше +0,010, 3 из 3"),
        ("O11b", "Помесячное переобучение для уровня волатильности", "7219a428", "outlook/O11_accuracy_batch/part_b/frozen_P_b.sha256", "2026-11-01", "2027-02-01", "прирост не меньше +0,003, 3 из 3"),
    ]
    out, unchecked = [], []
    for tid, what, sha, src, a, b, crit in rows:
        base = OUTL if src.startswith("outlook/") else TAC
        p = base / src
        if not p.exists():
            unchecked.append(tid)       # the outlook worktree is optional; the shas below were checked when it was present
        else:
            txt = open(p, encoding="utf-8").read()
            assert (tid == "O7" and "79f7f25d" in open(OUTL / "outlook/O5_geom_bestbase/REPORT.md", encoding="utf-8").read()) or txt.strip().startswith(sha) or sha in txt, (tid, sha)
        out.append({"id": tid, "what": what, "sha": sha, "src": src, "a": a, "b": b, "crit": crit, "group": "earlier"})
    gA, gB = F8["freeze"]["files"]["gbm_A.txt"]["sha256"], F8["freeze"]["files"]["gbm_B.txt"]["sha256"]
    out.append({"id": "M8 (M7c)", "what": "Модуль M7c_gbm_resid: GBM на остатках V3 и H3, 12 ключей; заморожен правилом M8", "sha": f"{gA[:8]} · {gB[:8]}",
                "src": "tasks/T14_multimodal/M8_module/frozen_P.json", "a": "2026-10-01", "b": "2027-01-01",
                "crit": "A не меньше +0,005 и p < 0,0294; B не меньше 1%; взгляды на 3 и на 6 месяцев (Pocock)", "group": "new"})
    out.append({"id": "M9h6", "what": "Кандидат после факта: профиль «час × выходной» плюс |растяжение|, 29 столбцов", "sha": F9["recipe_sha256"][:8],
                "src": "tasks/T14_multimodal/M9_module/frozen_P_M9h6_prof_str.json", "a": "2026-10-01", "b": "2027-01-01",
                "crit": "тот же критерий; судья только проверка вперёд", "group": "new"})
    return out, unchecked


TL, TL_UNCHECKED = timeline_rows()

# ------------------------------------------------------------------------------------------------ data blocks for the page
def story():
    m6, m7, m7c = RUNS["M6"], RUNS["M7"], RUNS["M7c"]
    ab = R8["ablation"]
    grp = ["hourwk", "stretch", "volpct", "lrv", "sr"]
    names = {"hourwk": "час × выходной", "stretch": "|растяжение|", "volpct": "перцентиль волатильности", "lrv": "лог-RV последних часов", "sr": "расстояние до S/R"}
    abl = {t: [{"g": g, "name": names[g], "share": ab[t]["groups"][g]["share"], "drop": ab[t]["groups"][g]["drop"]["mean"], "lo": ab[t]["groups"][g]["drop"]["lo"],
                "hi": ab[t]["groups"][g]["drop"]["hi"], "carries": ab[t]["groups"][g]["carries"]} for g in grp] for t in "AB"}
    rec_keys = [("M7c_gbm_resid", "M7c GBM на остатках"), ("M8_lin", "M8 линейный"), ("M8x_lin_wk", "M8x лин. + выходной"), ("M9_prof", "M9 профиль"),
                ("M9_prof_str", "M9 профиль + |растяжение|"), ("M9_prof_str_vp", "M9 + перцентиль vol"), ("M9_gbm7", "M9 GBM на 7 ключах"), ("M9h6_prof_str", "M9h6 (кандидат)")]
    rec = [{"key": k, "name": n, "RA": R9["views"][k]["A"]["R"], "RB": R9["views"][k]["B"]["R"],
            "passA": R9["views"][k]["A"]["pass_both_looks"], "passB": R9["views"][k]["B"]["pass_both_looks"]} for k, n in rec_keys]
    prof = {k: PROF[k] for k in ("hours", "A_view", "A_base", "A_real", "B_view", "B_base", "B_real", "n_by_hour")}
    prof["pool"] = {"first": PROF["pool"]["first_pool_ta"], "last": PROF["pool"]["last_pool_ta"], "n": PROF["pool"]["n_pool"]}
    return {"abl": abl, "rec": rec, "prof": prof,
            "m6": {k: {lk: RUNS[k][lk] for lk in ("look1", "look2")} for k in ("M6", "M6b_k50")},
            "m6c2b": [RUNS["M6"][lk]["A"]["c2b"] for lk in ("look1", "look2")],
            "m7": {lk: RUNS["M7"][lk] for lk in ("look1", "look2")}, "m7c": {lk: RUNS["M7c"][lk] for lk in ("look1", "look2")},
            "m8": {"carriersA": ab["A"]["main_carrier"], "R_lin_A": R8["linear"]["A"]["R"], "R_lin_B": R8["linear"]["B"]["R"]}}


STORY = story()

CONTROL = [{"k": k, "name": {"M1": "Chronos-2", "M1b": "Chronos-2 + HAR", "M2": "TTM", "M2b": "TTM дообуч.", "M3": "Kronos", "M5": "VisionTS", "M5b": "VisionTS сезонн.",
                             "M4_img": "CLIP-картинка", "M4b_img_gbm": "картинка в GBM"}[k],
            "look1": RUNS[k]["look1"], "look2": RUNS[k]["look2"]} for k in FM]

HERO = [{"k": k, "name": n, "look1": RUNS[k]["look1"]["A"]["view"], "look2": RUNS[k]["look2"]["A"]["view"], "c2a1": RUNS[k]["look1"]["A"]["c2a"], "c2a2": RUNS[k]["look2"]["A"]["c2a"],
         "st": st} for k, n, st in [("M1", "Chronos-2", "nogain"), ("M2", "TTM", "nogain"), ("M5", "VisionTS", "nogain"), ("M4_img", "CLIP-картинка", "nogain"),
                                    ("M4_cap", "подпись режима", "near"), ("M6b_k50", "аналоги FAISS (M6b)", "works"), ("M7", "GBM на 12 ключах (M7)", "works"),
                                    ("M7c", "M7c, остатки (заморожен)", "works"), ("M9h6", "M9h6, профиль (кандидат)", "near")] + ([("M3", "Kronos", "nogain")] if not M3_PENDING else [])]

FROZEN_N = len(TL)
n_pre_fail = sum(1 for k in ("M1", "M2", "M5", "M4_img", "M4_cap", "M4_both") if not RUNS[k]["pass"]["candidate"]) + (0 if M3_PENDING else int(not RUNS["M3"]["pass"]["candidate"]))
N_PRE = 6 + (0 if M3_PENDING else 1)
# the six pre-registered views: M1, M2, M3, M4 (ladder: three variants), M5, M6; the honest count is of the ready-made-model / image candidates
FOUND_PASS = 0   # none of M1, M2, M3, M4 (any variant), M5 passed
chk_true("none of the ready-made models and image views passed the pre-registered criterion", all(not RUNS[k]["pass"]["candidate"] for k in ("M1", "M2", "M5", "M4_img", "M4_both") + (() if M3_PENDING else ("M3",))))
FM_N = 5 - (1 if M3_PENDING else 0)

bad = [c for c in CHECKS if not c[4]]
if bad:
    for b in bad:
        print("CHECK FAILED:", b)
    sys.exit("build failed: %d number checks did not match the result files" % len(bad))
print(f"{len(CHECKS)} number checks passed")

D = {"pal": PALETTE, "cands": [{"id": c["id"], "status": c["status"], "runs": c["runs"]} for c in CANDS if c["runs"]], "nodes": concept_nodes(), "story": STORY, "control": CONTROL,
     "controlN": {"beat": C2A_BEATEN, "tot": n_tot}, "hero": HERO, "timeline": TL, "stat": STAT, "m3pending": M3_PENDING}


# ------------------------------------------------------------------------------------------------ the page
def appendix_html():
    keys = [k for k in ["M1", "M1b", "M2", "M2b", "M3", "M4_img", "M4_cap", "M4_both", "M4c_capnum", "M4b_img_gbm", "M5", "M5b", "M6", "M6b_k20", "M6b_k50", "M6b_k100", "M6x_labels",
                        "M6x_keys", "M6x_dist", "M7", "M7b", "M7c", "M7s", "M8_abl_stretch", "M8_abl_volpct", "M8_abl_hourwk", "M8_abl_sr", "M8_abl_lrv", "M8_lin", "M8x_abl_hour",
                        "M8x_abl_wkend", "M8x_lin_wk", "M9_prof", "M9_prof_str", "M9_prof_str_vp", "M9_gbm7", "M9h3", "M9h6", "M9h6_vp"] if RUNS.get(k)]
    lab = {"M6b_k20": "M6b k=20 (после факта)", "M6b_k50": "M6b k=50 (после факта)", "M6b_k100": "M6b k=100 (после факта)", "M6x_labels": "M6x метки (контроль)",
           "M6x_keys": "M6x ключи (контроль)", "M6x_dist": "M6x расстояния (контроль)", "M7s": "M7s (только 2026-06..09)", "M7b": "M7b (GBM + V3)", "M7c": "M7c (GBM на остатках)",
           "M8_abl_stretch": "M8 без |растяжения|", "M8_abl_volpct": "M8 без перцентиля vol", "M8_abl_hourwk": "M8 без час × выходной", "M8_abl_sr": "M8 без S/R",
           "M8_abl_lrv": "M8 без лог-RV часов", "M8_lin": "M8 линейный", "M8x_abl_hour": "M8x без часа", "M8x_abl_wkend": "M8x без выходного", "M8x_lin_wk": "M8x лин. + выходной",
           "M9h3": "M9h3 (после факта)", "M9h6": "M9h6 (после факта)", "M9h6_vp": "M9h6 + vol (после факта)", "M9_gbm7": "M9 GBM на 7 ключах"}
    return table(keys, lab)


def rec_table():
    rows = []
    for k, n in [("M9_prof", "M9_prof"), ("M9_prof_str", "M9_prof_str"), ("M9_prof_str_vp", "M9_prof_str_vp"), ("M9_gbm7", "M9_gbm7"), ("M9h3_prof_str", "M9h3_prof_str (после факта)"),
                 ("M9h6_prof_str", "M9h6_prof_str (после факта)"), ("M9h6_prof_str_vp", "M9h6_prof_str_vp (после факта)")]:
        v = R9["views"][k]
        rows.append(f"<tr><td>{n}</td><td>{v['A']['R']:.2f}</td><td>{v['B']['R']:.2f}</td><td>{fl(v['A']['pass_both_looks'])}</td><td>{fl(v['B']['pass_both_looks'])}</td></tr>")
    return ("<table class=\"rt\"><tr><th>вид</th><th>R по A</th><th>R по B</th><th>A прошёл оба взгляда</th><th>B прошёл оба взгляда</th></tr>" + "".join(rows) + "</table>")


def abl_table():
    rows = []
    for t, tn in (("A", "A (прирост скилла)"), ("B", "B (Брайер)")):
        for g, v in R8["ablation"][t]["groups"].items():
            d = v["drop"]
            fmt = (lambda x: rus(x)) if t == "A" else (lambda x: pct(x))
            rows.append(f"<tr><td>{tn}</td><td>{g}</td><td>{fmt(d['mean'])} [{fmt(d['lo'])}; {fmt(d['hi'])}]</td><td>{v['share'] * 100:.0f}%</td><td>{fl(v['carries'])}</td></tr>")
    return "<table class=\"rt\"><tr><th>цель</th><th>группа убрана</th><th>потеря к полному M7, 24 месяца [95% CI]</th><th>доля</th><th>несёт (доля не меньше 25% и CI выше 0)</th></tr>" + "".join(rows) + "</table>"


def install_html():
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td><code>{e}</code></td><td><code>{f}</code></td></tr>" for a, b, c, d, e, f in INSTALL)
    return ("<table class=\"rt\"><tr><th>что</th><th>среда</th><th>версии</th><th>веса</th><th>sha256, начало</th><th>как запустить (HF_HOME=D:/nt/nt_mm_cache/hf)</th></tr>" + rows + "</table>")


def tl_names():
    return ""


def page():
    css = (POS / "pos.css").read_text(encoding="utf-8") + "\n" + (ASSETS / "mm.css").read_text(encoding="utf-8")
    for k, v in PALETTE.items():
        css = css.replace("{{" + k + "}}", v)
    css = css.replace("{{font}}", T.FONT)
    pos_js = (POS / "pos.js").read_text(encoding="utf-8")
    i0 = pos_js.index("(()=>{") + len("(()=>{")
    i1 = pos_js.index("// ---------------------------------------------------------------- sparklines")
    prelude = pos_js[i0:i1]         # the helper prelude of the positive report: el, run, once, nav highlight, counters
    js = prelude + (ASSETS / "mm.js").read_text(encoding="utf-8")
    cards = "".join(cand_html(c) for c in CANDS)
    m3note = ("<p class=\"note\">Kronos: результат прочитан из " + M3_SRC + ".</p>") if not M3_PENDING else "<p class=\"note\">Kronos ещё в расчёте.</p>"
    nA = len(CANDS)
    m7 = RUNS["M7"]
    tiles = (
        f'<div class="kpi st-nogain"><div class="kv"><span class="cnt" data-to="0" data-dec="0">0</span> из {FM_N}</div><div class="kl">готовых моделей и картинок прошли заранее записанный критерий: '
        f'контроль C2a не обыгран (C2a выше вида в {C2A_BEATEN} из {n_tot} сравнений)</div></div>'
        f'<div class="kpi"><div class="kv">{rus(m7["look1"]["A"]["view"])} / {rus(m7["look2"]["A"]["view"])}</div><div class="kl">M7, GBM на 12 режимных ключах: прирост к V3 по цели A на взглядах 1 и 2; '
        f'по B {pct(m7["look1"]["B"]["view"])} / {pct(m7["look2"]["B"]["view"])}; обе цели, оба взгляда пройдены</div></div>'
        f'<div class="kpi"><div class="kv"><span class="cnt" data-to="{R8["ablation"]["B"]["groups"]["hourwk"]["share"] * 100:.0f}" data-dec="0" data-suf="%">92%</span></div>'
        f'<div class="kl">выигрыша по B несёт «час × выходной» (по A {R8["ablation"]["A"]["groups"]["hourwk"]["share"] * 100:.0f}%; ещё |растяжение| {R8["ablation"]["A"]["groups"]["stretch"]["share"] * 100:.0f}%)</div></div>'
        f'<div class="kpi"><div class="kv">{rus(R9["views"]["M9h6_prof_str"]["A"]["R"], 2, False)} / {rus(R9["views"]["M9h6_prof_str"]["B"]["R"], 2, False)}</div><div class="kl">доля выигрыша M7, которую возвращает компактный модуль M9h6 '
        f'(по A / по B; кандидат после факта)</div></div>'
        f'<div class="kpi"><div class="kv"><span class="cnt" data-to="{FROZEN_N}" data-dec="0">{FROZEN_N}</span></div><div class="kl">ставок заморожено до данных; первые вердикты в январе–феврале 2027</div></div>')
    next_cards = [
        ("1", "Профиль «час × выходной» как продукт для H3", f"Модуль M9h6 уже заморожен (sha {F9['recipe_sha256'][:8]}…). Вернул {R9['views']['M9h6_prof_str']['A']['R']:.2f} и {R9['views']['M9h6_prof_str']['B']['R']:.2f} выигрыша M7. Следующий шаг: прогнать его в H3 как штатный вход и сравнить с проспективной оценкой.", "M9_module/profile_M9h6_prof_str.json"),
        ("2", "Дообучение TTM и VisionTS — после профиля", f"M2b даёт {av('M2b')} / {av('M2b','look2')}: есть что дообучать. Следующий шаг: добавить профиль во вход и повторить M2b и M5b; без профиля дообучение упирается в C2a.", "M2b_ttm_ft, M5b_visionts_seasonal"),
        ("3", "Kronos: больше путей на якорь", ("Следующий шаг после результата M3: десятки путей вместо четырёх, P(касания) из путей против H3." if not M3_PENDING else "Решать после результата M3 (в расчёте).") + " Только если стоимость прогона оправдана.", "M3_kronos"),
        ("4", "|Растяжение| как отдельный индикатор", f"Без него A теряет {R8['ablation']['A']['groups']['stretch']['share'] * 100:.0f}% выигрыша. Следующий шаг: добавить в каталог индикаторов нейросети как обучаемый признак.", "M8_module/tables_m8.txt"),
        ("5", "Память аналогов поверх профиля", f"M6b даёт A {av('M6b_k50')} / {av('M6b_k50','look2')} отдельно. Следующий шаг: проверить, остаётся ли у ретривала прирост над M9h6.", "M6b_faiss_resid_k50"),
        ("6", "Частота и граф", "Вейвлет-энергии и граф видимости не запускались. Следующий шаг: тест с теми же контролями (C1, C2a).", "research/2026-10-17-multimodal"),
        ("7", "Новые данные: funding, открытый интерес, стакан, новости", "Нужно решение владельца по D-050. Funding и открытый интерес: бесплатные архивы data.binance.vision, пилот 1–3 дня, около 20 ГБ. Стакан (bookDepth): сначала сказать, считается ли это тиковым стаканом (D-044). Новости: ограничения API, риск заглядывания вперёд.", "research/2026-10-17-multimodal, раздел 5"),
    ]
    nx = "".join(f'<div class="card nx"><div class="ico">{n}</div><h3>{t}</h3><p>{x}</p><div class="ev">{e}</div></div>' for n, t, x, e in next_cards)
    dev = [
        "Взгляд 1 (2024-07..2025-07, разрезанный файл): чистый для Kronos после 2024-07; TTM в предобучении перечисляет Bitcoin; корпуса остальных моделей не раскрыты. Взгляд 2 (2025-11..2026-09, файл Binance) чист для всех. Гиперпараметры слияния выбирались на 2019-01..2024-06 и там ничего не судилось.",
        "Предрегистрированные виды M9 (M9_prof, M9_prof_str, M9_prof_str_vp) не прошли оба взгляда ни по одной цели; правило заморозки ничего не заморозило. На взгляде 2 мало обучающих строк в первые месяцы Binance (ноябрь, декабрь 2025): без них (описательная таблица чувствительности, без флагов) R по A у M9h6 равен "
        f"{SENS_RA:.2f}. M9h6 выбран после факта, его судит только проверка вперёд.",
        "Шаги улучшения (M1b, M2b, M4b, M4c, M5b, M6b, M6x, M8x, M9x, M9h*) выполнены после факта и помечены; в предрегистрированные вердикты не входят.",
        "M7s: тот же GBM, но только на 2026-06..09 (малый хвост) — не прошёл; M8x: ablation и линейные варианты после факта.",
        "Holm по 6 × 2 считает ведущий над односторонними p; здесь показаны односторонние p без поправки.",
        "Профиль на часах — подогнанные средние итоговой модели на пуле Binance 2025-10-16..2026-09-30 (внутри выборки); точки — реализованные средние.",
    ]
    dev_html = "".join(f"<li>{x}</li>" for x in dev)
    tl_n_new = sum(1 for r in TL if r["group"] == "new")
    body = f"""<nav id="nav">
<h1>Мультимодальная ночь</h1>
<div class="sub">neural_trade · T14 · 2026-10-17<br>утренний отчёт</div>
<a href="#top">Итог ночи</a>
<a href="#map">1 · Карта концепций</a>
<a href="#views">2 · Шесть взглядов</a>
<a href="#worked">3 · Что сработало</a>
<a href="#controls">4 · Контроли</a>
<a href="#install">5 · Что установлено</a>
<a href="#bets">6 · Ставки вперёд</a>
<a href="#next">7 · Дальше</a>
<a href="#appx">8 · Приложение</a>
</nav>
<main>

<section id="top" class="hero">
<div class="kick">neural_trade · тактическая сессия · T14 · BTC/USDT 1 мин, Binance</div>
<h1>Нелинейность на режимных признаках.<br><span class="grad">Носитель — суточный профиль и растяжение.</span></h1>
<p class="lead">Готовые модели временных рядов и картинки не добавили к нашему стеку ничего сверх контролей; сработала нелинейность на режимных признаках, и её носитель — суточный профиль будней и выходных плюс |растяжение|.</p>
<div class="kpis">{tiles}</div>
<h3 style="margin-top:22px">Прирост к V3 по цели A, взгляд 1 и взгляд 2</h3>
<div class="stage soft"><svg id="heroBars" viewBox="0 0 1200 {110 + 36 * len(HERO)}" role="img" aria-label="Прирост к V3 по цели A у каждого вида"></svg>
<button class="replay" data-for="hero" aria-label="Повторить">&#8635; повторить</button></div>
<p class="cap"><span class="key"><i style="background:var(--violet)"></i>взгляд 1: 2024-07..2025-07</span><span class="key"><i style="background:var(--gold)"></i>взгляд 2: 2025-11..2026-09</span>
<span class="key"><i style="background:var(--axis);border:1px solid var(--mut)"></i>засечка: контроль C2a того же прогона</span> Порог предрегистрации: +0,005. Метка слева — статус вида.</p>
<details><summary>Детали: что такое A, B, V3, H3, C1, C2a</summary>
<p class="note">V3 — рабочая база цели A (S3 + 96 UTC-слотов, ridge, переобучение раз в месяц на 365 днях); цель A — изменение log RV следующего часа, мера — месячный скилл против AR(1). H3 — рабочая база цели B (логистика на HAR-признаках); цель B — вероятность, что |ход| не меньше 50 б.п. в течение 60 минут, мера — относительное улучшение Брайера. База: V3 {rus(base['looks']['look1']['A']['V3_skill_vs_AR']['mean'])} / {rus(base['looks']['look2']['A']['V3_skill_vs_AR']['mean'])}; Брайер H3 {base['looks']['look1']['B']['brier']['H3']:.5f} / {base['looks']['look2']['B']['brier']['H3']:.5f}. C1 — те же столбцы, перемешанные внутри месяца; C2a — числа S3 того же окна плюс монотонное нелинейное преобразование.</p></details>
</section>

<section id="map">
<h2>1 · Карта концепций</h2>
<p class="lead">Каждый узел — взгляд на те же данные; ребро ведёт к рабочему стеку HAR / V3 / H3. Зелёные работают, жёлтые почти, серые не дали прироста над контролем, фиолетовые ещё не проверены, красный ждёт новых данных. Нажмите узел.</p>
<div class="mapwrap"><div class="stage"><svg id="mapSvg" viewBox="0 0 1100 560" role="img" aria-label="Карта концепций: взгляды на данные и рабочий стек"></svg>
<button class="replay" data-for="map" aria-label="Повторить">&#8635; повторить</button></div>
<div class="card mapcard" id="mapCard"><div class="note">Нажмите на узел карты.</div></div></div>
<p class="cap"><span class="key"><i style="background:var(--good)"></i>▲ работает</span><span class="key"><i style="background:var(--gold)"></i>◐ почти</span>
<span class="key"><i style="background:var(--neu)"></i>– нет прироста над контролем</span><span class="key"><i style="background:var(--violet)"></i>… в очереди</span><span class="key"><i style="background:var(--salmon)"></i>◆ нужны новые данные</span></p>
<details><summary>Детали: из каталога</summary><p class="note">Каталог и список из двенадцати кандидатов: runs/tactical/research/2026-10-17-multimodal/README.md и candidates.json. Правило каталога: любая картинка, граф или частотное кодирование — функция того же окна OHLCV; она может добавить нелинейность, но не информацию, поэтому у каждой есть числовой контроль той же информации.</p></details>
</section>

<section id="views">
<h2>2 · Шесть взглядов</h2>
<p class="lead">Шесть взглядов на данные добавлены к рабочему стеку как дополнительные столбцы: M1 Chronos-2, M2 TTM, M3 Kronos, M4 карточка режима, M5 VisionTS, M6 поиск аналогов. Ни один из пяти готовых или картиночных видов не обыграл контроль; поиск аналогов дал подсказку, и она привела к разделу 3. У каждого вида есть следующий шаг.</p>
<div class="cards vcards">{cards}</div>{m3note}
<p class="cap"><span class="key"><i style="background:var(--violet)"></i>вид (с 95% интервалом)</span><span class="key"><i style="background:var(--neu)"></i>C1: перемешанные столбцы</span><span class="key"><i style="background:var(--tan)"></i>C2a: числа S3 + нелинейность</span>
Пунктир: порог предрегистрации (A: +0,005; B: +1%). Взгляд 1 — слева, взгляд 2 — справа в каждой группе.</p>
</section>

<section id="worked">
<h2>3 · Что сработало</h2>
<p class="lead">Цепочка из четырёх шагов: аналоги намекнули на нелинейность, GBM на 12 ключах прошёл обе цели на обоих взглядах, ablation показал, что выигрыш несут «час × выходной» и |растяжение|, компактный профильный модуль вернул большую часть выигрыша.</p>
<div class="flow" id="flow">
<button class="step on" data-s="0"><span class="n">1</span><b>M6 · аналоги</b><span>подсказка</span></button>
<button class="step" data-s="1"><span class="n">2</span><b>M7 · GBM</b><span>проходит обе цели</span></button>
<button class="step" data-s="2"><span class="n">3</span><b>M8 · ablation</b><span>кто несёт</span></button>
<button class="step" data-s="3"><span class="n">4</span><b>M9 · модуль</b><span>восстановление</span></button>
</div>
<div class="stage soft"><svg id="stepSvg" viewBox="0 0 1100 330" role="img" aria-label="Шаги цепочки M6 – M9"></svg>
<button class="replay" data-for="step" aria-label="Повторить">&#8635; повторить</button></div>
<div class="readout wide" id="stepNote"></div>
<div class="two">
<div><h3>Ablation: кто несёт выигрыш M7</h3><div class="seg" id="ablSeg"><button class="on" data-t="A">цель A</button><button data-t="B">цель B</button></div>
<div class="stage soft"><svg id="ablSvg" viewBox="0 0 540 330"></svg></div>
<p class="cap">Доля: на сколько процентов падает выигрыш M7, если убрать группу (24 месяца, парная разность). Метка «несёт» — доля не меньше 25% и интервал выше нуля.</p></div>
<div><h3>Суточные часы: будни и выходные</h3>
<div class="seg" id="clkSeg"><button class="on" data-t="B">P(касания ±50 б.п. за 60 мин)</button><button data-t="A">изменение log RV следующего часа</button></div>
<div class="stage soft"><svg id="clockSvg" viewBox="0 0 560 560"></svg></div>
<div class="readout" id="clockRead"></div></div>
</div>
<p class="cap">Часы: подогнанный профиль финальной модели M9h6 (линии; золото — будни, розовый — выходные), точки — реализованные средние; пул Binance {PROF['pool']['first_pool_ta'][:10]}..{PROF['pool']['last_pool_ta'][:10]}, {PROF['pool']['n_pool']} якорей (внутри выборки; это описание профиля, а не проверка). Часы по UTC; отметки 13 и 14 — открытие США (13:30 летом, 14:30 зимой).</p>
<h3>Восстановление выигрыша M7 компактными модулями</h3>
<div class="stage soft"><svg id="recSvg" viewBox="0 0 1100 {80 + 40 * len(STORY['rec'])}"></svg></div>
<p class="cap">R = выигрыш модуля / выигрыш M7 за 24 месяца, отдельно по A и по B. Пороги заранее записаны в SPEC M9: «восстанавливает» от 0,75, «частично» от 0,40.</p>
<div class="two"><figure class="shot"><img alt="M8" src="{IMGS['M8']}"><figcaption>M8: ablation и линейное восстановление на реальных данных.</figcaption></figure>
<figure class="shot"><img alt="M9" src="{IMGS['M9']}"><figcaption>M9: профиль M9h6 и восстановление M7; пунктир — порог предрегистрации.</figcaption></figure></div>
<details><summary>Детали: ablation, восстановление, сопровождающие фигуры</summary>
<h3>Ablation M8 (цель A и B)</h3>{abl_table()}
<h3>M7: что показывает частичная зависимость</h3><figure class="shot"><img alt="M7 pd" src="{IMGS['M7pd']}"><figcaption>M7: частичная зависимость от перцентиля волатильности и часа.</figcaption></figure>
<h3>Восстановление R по видам M9</h3>{rec_table()}
<h3>M7 и M7c</h3>{table(['M7', 'M7c', 'M7b'], {'M7': 'M7 GBM', 'M7c': 'M7c GBM на остатках', 'M7b': 'M7b GBM + V3'})}
<p class="note">M7 прошёл обе цели на обоих взглядах (предрегистрировано); M7 выше контроля C2b — ядерного сглаживателя Надарая–Ватсона на тех же ключах (M7_kernel, без деревьев): флаги above_c2b на обеих целях и обоих взглядах истинны. M8 заморозил M7c_gbm_resid по правилу «большее среднее прироста A по двум взглядам» ({F8['rule_values']['M7c_gbm_resid']:.4f} против {F8['rule_values']['M7_gbm']:.4f}).</p>
<p class="note">Заранее зарегистрированные виды M9 не прошли оба взгляда на обеих целях: в первые месяцы Binance мало обучающих строк и профиль переобучается; M9h6 выбран после факта (метка в файле заморозки: «{F9['label']}»).</p>
</details>
</section>

<section id="controls">
<h2>4 · Контроли</h2>
<p class="lead">Информационно-эквивалентный числовой контроль C2a — те же числа S3 с монотонным нелинейным преобразованием — оказался сильнее готовых видов: C2a выше вида в {C2A_BEATEN} из {n_tot} сравнений (вид × взгляд × цель). Значит, ценность была не в модели, а в нелинейности по числам, и этот контроль теперь обязателен для каждого нового вида.</p>
<div class="seg" id="ctlSeg"><button class="on" data-t="A">цель A: прирост к V3</button><button data-t="B">цель B: Брайер, %</button></div>
<div class="stage soft"><svg id="ctlSvg" viewBox="0 0 1100 {90 + 22 * 2 * len(FM)}" role="img" aria-label="Вид против контроля C2a"></svg>
<button class="replay" data-for="ctl" aria-label="Повторить">&#8635; повторить</button></div>
<p class="cap"><span class="key"><i style="background:var(--violet)"></i>вид</span><span class="key"><i style="background:var(--tan)"></i>C2a</span> Для каждого вида два взгляда: сверху взгляд 1, снизу взгляд 2.</p>
<details><summary>Детали: сравнения вид и C2a</summary>
<table class="rt"><tr><th>вид</th><th>взгляд</th><th>цель</th><th>вид</th><th>C2a</th><th>вид выше C2a</th></tr>{"".join(f"<tr><td>{k}</td><td>{lk[-1]}</td><td>{t}</td><td>{rus(v) if t == 'A' else pct(v)}</td><td>{rus(c2) if t == 'A' else pct(c2)}</td><td>{fl(v > c2)}</td></tr>" for k, lk, t, v, c2 in cmp_rows)}</table>
<p class="note">Какие виды в сравнении: {", ".join(FM)}. Контроль: README каталога, раздел 2 (каждый образ, граф или частотное кодирование — функция того же окна OHLCV).</p></details>
</section>

<section id="install">
<h2>5 · Что установлено</h2>
<p class="lead">Две изолированные среды (основная среда <code>nt</code> с TensorFlow не тронута), общий кэш весов; у каждого вида есть загрузчик, который сразу делает проверочный прогон. Ниже то, что нужно, чтобы использовать установку повторно.</p>
<div class="kpis"><div class="kpi"><div class="kv">5,0 ГБ</div><div class="kl"><code>D:/nt/nt_mm_env</code>: Python 3.11, torch 2.11.0+cu126 (M1, M2, M4, M5, M6)</div></div>
<div class="kpi"><div class="kv">4,8 ГБ</div><div class="kl"><code>D:/nt/nt_mm_legacy</code>: Python 3.10, torch 2.5.1+cu124 (M3 Kronos)</div></div>
<div class="kpi"><div class="kv">6,7 ГБ</div><div class="kl"><code>D:/nt/nt_mm_cache</code>: веса, pip-кэш, клон Kronos</div></div></div>
{install_html()}
<details><summary>Детали: как не сломать среду и точные команды</summary>
<ul class="pts"><li>В <code>mm_env</code> нельзя ставить пакеты без проверки torch: <code>open_clip_torch</code> один раз подтянул CPU-сборку torch 2.14.1 поверх CUDA-колеса. Новые пакеты ставить с <code>--no-deps</code> или добавлять <code>torch==2.11.0+cu126 torchvision==0.26.0+cu126</code> в команду.</li>
<li>Chronos-2: <code>predict_df</code> требует столбец <code>item_id</code>; TTM: <code>get_model(repo, context_length=512, prediction_length=P)</code>, на каждую пару своя ветка снимка; Kronos: <code>predict(sample_count=N)</code> усредняет пути, для путей по отдельности вызывать с <code>sample_count=1</code> в цикле; VisionTS: веса с dl.fbaipublicfiles.com, пакету нужен <code>requests</code>.</li>
<li>Полные хеши и размеры: <code>weights_sha256.txt</code>; pip freeze: <code>versions_mm_env.txt</code>, <code>versions_mm_legacy.txt</code>; команды установки: <code>smoke.md</code>, <code>install_log.txt</code>.</li></ul></details>
</section>

<section id="bets">
<h2>6 · Проспективные ставки</h2>
<p class="lead">{FROZEN_N} ставок заморожены до появления данных (хеш записан, оценка один раз, без подгонки); {tl_n_new} из них новые: модуль M8 (M7c) и кандидат M9h6. Первые вердикты в январе–феврале 2027.</p>
<div class="card accent" id="ownerQ"><h3>Вопрос к владельцу</h3><p>Чтобы оценить эти ставки, нужен файл Binance, продлённый после 2026-09 (форвард-файл заканчивается 2026-09-30). Для этого нужно разрешение владельца на продление данных; <b>ответа пока нет, вопрос открыт</b>. Пока файла нет, проспективные оценки не запускаются.</p></div>
<div class="tl" id="tl"></div>
<p class="note">Нижняя шкала — месяцы 2026-10..2027-02; вертикальная линия — сегодня (2026-10-17). Полоса — окно оценки, ромб — момент, когда вердикт становится возможным. Фиолетовое — ставки прошлых сессий, золотая метка NEW — ставки этого спринта.</p>
<details><summary>Детали: критерии и источники хешей</summary>
<table class="rt"><tr><th>ставка</th><th>sha256 (начало)</th><th>критерий</th><th>файл</th></tr>{"".join(f"<tr><td>{r['id']}</td><td><code>{r['sha']}</code></td><td>{r['crit']}</td><td>{r['src']}</td></tr>" for r in TL)}</table>
<p class="note">Для M8 и M9h6 первый взгляд — после трёх полных месяцев с 2026-10, второй после шести, уровень Pocock одностороннего p равен 0,0294 (SPEC_M8, SPEC_M9). {'Хеши O4–O11 сверены с файлами заморозки при сборке.' if not TL_UNCHECKED else 'Не сверено при сборке (нет worktree с outlook): ' + ', '.join(TL_UNCHECKED) + '.'}</p></details>
</section>

<section id="next">
<h2>7 · Прорывные направления дальше</h2>
<p class="lead">Идём от того, что работает: профиль «час × выходной» и |растяжение| становятся входом, на него опираются остальные виды. Каждый шаг ниже — конкретный следующий эксперимент.</p>
<div class="cards">{nx}</div>
</section>

<section id="appx">
<h2>8 · Приложение</h2>
<p class="lead">Все числа спринта в одном месте, как они записаны в файлах результатов: оба взгляда, обе цели, контроли и односторонние p; ниже отклонения от плана.</p>
<details><summary>Таблица всех видов и вариантов</summary>{appendix_html()}</details>
<h3 style="margin-top:18px">Отклонения и оговорки</h3><ul class="pts wide">{dev_html}</ul>
<h3>Источники чисел</h3>
<p class="note">runs/tactical/tasks/T14_multimodal/*/results.json и tables.txt (виды), M8_module/results_m8.json, M8_module/frozen_P.json, M9_module/results_m9.json, M9_module/frozen_P_M9h6_prof_str.json, M9_module/profile_M9h6_prof_str.json, core/reference/baselines.json, weights_sha256.txt, versions_*.txt, smoke.md; Kronos: {M3_SRC}. При сборке выполнено {len(CHECKS)} проверок ключевых чисел против файлов.</p>
</section>
<footer>Утренний отчёт T14 · собран скриптом runs/tactical/report/build_multimodal.py из закоммиченных файлов результатов; журнал: runs/tactical/LOG.md</footer>
</main>"""
    data_js = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    html = ("<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Мультимодальная ночь: утренний отчёт T14</title><style>" + css + "</style></head><body>" + body +
            "<script id=\"data\" type=\"application/json\">" + data_js + "</script><script>(()=>{" + js + "})();</script></body></html>")
    return html


def main():
    html = page()
    out = HERE / "multimodal_report.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    assert mb < 12, mb
    print("written", out, f"{mb:.2f} MB; images:", {k: f"{v / 1e3:.0f} kB" for k, v in img_sizes.items()}, "M3:", "pending" if M3_PENDING else M3_SRC)


if __name__ == "__main__":
    main()
