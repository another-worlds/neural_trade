"""T13 judgement (SPEC.md): J1 = Bitstamp 2017-10..2018-12 walk-forward (part B: 2018-07..2018-12), J2 = Binance 2025-11..2026-09 with the models frozen at
2025-07-20 (parts A and B; part C as T11's report-only recipe: a monthly refit on the forward file's own bars from 2025-10-01). Runs ONLY after selection.json is
committed and clean (the first statement). usage: python run_judge.py [A|B|C|ABC] -> judge_part_<X>.json, then results.json and tables.txt once all three exist"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t13_lib as L

SEL = os.path.join(HERE, "selection.json")
HEAD = L.assert_selection_committed(SEL)                  # first: tracked, clean, in HEAD (no data is read before this)
SELJ = json.load(open(SEL)); SEL_SHA = hashlib.sha256(open(SEL, "rb").read()).hexdigest()
PAIRS = {p: L.pair_from_key(SELJ["chosen"][p]) for p in "ABC"}
try:
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
t0 = time.time()


def el():
    return f"{time.time() - t0:.0f}s"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def t_summary(g):
    mu, lo, hi, p = L.tcrit_ci(list(g.values()))
    return {"mean": mu, "ci": [lo, hi], "p": p, "pos": int(sum(v > 0 for v in g.values())), "n": len(g), "monthly": g}


def run_A():
    pr = PAIRS["A"]
    bars = L.L12.load_lookA(); P = L.L10.prep(bars, "cut"); B = L.L10.build(P, L.T_MIN)
    R1, info1 = L.a_walk(B, P, L.T_MIN, L.J1A_MONTHS, pr, days_fn=L.L12.train_days); s1 = L.a_summary(R1)
    out = {"pair": L.pair_key(pr), "J1": {"summary": s1, "criterion": L.a_criterion(s1), "info": info1}}; log("A J1", s1["A1_vs_H3"], el()); del P, B, bars
    cut = L.guard.load_ohlcv("cut"); Pc = L.L10.prep(cut, "cut"); Bc = L.L10.build(Pc, L.T_MIN); Bcp = L.a_pair_B(Pc, Bc, pr)
    fr = L.a_fit(Bcp, Pc, L.T_MIN, L.guard.CUT_END + pd.Timedelta(minutes=1), L.L10.TRAIN_DAYS, pr)
    log("A frozen fit", fr["n_train"], fr["last_label_end"], el()); del cut, Pc, Bc, Bcp
    fwd = L.guard.load_ohlcv("fwd"); Pf = L.L10.prep(fwd, "fwd"); Bf = L.L10.build(Pf, L.T_MIN)
    R2, info2 = L.a_walk(Bf, Pf, L.T_MIN, L.J2_MONTHS, pr, frozen=fr); s2 = L.a_summary(R2)
    out["J2"] = {"summary": s2, "criterion": L.a_criterion(s2), "info": info2, "frozen": {"n_train": fr["n_train"], "last_label_end": fr["last_label_end"]}}; log("A J2", s2["A1_vs_H3"], el())
    return out


def run_B():
    pr = PAIRS["B"]
    z = np.load(os.path.join(L.SCRATCH, "frame_sel_B.npz")); F = {k: z[k] for k in z.files}
    rows1 = L.b_walk(F, L.J1B_MONTHS, pr); del F, z
    out = {"pair": L.pair_key(pr), "J1": {"rows": rows1, **t_summary(L.b_gain(rows1))}}; log("B J1", out["J1"]["mean"], el())
    s = L.T01.load_bars(L.T01.TRAIN_CSV, L.O5.LOAD_FROM, L.T01.CUTOFF); Fc = L.build_B_frame(s, L.O5.FIRST5, pairs=[pr]); del s
    fr = L.b_fit(Fc, L.O1.TRAIN_END, pr); assert pd.Timestamp(fr["last_label_end"]) <= L.O1.TRAIN_END; log("B frozen fit", fr["n_tr"], fr["last_label_end"], el()); del Fc
    fw = L.O5.load_freeze_bars(); Ff = L.build_B_frame(fw, L.O3.FIRST, pairs=[pr]); rows2 = L.b_walk(Ff, L.J2_MONTHS, pr, frozen=fr)
    out["J2"] = {"rows": rows2, **t_summary(L.b_gain(rows2)), "frozen": {"n_tr": fr["n_tr"], "last_label_end": fr["last_label_end"]}}; log("B J2", out["J2"]["mean"], el())
    return out


def run_C():
    pr = PAIRS["C"]
    s = L.T01.load_bars(L.T01.TRAIN_CSV, L.L11.A_LO, L.L11.A_HI); BA = L.build_C(s, 1441, "bitstamp", pairs=[pr]); rows1 = L.c_walk(BA, L.J1A_MONTHS, pr); del s, BA
    out = {"pair": L.pair_key(pr), "J1": {"rows": rows1, **t_summary(L.c_gain(rows1))}}; log("C J1", out["J1"]["mean"], el())
    fw = L.L11.load_freeze_bars(L.T01.FWD_CSV); assert fw.index.max() <= L.L11.FW_HI
    BB = L.build_C(fw, L.T01.WARM, "binance", pairs=[pr]); rows2 = L.c_walk(BB, L.J2_MONTHS, pr, start=L.L11.FW_LO)
    out["J2"] = {"rows": rows2, **t_summary(L.c_gain(rows2))}; log("C J2", out["J2"]["mean"], el())
    return out


def verdicts(out):
    a1, a2 = out["A"]["J1"]["criterion"], out["A"]["J2"]["criterion"]
    out["A"]["criterion"] = {"J1": a1["pass"], "J2": a2["pass"], "pass": bool(a1["pass"] and a2["pass"]), "p": max(a1["p"], a2["p"])}
    for part, thr in (("B", 0.003), ("C", 0.005)):
        c = {}
        for look in ("J1", "J2"):
            d = out[part][look]; c[look] = bool(d["mean"] >= thr and d["ci"][0] > 0)
        out[part]["criterion"] = {"threshold": thr, **c, "pass": bool(c["J1"] and c["J2"]), "p": max(out[part]["J1"]["p"], out[part]["J2"]["p"])}
    ps = [out[p]["criterion"]["p"] for p in "ABC"]; o = sorted(range(3), key=lambda i: ps[i]); adj = [0.0] * 3; run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (3 - r) * ps[i])); adj[i] = run
    for p, ad in zip("ABC", adj):
        out[p]["criterion"]["holm_p"] = ad; out[p]["criterion"]["PASS"] = bool(out[p]["criterion"]["pass"] and ad < 0.05)


def tables(out):
    def f(d, nd=4):
        return f"{d['mean']:+.{nd}f} [{d['ci'][0]:+.{nd}f},{d['ci'][1]:+.{nd}f}] p={d['p']:.4f} +{d['pos']}/{d['n']}"

    def q(d):
        return f"{100 * d['rel']:+.2f}% [{100 * d['ci'][0]:+.2f},{100 * d['ci'][1]:+.2f}] p={d['p']:.4f}"

    def mean_of(part, look, model, key):
        return np.mean([r[key][model] for r in out[part][look]["rows"]])
    ln = [f"T13 judgement, run at HEAD {HEAD}, selection.json sha256 {SEL_SHA}; chosen pairs {SELJ['chosen']}", ""]
    for look, title in (("J1", "J1 Bitstamp 2017-10..2018-12 (B: 2018-07..2018-12)"), ("J2", "J2 Binance 2025-11..2026-09 (A, B frozen at 2025-07-20; C: monthly refit on the forward file)")):
        s = out["A"][look]["summary"]
        ln += [title, f"  A  n={s['n']} base_rate={s['base_rate']:.4f} months={s['n_months']}  Brier H3 {s['brier']['H3']:.5f} A1 {s['brier']['A1']:.5f} AS {s['brier']['AS']:.5f}",
               f"     A1 vs H3 {q(s['A1_vs_H3'])}   A1 vs AS {q(s['A1_vs_AS'])}   AS vs H3 {q(s['AS_vs_H3'])}   AUC H3 {s['auc']['H3']:.4f} A1 {s['auc']['A1']:.4f} AS {s['auc']['AS']:.4f}",
               f"     criterion: {out['A'][look]['criterion']}",
               f"  B  Spearman gain B1 - L1 {f(out['B'][look])}   (mean Spearman L1 {mean_of('B', look, 'L1', 'spearman'):.4f}, B1 {mean_of('B', look, 'B1', 'spearman'):.4f})",
               f"  C  skill_ar gain C1 - V3 {f(out['C'][look])}   (mean skill V3 {mean_of('C', look, 'V3', 'skill_ar'):+.4f}, C1 {mean_of('C', look, 'C1', 'skill_ar'):+.4f})", ""]
    ln.append("Criteria (both looks) and Holm over A, B, C:")
    for p in "ABC":
        c = out[p]["criterion"]
        ln.append(f"  {p} pair {out[p]['pair']}: J1 {c['J1']} J2 {c['J2']} -> criteria {'PASS' if c['pass'] else 'FAIL'}; p(max over looks) {c['p']:.4f}, Holm p {c['holm_p']:.4f} -> {'PASS' if c['PASS'] else 'FAIL'}")
    ln += ["", "Monthly gains B (Spearman B1 - L1):"] + [f"  {look} " + " ".join(f"{m}:{v:+.4f}" for m, v in out["B"][look]["monthly"].items()) for look in ("J1", "J2")]
    ln += ["Monthly gains C (skill C1 - V3):"] + [f"  {look} " + " ".join(f"{m}:{v:+.4f}" for m, v in out["C"][look]["monthly"].items()) for look in ("J1", "J2")]
    ln += ["Monthly gains A (relative Brier A1 vs H3):"] + [f"  {look} " + " ".join(f"{m}:{v:+.4f}" for m, v in out["A"][look]["summary"]["monthly_gain_A1_vs_H3"].items()) for look in ("J1", "J2")]
    return ln


if __name__ == "__main__":
    out = {"head": HEAD, "selection_sha256": SEL_SHA, "chosen": SELJ["chosen"]}
    only = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    part_file = lambda p: os.path.join(HERE, f"judge_part_{p}.json")
    for p, fn in (("A", run_A), ("B", run_B), ("C", run_C)):
        if p in only and not os.path.exists(part_file(p)):
            o = fn(); json.dump(o, open(part_file(p), "w"), indent=1, default=float); log("part", p, "saved", el())
    if all(os.path.exists(part_file(p)) for p in "ABC"):
        for p in "ABC":
            out[p] = json.load(open(part_file(p)))
        verdicts(out); out["runtime_s"] = time.time() - t0
        json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
        ln = tables(out); open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(ln) + "\n"); print("\n".join(ln)); log("total", el())
