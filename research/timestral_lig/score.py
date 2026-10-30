"""Score Timestral-LIG on Test against the pre-registered pass rule (P1-P7). Run once, after `train.py export`.

Per pair (variant A vs base B, identical origins): per-horizon % changes of pinball21 and QLIKE with 95% CIs, and
the aggregate test of the spec: d_i = (1/3) sum_h (l^A_hi - l^B_hi) / den_h, where den_h is B's mean pinball21 or
B's unit-free mean QLIKE on y != 0; 1,000 moving-block bootstrap draws, block 20 origins (60 min), rng seed 0.
Tail tests (P3) and MAE / AUC (P7) come from metrics2.py run on a directory holding only the files it needs.
Writes $LIG_WORK/score.json and prints markdown tables.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import train
from data import Data
from losses import block_ci, eval_losses

BENCH = os.environ.get("LIG_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tsfm_eval", "bench"))
A0 = "timestral-28M_ctx2048_price"
REF209 = "timestral-209M_ctx2048_price"
TAG = train.TAG


def load_q(D, tag):
    for d in (D.results, D.bench_results):
        p = os.path.join(d, f"{tag}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            assert np.array_equal(z["origins"], D.test), tag
            return (np.log(np.maximum(z["q"].astype(np.float64), 1e-9)) - D.lc[D.test][:, None, None]) * 1e4
    return None


def pair(D, qa, qb):
    y = np.stack([(D.lc[D.test + h] - D.lc[D.test]) * 1e4 for h in (1, 5, 15)], 1)
    # eval_losses expects (N, 15, 21) and (N, 15); embed the three horizons at steps 0, 4, 14
    emb = lambda q: np.repeat(q[:, [0]], 15, 1).copy()
    QA, QB, Y = emb(qa), emb(qb), np.zeros((len(y), 15))
    for j, k in enumerate((0, 4, 14)):
        QA[:, k], QB[:, k], Y[:, k] = qa[:, j], qb[:, j], y[:, j]
    pa, la, _ = eval_losses(QA, Y)
    pb, lb, ub = eval_losses(QB, Y)
    den_p, den_q = pb.mean(0), np.abs(np.nanmean(ub, 0))
    out = {}
    for name, a, b, den in (("pin21", pa, pb, den_p), ("qlike", la, lb, den_q)):
        cells = []
        for h in range(3):
            d = 100 * (a[:, h] - b[:, h]) / den[h]
            cells.append([round(float(d.mean()), 3), [round(v, 3) for v in block_ci(d, 20)]])
        agg = 100 * ((a - b) / den).mean(1)
        out[name] = {"cells": cells, "agg": round(float(agg.mean()), 3), "agg_ci": [round(v, 3) for v in block_ci(agg, 20)]}
    return out


def run_metrics2(ds, D, tags):
    """metrics2.py on a directory holding only `tags` (tail tests, MAE skill, AUC)."""
    mdir = D.results + "_m"
    os.makedirs(mdir, exist_ok=True)
    for f in os.listdir(mdir):
        os.remove(os.path.join(mdir, f))
    for t in tags:
        for d in (D.results, D.bench_results):
            if os.path.exists(os.path.join(d, f"{t}.npz")):
                shutil.copy(os.path.join(d, f"{t}.npz"), mdir)
                break
    env = dict(os.environ, BENCH_DATA=ds, BENCH_RESULTS=mdir)
    subprocess.run([sys.executable, os.path.join(BENCH, "metrics2.py")], env=env, check=True,
                   stdout=subprocess.DEVNULL, cwd=BENCH)
    rows = json.load(open(os.path.join(mdir, "metrics2.json")))["rows"]
    return {(r["model"], r["h"]): r for r in rows if r.get("set") == "full"}


def tails_failed(m, tag):
    return sum(int(m[(tag, h)][k] < 0.05) for h in (1, 5, 15) for k in ("nw_p_q01", "nw_p_q99"))


def main():
    out = {"sets": {}}
    per = {}
    for ds in ("repo", "post"):
        D = Data(ds)
        cj = json.load(open(train.wdir(ds, "candidate.json")))
        cand = cj["candidate"]
        rungs = {r: v["seeds"] for r, v in cj["rungs"].items()}
        tag_of = lambda r: f"{TAG}_{r}_ens" if len(rungs.get(r, [])) > 1 else f"{TAG}_{r}_s0"
        q0 = load_q(D, A0)
        s = {"candidate": cand, "vs_A0": {}, "pairs": {}}
        for r in rungs:
            q = load_q(D, tag_of(r))
            if q is not None:
                s["vs_A0"][r] = pair(D, q, q0)
            for sd in rungs[r]:
                qs = load_q(D, f"{TAG}_{r}_s{sd}")
                if qs is not None:
                    s["vs_A0"][f"{r}_s{sd}"] = pair(D, qs, q0)
        if cand != "A0":
            qc = load_q(D, tag_of(cand))
            for base in ("A1", "A2-plc", "C-FHS", "A4L"):
                if base in rungs and base != cand:
                    s["pairs"][f"cand_vs_{base}"] = pair(D, qc, load_q(D, tag_of(base)))
        if "A2" in rungs and "A2-close" in rungs:
            s["pairs"]["A2_vs_A2-close"] = pair(D, load_q(D, tag_of("A2")), load_q(D, tag_of("A2-close")))
        # V2: A2 vs placebo (P5 second half)
        X = train.Ctx(ds)
        if "A2" in rungs and "A2-plc" in rungs:
            from losses import agg_vs_base
            Sd, d, _ = agg_vs_base(train.v2q(ds, "A2", rungs["A2"]), train.v2q(ds, "A2-plc", rungs["A2-plc"]),
                                   X.D.target(X.D.v2)[0].numpy())
            s["v2_A2_vs_plc"] = {"S": round(Sd, 4), "ci": [round(v, 4) for v in block_ci(d, 60)]}
        mtags = [A0, REF209] + [tag_of(r) for r in rungs] + [f"{TAG}_{cand}_s{sd}" for sd in rungs.get(cand, [])]
        m = run_metrics2(ds, D, sorted(set(mtags)))
        s["metrics2"] = {t: {h: {k: m[(t, h)].get(k) for k in ("pinball21_bps", "qlike", "qlike_vs_fhs_pct", "qlike_vs_fhs_ci95_pct",
                                                                 "nw_p_q01", "nw_p_q99", "cov98", "mae_skill_vs_rw", "auc_up", "auc_up_ci95")}
                             for h in (1, 5, 15)} for t in sorted(set(mtags)) if (t, 1) in m}
        s["tails_failed"] = {t: tails_failed(m, t) for t in sorted(set(mtags)) if (t, 1) in m}
        out["sets"][ds] = s
        per[ds] = (s, rungs, tag_of, m)

    # ---------------------------------------------------------------- pass rule
    def meets(ds, key):
        s, rungs, tag_of, m = per[ds]
        return s["vs_A0"].get(key)

    def p1(r):
        return all((v := meets(ds, r)) and v["qlike"]["agg"] <= -0.5 and v["qlike"]["agg_ci"][1] < 0
                   and not any(c[1][0] > 0 for c in v["qlike"]["cells"]) for ds in per)

    def p2(r):
        return all((v := meets(ds, r)) and v["pin21"]["agg_ci"][1] <= 0.10
                   and not any(c[0] > 0.30 for c in v["pin21"]["cells"]) for ds in per)

    def p3(r):
        return all(per[ds][0]["tails_failed"].get(per[ds][2](r), 99) <= per[ds][0]["tails_failed"][A0] for ds in per)

    def p4(r):
        ok = True
        for ds in per:
            for sd in per[ds][1].get(r, []):
                v = meets(ds, f"{r}_s{sd}")
                ok &= bool(v) and v["qlike"]["agg"] < 0 and v["pin21"]["agg_ci"][1] <= 0.10 \
                    and not any(c[0] > 0.30 for c in v["pin21"]["cells"])
        return ok

    cands = {ds: per[ds][0]["candidate"] for ds in per}
    rule = {"candidates": cands}
    same = len(set(cands.values())) == 1 and cands["repo"] != "A0"
    if same:
        c = cands["repo"]
        rule.update(P1=p1(c), P2=p2(c), P3=p3(c), P4=p4(c))
        p5a = all((v := per[ds][0]["pairs"].get("cand_vs_A1")) is not None and v["qlike"]["agg"] < 0 for ds in per) \
            and any(per[ds][0]["pairs"]["cand_vs_A1"]["qlike"]["agg_ci"][1] < 0 for ds in per) if c != "A1" else False
        p5b = all(per[ds][0].get("v2_A2_vs_plc", {}).get("ci", [0, 0])[1] < 0 for ds in per)
        rule["P5"] = bool(p5a and p5b)
        rule["P6"] = all((v := per[ds][0]["pairs"].get("cand_vs_C-FHS")) is not None and v["qlike"]["agg_ci"][1] < 0 for ds in per)
        mae = lambda ds, t, h: per[ds][3][(t, h)]["mae_skill_vs_rw"]
        rule["P7"] = all(mae(ds, per[ds][2](c), h) >= mae(ds, A0, h) - 0.002 for ds in per for h in (1, 5, 15))
    rule["A1_P1-P4"] = all(f("A1") for f in (p1, p2, p3, p4)) if all("A1" in per[ds][1] for ds in per) else False
    if not same:
        verdict = "FAIL: no common candidate rung promoted on V2 for both sets"
    elif all(rule[k] for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")):
        verdict = "PASS"
    elif all(rule[k] for k in ("P1", "P2", "P3", "P4", "P5")):
        verdict = "PASS-licence-free (recommend C-FHS)"
    elif rule["A1_P1-P4"] and not rule.get("P5", False):
        verdict = "PASS-recal (ship A1; indicators unproven)"
    else:
        verdict = "FAIL"
    rule["verdict"] = verdict
    out["pass_rule"] = rule
    json.dump(out, open(os.path.join(train.WORK, "score.json"), "w"), indent=1)

    # ---------------------------------------------------------------- tables
    cell = lambda c: f"{c[0]:+.2f} [{c[1][0]:+.2f}, {c[1][1]:+.2f}]"
    for ds in per:
        s = per[ds][0]
        print(f"\n#### Test, {ds} set: rungs vs zero-shot Timestral-28M (A0); % change, negative = better; candidate = {s['candidate']}\n")
        print("| Rung | QLIKE agg [95% CI] | QLIKE h=1 / 5 / 15 | Pinball21 agg [95% CI] | Pinball21 h=1 / 5 / 15 |")
        print("|---|---|---|---|---|")
        for r, v in s["vs_A0"].items():
            q, p = v["qlike"], v["pin21"]
            print(f"| {r} | {q['agg']:+.2f} [{q['agg_ci'][0]:+.2f}, {q['agg_ci'][1]:+.2f}] | "
                  + " / ".join(f"{c[0]:+.2f}" for c in q["cells"])
                  + f" | {p['agg']:+.2f} [{p['agg_ci'][0]:+.2f}, {p['agg_ci'][1]:+.2f}] | " + " / ".join(f"{c[0]:+.2f}" for c in p["cells"]) + " |")
        for k, v in s["pairs"].items():
            print(f"\n{k}: QLIKE agg {v['qlike']['agg']:+.2f} {v['qlike']['agg_ci']}, pinball21 agg {v['pin21']['agg']:+.2f} {v['pin21']['agg_ci']}")
        print("tails failed (of 6):", s["tails_failed"])
    print("\nPass rule:", json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
