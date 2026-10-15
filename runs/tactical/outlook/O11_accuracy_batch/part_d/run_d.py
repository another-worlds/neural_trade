"""O11 part d runner. usage: run_d.py [select|judge|freeze]
  select : monthly AUC of base, d1, d2, d3 on S (2021-01..2024-06) -> selection.jsonl, tables_selection.txt, choice_d.json (committed ALONE before judge)
  judge  : needs choice_d.json committed and unchanged. B (2024-07..2025-07-20 walk-forward) and C (2025-10..2026-09, frozen on the 365 / 90 days to 2025-07-20)
           for base and all three variants -> results.jsonl, tables.txt
  freeze : only if the primary passed. Fit base and the chosen variant once on the Binance file to 2026-09-30 23:59 -> frozen_P_d.json (+ sha256)
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o11d_lib as D  # noqa: E402
L9 = D.L9; L8 = D.L8; P5 = D.P5; O3 = D.O3; T = D.T; P2 = D.P2
LOG = os.path.join(HERE, "run.log"); SEL = os.path.join(HERE, "selection.jsonl"); TABLE_SEL = os.path.join(HERE, "tables_selection.txt")
RES = os.path.join(HERE, "results.jsonl"); TABLE = os.path.join(HERE, "tables.txt"); FROZEN = os.path.join(HERE, "frozen_P_d.json")
VARS = list(D.VARIANTS)


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nice():
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def free_gb():
    import psutil; return psutil.virtual_memory().available / 2 ** 30


def load_cut():
    s = L8.load_bars_cut(); DF = L8.load_dframe(); TF = L8.load_train_frame()
    X = L9.load_train_extras(TF, s); DX = L9.load_dec_extras(DF, s)
    return s, DF, TF, X, DX


def cached_pred(name, fn):
    os.makedirs(D.SCRATCH, exist_ok=True); p = os.path.join(D.SCRATCH, f"pred_{name}.npz")
    if os.path.exists(p):
        z = np.load(p); return {v: z[v] for v in z.files}
    P = fn(); np.savez(p, **P); return P


def fmt(x, p=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


# ------------------------------------------------------------------ selection
def select():
    nice(); t0 = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    s, DF, TF, X, DX = load_cut(); TX = D.train_data(TF, X); DD = D.decision_data(s, DF, DX)
    log("loaded", len(DF["ta"]), "decisions", int(DD["valid"].sum()), "valid", f"{time.time() - t0:.0f}s")
    P = cached_pred("S", lambda: D.predict_months(TX, DD, D.S_MONTHS, log=log))
    tab = D.table(P, DD, D.S_MONTHS)
    with open(SEL, "w", encoding="utf-8") as f:
        for v in VARS:
            for mo, m in tab[v].items():
                f.write(json.dumps({"span": "S", "variant": v, "month": mo, **m}) + "\n")
    months = [m for m in D.S_MONTHS if all(m in tab[v] for v in VARS)]
    summ = {}
    for v in VARS:
        a = np.array([tab[v][m]["auc"] for m in months]); summ[v] = D.mean_t(a); summ[v]["t_half"] = D.mean_t(a - 0.5)["t"]
        if v != "base":
            summ[v]["vs_base"] = D.mean_t(D.paired(tab, v, "base", "auc", months))
        summ[v]["brier"] = float(np.mean([tab[v][m]["brier"] for m in months]))
        summ[v]["auc_gated"] = float(np.nanmean([tab[v][m]["auc_gated"] for m in months])); summ[v]["hit_gated"] = float(np.nanmean([tab[v][m]["hit_gated"] for m in months]))
    choice = D.choose({v: summ[v]["mean"] for v in VARS})
    with open(SEL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summ, "n_months": len(months), "choice": choice}) + "\n")
    o = []; Q = o.append
    Q("O11 part d SELECTION on S (2021-01..2024-06). Monthly AUC of p_up for the sign of the next-open 60-minute return at 15-minute decisions, all valid decision rows,")
    Q(f"no dead band in scoring; {len(months)} months; mean over months, t = (mean - 0.5) / se. Rule: among d1, d2, d3 the highest mean monthly AUC.")
    Q(f"{'variant':8s} {'features':12s} {'train days':>10s} {'mean AUC':>9s} {'t(vs .5)':>8s} {'AUC-.5':>7s} {'vs base':>9s} {'t(vs)':>7s} {'mo+':>6s} {'Brier':>7s} {'AUC gated':>10s} {'hit gated':>10s}")
    for v in VARS:
        ug, dy = D.VARIANTS[v]; r = summ[v]; vb = r.get("vs_base")
        Q(f"{v:8s} {'tb7+G15' if ug else 'tb7':12s} {dy:10d} {r['mean']:9.4f} {r['t_half']:8.2f} {r['mean'] - 0.5:7.4f} "
          f"{fmt(vb['mean']) if vb else '':>9s} {fmt(vb['t'], 2) if vb else '':>7s} {(str(vb['pos']) + '/' + str(vb['n'])) if vb else '':>6s} {r['brier']:7.4f} {r['auc_gated']:10.4f} {r['hit_gated']:10.4f}")
    Q(f"CHOICE (highest mean AUC among d1, d2, d3): {choice}")
    Q("\nmonthly AUC by year (mean of the months): " + "  ".join(f"{v}:" + ",".join(f"{y}:{np.mean([tab[v][m]['auc'] for m in months if m[:4] == y]):.4f}" for y in sorted({m[:4] for m in months})) for v in VARS))
    open(TABLE_SEL, "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))
    ch = {"part": "O11 d", "rule": "highest mean monthly AUC on S among d1, d2, d3 (SPEC part d)", "variant": choice, "features": "tb7+G15" if D.VARIANTS[choice][0] else "tb7",
          "train_days": D.VARIANTS[choice][1], "selection_span": "2021-01..2024-06", "n_months": len(months),
          "mean_auc_S": {v: summ[v]["mean"] for v in VARS}, "t_vs_half_S": {v: summ[v]["t_half"] for v in VARS}}
    json.dump(ch, open(D.CHOICE, "w", encoding="utf-8"), indent=1)
    log("select done", choice, f"{time.time() - t0:.0f}s"); json.dump({"runtime_s": time.time() - t0}, open(os.path.join(HERE, "runtime_select.json"), "w"))


# ------------------------------------------------------------------ judgement
def span_C(TF, X):
    """C: Binance bars 2025-10-01 .. 2026-10-02 (forbidden gap refused by the loader); features from forward bars only; models frozen at the cut."""
    s = T.load_bars(T.FWD_CSV, T.FWD_START, D.C_END)
    assert s.index.min() >= T.FWD_START and s.index.max() <= D.C_END
    DF, DX = P2.make_dframe_for(s, P5.FREEZE_FIRST); DD = D.decision_data(s, DF, DX); TX = D.train_data(TF, X)
    P = D.predict_months(TX, DD, D.C_MONTHS, frozen_m0=O3.TRAIN_END, log=log)
    return DD, P


def judge():
    ch = D.choice_gate(); chosen = ch["variant"]; assert chosen in D.CHOOSABLE
    nice(); t0 = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    if os.path.exists(RES):
        raise SystemExit("REFUSED: results.jsonl exists; J is computed once.")
    log("judge: chosen", chosen, "(choice_d.json committed and unchanged)")
    s, DF, TF, X, DX = load_cut(); TX = D.train_data(TF, X); DD = D.decision_data(s, DF, DX)
    PB = cached_pred("B", lambda: D.predict_months(TX, DD, D.B_MONTHS, log=log)); tabB = D.table(PB, DD, D.B_MONTHS)
    log("B done", f"{time.time() - t0:.0f}s")
    DDc, PC = span_C(TF, X); tabC = D.table(PC, DDc, D.C_MONTHS)
    log("C done", f"{time.time() - t0:.0f}s")
    tab = {v: {**{m: {**x, "span": "B"} for m, x in tabB[v].items()}, **{m: {**x, "span": "C"} for m, x in tabC[v].items()}} for v in VARS}
    mB = [m for m in D.B_MONTHS if all(m in tabB[v] for v in VARS)]; mC = [m for m in D.C_MONTHS if all(m in tabC[v] for v in VARS)]; mJ = mB + mC
    rec = []
    for v in VARS:
        for mo in mJ:
            rec.append({"kind": "month", "variant": v, "month": mo, **tab[v][mo]})
    out = {"kind": "summary", "chosen": chosen, "months_B": mB, "months_C": mC}
    for name, ms in (("J", mJ), ("B", mB), ("C", mC)):
        o = {}
        for v in VARS:
            o[v] = {"auc": {**D.mean_t([tab[v][m]["auc"] for m in ms]), "t_half": D.mean_t(np.array([tab[v][m]["auc"] for m in ms]) - 0.5)["t"]}, "brier": D.mean_t([tab[v][m]["brier"] for m in ms]),
                    "auc_gated": D.mean_t([tab[v][m]["auc_gated"] for m in ms]), "hit_gated": D.mean_t([tab[v][m]["hit_gated"] for m in ms]), "hit_all": D.mean_t([tab[v][m]["hit_all"] for m in ms])}
        for v in VARS:
            if v == "base":
                continue
            o[v]["vs_base"] = {k: D.mean_t(D.paired(tab, v, "base", k, ms)) for k in ("auc", "brier", "auc_gated", "hit_gated", "hit_all")}
        out[name] = o
    pj = out["J"][chosen]["vs_base"]["auc"]
    out["primary"] = {"variant": chosen, "test": "chosen - base, monthly AUC, B + C pooled months", **pj, "min_effect": 0.005,
                      "pass_before_holm": bool(pj["mean"] >= 0.005 and pj["lo"] > 0)}
    # pooled over all J rows: gated hit rate and base rate
    pooled = {}
    for v in VARS:
        ng = sum(tab[v][m]["n_gated"] for m in mJ); n = sum(tab[v][m]["n"] for m in mJ)
        pooled[v] = {"n": n, "n_gated": ng, "hit_gated": float(sum(tab[v][m]["hit_gated"] * tab[v][m]["n_gated"] for m in mJ if tab[v][m]["n_gated"]) / ng) if ng else None,
                     "base_rate": float(sum(tab[v][m]["base_rate"] * tab[v][m]["n"] for m in mJ) / n)}
    out["pooled_rows"] = pooled
    rec.append(out)
    with open(RES, "w", encoding="utf-8") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    write_tables(tab, out, mB, mC)
    json.dump({"runtime_s": time.time() - t0}, open(os.path.join(HERE, "runtime_judge.json"), "w"))
    log("judge done", "primary", {k: out["primary"].get(k) for k in ("mean", "lo", "hi", "p", "pass_before_holm")}, f"{time.time() - t0:.0f}s")


def write_tables(tab, out, mB, mC):
    o = []; Q = o.append; chosen = out["chosen"]
    Q(f"O11 part d JUDGEMENT. chosen on S: {chosen}. J = B {mB[0]}..{mB[-1]} ({len(mB)} months, walk-forward) + C {mC[0]}..{mC[-1]} ({len(mC)} months, frozen at the 2025-07-20 cut): {len(mB) + len(mC)} months.")
    Q("Second-look span: J was read by earlier studies for other claims (SPEC); P is the clean verdict.")
    p = out["primary"]
    Q(f"PRIMARY {chosen} - base, monthly AUC: mean {p['mean']:+.4f}  95% t [{p['lo']:+.4f}, {p['hi']:+.4f}]  t {p['t']:+.2f}  months positive {p['pos']}/{p['n']}  raw two-sided p {p['p']:.4f}")
    Q(f"  pass rule: mean >= +0.005 and lower bound > 0 (before Holm): {'PASS' if p['pass_before_holm'] else 'NOT passed'}")
    for name in ("J", "B", "C"):
        Q(f"\n== {name}: mean over months ==")
        Q(f"{'variant':8s} {'AUC':>7s} {'t.5':>6s} {'vs base':>9s} [{'95% t':>17s}] {'mo+':>6s} {'p':>6s} | {'Brier':>7s} {'dBrier':>8s} | {'AUC gated':>9s} {'d':>8s} | {'hit gated':>9s} {'d':>8s} | {'hit all':>7s}")
        for v in VARS:
            r = out[name][v]; vb = r.get("vs_base"); a = vb["auc"] if vb else None
            Q(f"{v:8s} {r['auc']['mean']:7.4f} {r['auc']['t_half']:6.2f} {fmt(a['mean']) if a else '':>9s} "
              f"[{(fmt(a['lo']) + ', ' + fmt(a['hi'])) if a else '':>17s}] {(str(a['pos']) + '/' + str(a['n'])) if a else '':>6s} {fmt(a['p'], 3) if a else '':>6s} | "
              f"{r['brier']['mean']:7.4f} {fmt(vb['brier']['mean'], 5) if vb else '':>8s} | {fmt(r['auc_gated']['mean']):>9s} {fmt(vb['auc_gated']['mean']) if vb else '':>8s} | "
              f"{fmt(r['hit_gated']['mean']):>9s} {fmt(vb['hit_gated']['mean']) if vb else '':>8s} | {r['hit_all']['mean']:7.4f}")
    Q("\npooled rows (J): " + "; ".join(f"{v}: n {r['n']}, gated {r['n_gated']}, gated hit {fmt(r['hit_gated'])}, up-rate {r['base_rate']:.4f}" for v, r in out["pooled_rows"].items()))
    Q("\nmonthly AUC (month: " + " / ".join(VARS) + ")")
    for m in mB + mC:
        Q(f"  {m} {tab['base'][m]['span']}: " + " ".join(f"{tab[v][m]['auc']:.4f}" for v in VARS) + f"   gated n {tab['base'][m]['n_gated']}/{tab[chosen][m]['n_gated']}  n {tab['base'][m]['n']}")
    open(TABLE, "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


# ------------------------------------------------------------------ freeze
def freeze():
    ch = D.choice_gate()
    res = [json.loads(x) for x in open(RES, encoding="utf-8")][-1]
    if not res["primary"]["pass_before_holm"]:
        raise SystemExit("REFUSED: the primary did not pass before Holm; nothing to freeze.")
    nice()
    if os.path.exists(FROZEN):
        raise SystemExit("REFUSED: frozen_P_d.json exists; the freeze is made once.")
    bars = P5.load_freeze_bars(); assert bars.index.max() <= P5.B_HI and bars.index.min() >= P5.B_LO
    log("freeze bars", len(bars), bars.index.min(), bars.index.max())
    art = D.freeze_fit(bars, P5.B_M0, ch["variant"]); json.dump(art, open(FROZEN, "w"), indent=1)
    open(os.path.join(HERE, "frozen_P_d.sha256"), "w").write(art["sha256"] + "\n")
    log("FROZEN sha256", art["sha256"], {v: (m["n_tr"], m["last_label_end"]) for v, m in art["models"].items()})


def retable():
    """Rewrite tables.txt from results.jsonl (no recomputation); used after a formatting change."""
    rec = [json.loads(x) for x in open(RES, encoding="utf-8")]; out = rec[-1]; chosen = out["chosen"]
    tab = {v: {} for v in VARS}
    for r in rec[:-1]:
        tab[r["variant"]][r["month"]] = r
    for name, ms in (("J", out["months_B"] + out["months_C"]), ("B", out["months_B"]), ("C", out["months_C"])):
        for v in VARS:
            out[name][v]["auc"]["t_half"] = D.mean_t(np.array([tab[v][m]["auc"] for m in ms]) - 0.5)["t"]
    write_tables(tab, out, out["months_B"], out["months_C"])


if __name__ == "__main__":
    {"select": select, "judge": judge, "freeze": freeze, "retable": retable}[sys.argv[1]]()
