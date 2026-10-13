"""O9 runner (SPEC.md). usage: run_o9.py [select|describe|freeze].
  select   : PHASE 1. Every grid configuration (32, the SPEC grid read literally) + META on 2021-01..2024-06, PESS fills primary, costs 0/2/5 ->
             selection.jsonl, tables_selection.txt, choice.json.
  describe : PHASE 2 (not implemented). Refuses unless choice.json is tracked at HEAD and unchanged.
  freeze   : PHASE 2 (not implemented). Same gate.
"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o9_lib as L  # noqa: E402
L8 = L.L8; P5 = L.P5
LOG = os.path.join(HERE, "run.log"); SEL = os.path.join(HERE, "selection.jsonl"); CHOICE = os.path.join(HERE, "choice.json")
TABLE = os.path.join(HERE, "tables_selection.txt")
MIN_TPM = 20                                                 # eligibility: at least 20 trades per month on average (PESS)


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


def load_all():
    s = L8.load_bars_cut(); DF = L8.load_dframe(); TF = L8.load_train_frame()
    X = L.load_train_extras(TF, s); DX = L.load_dec_extras(DF, s)
    return s, DF, TF, X, DX


def load_signals(TF, X, DF, DX):
    p = os.path.join(L.SCRATCH, "sig_select.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in ("sigma", "dv", "p")}
    S = L.signals(TF, X, DF, DX, L.SIG_MONTHS, log=log); np.savez(p, **S); return S


def run_cfg(c, rows, S, s, DF, ref_row, aL, aS, pess):
    O, Hi, Lo, Cl = s["open"].values, s["high"].values, s["low"].values, s["close"].values
    tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    tr = L.simulate9(O, Hi, Lo, Cl, DF["a"][rows], DF["m"][rows], S["sigma"][rows], c["k"], L.H, 1.0, allow_long=aL[rows] if aL is not None else None,
                     allow_short=aS[rows] if aS is not None else None, tau=c.get("tau"), tmin=tmin, pess=pess)
    r = rows[tr["j"]] if len(tr["j"]) else np.array([], int)
    w = L.size_weights(S["sigma"][r], ref_row[r]) if c["size"] == "vt" else np.ones(len(r))
    em = s.index[tr["ebar"]].to_period("M").astype(str).values if len(r) else np.array([], dtype=str)
    return tr, w, em


def select():
    nice(); t_all = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    s, DF, TF, X, DX = load_all(); log("loaded", len(DF["ta"]), "decisions", f"{time.time() - t_all:.0f}s")
    S = load_signals(TF, X, DF, DX); log("signals", int(np.isfinite(S["sigma"]).sum()), "valid", f"{time.time() - t_all:.0f}s")
    months = L.SEL_MONTHS; mo_of = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values
    rows = np.where(np.isin(mo_of, months))[0]; assert pd.Timestamp(DF["ta"][rows].max()) < pd.Timestamp("2024-07-01")
    ref = L.sigma_ref_by_month(S["sigma"], DF["ta"], L.SIG_MONTHS[1:]); ref_row = np.array([ref.get(m, np.nan) for m in mo_of])
    if os.path.exists(SEL):
        os.remove(SEL)
    res = []

    def record(c, trP, wP, emP, trO, wO, emO):
        sp = L.summarise9(trP, wP, emP, months); so = L.summarise9(trO, wO, emO, months)
        r = {"config": L.cfg_name(c), **c, "arm": "L1", **sp, "eligible": bool(sp["trades_per_month"] >= MIN_TPM),
             "orig": {k: so[k] for k in ("n_trades", "mean_m_2bps", "t_2bps", "months_pos_2bps", "hit_2bps", "max_dd_2bps", "mean_m_0bps", "mean_m_5bps")}}
        res.append(r)
        with open(SEL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        log(r["config"], "trades/mo", f"{r['trades_per_month']:.1f}", "m2%", f"{100 * r['mean_m_2bps']:+.3f}", "t", f"{r['t_2bps']:+.2f}", "elig", r["eligible"], f"{time.time() - t_all:.0f}s")

    cache = {}
    for c in L.grid():
        key = (c["k"], c["delta"], c["g"], c["tau"])
        if key not in cache:
            aL, aS = L.gate_masks(S["p"], S["dv"], c["delta"], c["g"])
            cache[key] = {pess: run_cfg(c, rows, S, s, DF, ref_row, aL, aS, pess) for pess in (True, False)}
        trP, emP = cache[key][True][0], cache[key][True][2]; trO, emO = cache[key][False][0], cache[key][False][2]
        rP = rows[trP["j"]] if len(trP["j"]) else np.array([], int); rO = rows[trO["j"]] if len(trO["j"]) else np.array([], int)
        wP = L.size_weights(S["sigma"][rP], ref_row[rP]) if c["size"] == "vt" else np.ones(len(rP))
        wO = L.size_weights(S["sigma"][rO], ref_row[rO]) if c["size"] == "vt" else np.ones(len(rO))
        record(c, trP, wP, emP, trO, wO, emO)
    # META: the candidate stream = BRK k = 1.5, ungated, PESS, 1x, over the pool months (2019-01 .. 2024-06)
    pool_rows = np.where(np.isin(mo_of, L.POOL_MONTHS))[0]
    O, Hi, Lo, Cl = s["open"].values, s["high"].values, s["low"].values, s["close"].values
    tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    ct = L.simulate9(O, Hi, Lo, Cl, DF["a"][pool_rows], DF["m"][pool_rows], S["sigma"][pool_rows], L.K_META, L.H, 1.0, tmin=tmin, pess=True)
    cand = {"row": pool_rows[ct["j"]], "side": ct["side"], "t_exit": (s.index[ct["xbar"]] + pd.Timedelta(minutes=1)).values.astype("datetime64[ns]"),
            "y": ((ct["ret"] - 2e-4) > 0).astype(int)}
    log("META candidates", len(cand["row"]), "hit", float(cand["y"].mean()), f"{time.time() - t_all:.0f}s")
    gcols = L.geom_cols()
    aL, aS, info = L.meta_gate(cand, DF, S["sigma"], S["dv"], S["p"], gcols, L.POOL_MONTHS, log=None)
    json.dump(info, open(os.path.join(HERE, "meta_info_selection.json"), "w"), indent=1)
    cm = {"meta": True, "k": L.K_META, "delta": None, "g": None, "tau": None, "size": "1x"}
    out = {p_: run_cfg(cm, rows, S, s, DF, ref_row, aL, aS, p_) for p_ in (True, False)}
    record(cm, out[True][0], out[True][1], out[True][2], out[False][0], out[False][1], out[False][2])
    elig = [r for r in res if r["eligible"]]; assert elig, "no eligible configuration"
    best = max(elig, key=lambda r: r["t_2bps"])
    ch = {"rule": "highest t-statistic of the monthly net return at 2 bps (PESS fills) on 2021-01..2024-06 among configurations with >= 20 trades per month (SPEC.md)",
          "config": best["config"], "cfg": {k: best[k] for k in ("k", "delta", "g", "tau", "size")} | ({"meta": True} if best["config"].startswith("META") else {}),
          "t_2bps": best["t_2bps"], "mean_m_2bps": best["mean_m_2bps"], "trades_per_month": best["trades_per_month"], "n_trades": best["n_trades"],
          "n_months": best["n_months"], "n_eligible": len(elig), "n_configs": len(res), "fills": "PESS", "selection_months": [months[0], months[-1]]}
    tables(res, ch)
    json.dump(ch, open(CHOICE, "w"), indent=1); log("CHOICE", ch); log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime_select.json"), "w"))


def tables(res, ch):
    o = ["O9 selection: SPEC grid (32) + META, 42 months 2021-01..2024-06, PESS fills primary. Returns: mean monthly net return in % of capital (sum of w x position log returns, no compounding).",
         "eligible = at least 20 trades per month on average (PESS). Rank: t-statistic of the monthly net at 2 bps. orig = original fills (O8's primary variant)."]
    o.append(f"{'config':36s} {'el':>2s} {'tr/mo':>6s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t(2bp)':>7s} {'mo+':>5s} {'hit%':>6s} {'maxDD%':>7s} {'tgt/stop/time':>15s} {'avg w':>6s} | {'orig m2%':>8s} {'orig t':>7s} {'orig tr':>7s} {'orig hit':>8s}")
    for r in sorted(res, key=lambda r: -r["t_2bps"]):
        rs = r["reasons"]; ex = f"{rs.get('target', 0)}/{rs.get('stop', 0)}/{rs.get('time', 0)}"; og = r["orig"]
        hit = "n/a" if r["hit_2bps"] is None else f"{100 * r['hit_2bps']:.1f}"; oh = "n/a" if og["hit_2bps"] is None else f"{100 * og['hit_2bps']:.1f}"
        out_w = "n/a" if r["mean_w"] is None else f"{r['mean_w']:.2f}"
        o.append(f"{r['config']:36s} {'Y' if r['eligible'] else 'n':>2s} {r['trades_per_month']:6.1f} {100 * r['mean_m_0bps']:+7.3f} {100 * r['mean_m_2bps']:+7.3f} {100 * r['mean_m_5bps']:+7.3f} "
                 f"{r['t_2bps']:+7.2f} {r['months_pos_2bps']:2d}/{r['n_months']:<2d} {hit:>6s} {100 * r['max_dd_2bps']:7.2f} {ex:>15s} {out_w:>6s} | "
                 f"{100 * og['mean_m_2bps']:+8.3f} {og['t_2bps']:+7.2f} {og['n_trades']:7d} {oh:>8s}")
    o.append(f"\nCHOICE: {ch['config']}  t={ch['t_2bps']:+.2f}  mean monthly net {100 * ch['mean_m_2bps']:+.3f}%  {ch['trades_per_month']:.1f} trades/month  ({ch['n_eligible']} of {ch['n_configs']} eligible)")
    o.append("\nMarginal value of each component (descriptive; paired over the other settings; PESS, 2 bps; mean difference ON - OFF in monthly net % and in t)")
    ens = [r for r in res if r["config"].startswith("ENS")]
    key = lambda r: (r["k"], r["delta"], r["g"], r["tau"], r["size"])
    idx = {key(r): r for r in ens}
    for fac, on, off, lab in (("delta", 0.02, None, "regression gate p_up >= 0.52 / <= 0.48"), ("g", 0.0, None, "dV gate (dV >= 0)"),
                              ("tau", 2.0, None, "target at 2 sigma"), ("size", "vt", "1x", "vol-targeted size (cap 2)")):
        dm, dt, n_ = [], [], 0
        for r in ens:
            if r[fac] == off:
                k2 = dict(zip(("k", "delta", "g", "tau", "size"), key(r))); k2[fac] = on; r2 = idx[tuple(k2[x] for x in ("k", "delta", "g", "tau", "size"))]
                dm.append(r2["mean_m_2bps"] - r["mean_m_2bps"]); dt.append(r2["t_2bps"] - r["t_2bps"]); n_ += 1
        o.append(f"  {lab:42s} pairs {n_:2d}   d(mean monthly net) {100 * np.mean(dm):+.3f}%   d(t) {np.mean(dt):+.2f}   better in {sum(x > 0 for x in dm)}/{n_}")
    for k_ in (1.0, 1.5):
        plain = idx[(k_, None, None, None, "1x")]
        o.append(f"  plain BRK k={k_} (no gate, no target, 1x): m2 {100 * plain['mean_m_2bps']:+.3f}%  t {plain['t_2bps']:+.2f}  trades/mo {plain['trades_per_month']:.1f}")
    meta = [r for r in res if r["config"].startswith("META")]
    if meta:
        p = idx[(1.5, None, None, None, "1x")]; m = meta[0]
        o.append(f"  META vs plain BRK k=1.5: d(m2) {100 * (m['mean_m_2bps'] - p['mean_m_2bps']):+.3f}%  d(t) {m['t_2bps'] - p['t_2bps']:+.2f}  trades/mo {m['trades_per_month']:.1f} vs {p['trades_per_month']:.1f}")
    open(TABLE, "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


# ------------------------------------------------------------------ the gate of PHASE 2
def git(*args):
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True)


def choice_gate():
    """Refuse PHASE 2 unless choice.json is tracked at HEAD and unchanged in the working tree."""
    if git("ls-files", "--error-unmatch", "--", CHOICE).returncode != 0:
        raise SystemExit("REFUSED: choice.json is not tracked in git. Commit it before any other span is computed.")
    if git("diff", "--quiet", "HEAD", "--", CHOICE).returncode != 0:
        raise SystemExit("REFUSED: choice.json differs from HEAD (changed after its commit).")
    if git("cat-file", "-e", "HEAD:./choice.json").returncode != 0:
        raise SystemExit("REFUSED: choice.json is not in HEAD.")
    return json.load(open(CHOICE, encoding="utf-8"))


def describe():
    choice_gate(); raise SystemExit("PHASE 2 not implemented")


def freeze():
    choice_gate(); raise SystemExit("PHASE 2 not implemented")


if __name__ == "__main__":
    {"select": select, "describe": describe, "freeze": freeze}[sys.argv[1]]()
