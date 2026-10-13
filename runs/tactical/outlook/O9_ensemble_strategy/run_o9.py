"""O9 runner (SPEC.md). usage: run_o9.py [select|describe|freeze].
  select   : PHASE 1. Every grid configuration (32, the SPEC grid read literally) + META on 2021-01..2024-06, PESS fills primary, costs 0/2/5 ->
             selection.jsonl, tables_selection.txt, choice.json.
  describe : PHASE 2. A, B, C descriptive runs for PRIMARY (choice.json) and ROBUST (choice_robust.json). Both choice files must be committed and unchanged.
  freeze   : PHASE 2. Fit every model once on the Binance file to 2026-09-30 23:59 -> frozen_P.json, frozen_P.sha256.
"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o9_lib as L  # noqa: E402
L8 = L.L8; P5 = L.P5
LOG = os.path.join(HERE, "run.log"); SEL = os.path.join(HERE, "selection.jsonl"); CHOICE = os.path.join(HERE, "choice.json"); CHOICE_R = os.path.join(HERE, "choice_robust.json")
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
    """Refuse PHASE 2 unless choice.json and choice_robust.json are tracked at HEAD and unchanged in the working tree. Returns both."""
    out = []
    for path in (CHOICE, CHOICE_R):
        nm = os.path.basename(path)
        if git("ls-files", "--error-unmatch", "--", path).returncode != 0:
            raise SystemExit(f"REFUSED: {nm} is not tracked in git. Commit it before any other span is computed.")
        if git("diff", "--quiet", "HEAD", "--", path).returncode != 0:
            raise SystemExit(f"REFUSED: {nm} differs from HEAD (changed after its commit).")
        if git("cat-file", "-e", f"HEAD:./{nm}").returncode != 0:
            raise SystemExit(f"REFUSED: {nm} is not in HEAD.")
        out.append(json.load(open(path, encoding="utf-8")))
    return out


# ------------------------------------------------------------------ PHASE 2
import o9_p2 as P2  # noqa: E402
FWD_END = pd.Timestamp("2026-10-12 23:59")
SPAN_NAMES = {"A": "A 2018-07..2020-12 (walk-forward)", "B": "B 2024-07..2025-07-20 (walk-forward)", "C": "C 2025-10-05..2026-10-12 (frozen fit to 2025-07-20)"}
FILLS = {"PESS": True, "orig": False}
COL = {"PRIMARY/FULL": "#d55181", "ROBUST/FULL": "#2a9d8f", "PRIMARY/SHUF": "#9085e9", "ROBUST/SHUF": "#c98500", "BASE": "#9a9890"}


def span_months(sp):
    return L8.SPANS[sp] if sp in L8.SPANS else [str(m) for m in pd.period_range("2025-10", "2026-10", freq="M")]


def prepare_span(sp, ctx):
    """bars, decision frame, decision-time tb7 and the four signals for a span (A, B: monthly walk-forward on the cut file; C: one fit at the cut on
    features from forward-file bars only). The month before the span is included for sigma_ref."""
    months = span_months(sp); prev = str(pd.Period(months[0], freq="M") - 1); allm = [prev] + months if sp != "C" else months
    if sp == "C":
        s = L8.T.load_bars(L8.T.FWD_CSV, L8.T.FWD_START, FWD_END)                      # refuses a bar of 2025-07-21..2025-09-30
        assert s.index.min() >= L8.T.FWD_START and s.index.max() <= FWD_END
        DF, DX = P2.make_dframe_for(s, L8.P5.FREEZE_FIRST)
        S = P2.signals_arms(ctx["TF"], ctx["X"], DF, DX, allm, frozen_m0=L8.O3.TRAIN_END)
    else:
        if "cut" not in ctx:
            s = L8.load_bars_cut(); DF = L8.load_dframe(); ctx["cut"] = (s, DF, L.load_dec_extras(DF, s))
        s, DF, DX = ctx["cut"]
        cache = os.path.join(L.SCRATCH, f"sig_{sp}.npz")
        if os.path.exists(cache):
            z = np.load(cache); S = {k: z[k] for k in ("L1", "R0", "dv", "p")}
        else:
            S = P2.signals_arms(ctx["TF"], ctx["X"], DF, DX, allm); np.savez(cache, **S)
    return s, DF, S, months, allm


def run_span(sp, ctx, cands):
    s, DF, S, months, allm = prepare_span(sp, ctx)
    mo_of = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values
    rows = np.where(np.isin(mo_of, months) & np.isfinite(S["L1"]))[0]
    out = {}
    arms = [(f"{nm}/{a}", c, a) for nm, c in cands.items() for a in P2.ARMS9] + [("BASE", P2.BASE_CFG, "FULL")]
    for key, c, a in arms:
        sig, dv, p = P2.arm_inputs(S, a, DF["ta"], allm); ref = P2.ref_rows(sig, DF["ta"], allm); cfg = P2.arm_cfg(c, a)
        for fn, pess in FILLS.items():
            out[(key, fn)] = P2.trades9(s, DF, rows, sig, dv, p, ref, cfg, pess)
    return s, DF, S, out, months, rows


def buy_hold(s, months):
    lo = L8.P5.month_bounds(months[0])[0]; hi = L8.P5.month_bounds(months[-1])[1]
    x = s[(s.index >= lo) & (s.index < hi)]
    cum = np.log(x["close"].resample("D").last().dropna().values) - np.log(x["open"].iloc[0]); z = np.concatenate([[0.0], cum])
    return {"log_return": float(np.log(x["close"].iloc[-1] / x["open"].iloc[0])), "max_dd_daily": float((np.maximum.accumulate(z) - z).max()), "from": str(x.index[0]), "to": str(x.index[-1])}


def descr9(tf, months, s):
    d = {"n_trades": int(len(tf)), "months": list(months)}
    for c in L8.COSTS_BPS:
        mr = P2.monthly_tf(tf, months, c); d[f"monthly_{int(c)}bps"] = mr.tolist(); d[f"mean_m_{int(c)}bps"] = float(mr.mean())
    m2 = np.array(d["monthly_2bps"]); d["t_2bps"] = L8.t_stat(m2); mu, lo, hi, _ = L8.T.ci(m2); d["ci_2bps"] = [mu, lo, hi]; d["months_pos_2bps"] = int((m2 > 0).sum())
    net = tf["w"].values * (tf["ret_gross"].values - 2e-4) if len(tf) else np.array([])
    d["hit"] = float((net > 0).mean()) if len(net) else None
    d["mean_trade_bps"] = float(net.mean() * 1e4) if len(net) else None
    d["max_dd"] = L8.max_drawdown(net)
    lo_ = L8.P5.month_bounds(months[0])[0]; hi_ = min(L8.P5.month_bounds(months[-1])[1], s.index.max() + pd.Timedelta(minutes=1))
    days = pd.date_range(lo_.normalize(), (hi_ - pd.Timedelta(minutes=1)).normalize(), freq="D"); dr = P2.daily_net(tf, days, 2.0)
    d["sharpe_daily"] = float(dr.mean() / dr.std(ddof=1) * np.sqrt(365)) if dr.std(ddof=1) > 0 else None
    yrs = len(days) / 365.25; d["total_net"] = float(net.sum()); d["ann_net"] = float(net.sum() / yrs); d["calmar"] = float(d["ann_net"] / d["max_dd"]) if d["max_dd"] > 0 else None
    d["reasons"] = tf["reason"].value_counts().to_dict() if len(tf) else {}
    d["mean_w"] = float(tf["w"].mean()) if len(tf) else None; d["long_share"] = float((tf["side"] == 1).mean()) if len(tf) else None
    d["target_met"] = bool(d["hit"] is not None and d["hit"] > 0.60 and d["max_dd"] < 0.05)
    return d


def check_robust(ch2):
    """The ROBUST candidate's selection numbers from selection.jsonl against choice_robust.json and against the rule (SPEC addendum)."""
    rows = [json.loads(x) for x in open(SEL, encoding="utf-8")]
    elig = [r for r in rows if r["eligible"] and r["mean_m_5bps"] > 0]; best = max(elig, key=lambda r: r["t_2bps"])
    c = ch2["config"]; mine = [r for r in rows if r["config"] == L.cfg_name(c)][0]
    rep = {"rule_pick": best["config"], "choice_robust_config": mine["config"], "satisfies_rule": best["config"] == mine["config"],
           "selection_jsonl": {"t_2bps": mine["t_2bps"], "m2_pct": 100 * mine["mean_m_2bps"], "m5_pct": 100 * mine["mean_m_5bps"], "trades_per_month": mine["trades_per_month"],
                               "hit_pct": 100 * mine["hit_2bps"], "maxdd_pct": 100 * mine["max_dd_2bps"]}, "choice_robust_json": ch2["selection"]}
    return rep


def describe():
    ch1, ch2 = choice_gate()
    nice(); t_all = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    rep = check_robust(ch2); log("ROBUST check", rep)
    cands = {"PRIMARY": ch1["cfg"], "ROBUST": ch2["config"]}
    out_p = os.path.join(HERE, "describe.jsonl")
    if os.path.exists(out_p):
        raise SystemExit("REFUSED: describe.jsonl exists; the descriptive run is made once.")
    TF = L8.load_train_frame(); s_cut = L8.load_bars_cut(); X = L.load_train_extras(TF, s_cut); ctx = {"TF": TF, "X": X}
    rec = []; res = {}; trades = {}; bh = {}; frames_for_windows = {}
    for sp in ("B", "C", "A"):
        s, DF, S, out, months, rows = run_span(sp, ctx, cands); log("span", sp, "valid decisions", len(rows), f"{time.time() - t_all:.0f}s")
        bh[sp] = buy_hold(s, months); trades[sp] = out; allf = []
        for (key, fn), tf in out.items():
            d = descr9(tf, months, s); d.update({"span": sp, "arm": key, "fills": fn}); res[(sp, key, fn)] = d; rec.append(d); allf.append(tf.assign(arm=key, fills=fn))
        pd.concat(allf).to_csv(os.path.join(HERE, f"trades_{sp}.csv.gz"), index=False)
        if sp in ("A", "B"):
            frames_for_windows[sp] = (s, DF)
    with open(out_p, "w", encoding="utf-8") as f:
        for d in rec:
            f.write(json.dumps(d) + "\n")
        f.write(json.dumps({"verdict": "descriptive", "buy_hold": bh, "candidates": cands, "robust_check": rep}) + "\n")
    write_tables(res, bh, cands, rep)
    make_figures(trades)
    make_windows(trades, frames_for_windows, cands)
    log("describe done", f"{time.time() - t_all:.0f}s"); json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime_describe.json"), "w"))


def write_tables(res, bh, cands, rep):
    o = []; Q = o.append; f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"
    Q("O9 DESCRIPTIVE results on spans already seen by earlier studies (A, B, C). No verdict: these spans informed the design (SPEC \"Honest framing\").")
    Q(f"candidates: PRIMARY {cands['PRIMARY']}  |  ROBUST {cands['ROBUST']}  |  BASE = O8's BRK_k1.5_H60_s1.0")
    Q(f"ROBUST check against selection.jsonl: satisfies the rule: {rep['satisfies_rule']}; selection.jsonl {rep['selection_jsonl']}; choice_robust.json {rep['choice_robust_json']}")
    Q("Returns in % of capital per month: sum of w x position log returns, no compounding. t = t-statistic of the monthly net at 2 bps; CI = 95% t-interval of the monthly mean.")
    for sp in ("B", "C", "A"):
        any_ = res[(sp, "BASE", "PESS")]
        Q(f"\n===== {SPAN_NAMES[sp]}; {len(any_['months'])} months; buy-and-hold log return {100 * bh[sp]['log_return']:+.1f}% (daily-close max DD {100 * bh[sp]['max_dd_daily']:.1f}%) =====")
        Q(f"{'arm':14s} {'fills':5s} {'trades':>6s} {'hit%':>5s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t2':>6s} {'CI2 lo':>7s} {'CI2 hi':>7s} {'mo+':>6s} {'Sharpe':>6s} {'maxDD%':>7s} {'Calmar':>7s} {'tgt/stop/time':>14s} {'w':>5s} owner target")
        for key in [f"{n}/{a}" for n in cands for a in P2.ARMS9] + ["BASE"]:
            for fn in FILLS:
                d = res[(sp, key, fn)]; r = d["reasons"]
                Q(f"{key:14s} {fn:5s} {d['n_trades']:6d} {f(None if d['hit'] is None else 100 * d['hit'], 1):>5s} {100 * d['mean_m_0bps']:+7.3f} {100 * d['mean_m_2bps']:+7.3f} {100 * d['mean_m_5bps']:+7.3f} "
                  f"{d['t_2bps']:+6.2f} {100 * d['ci_2bps'][1]:+7.3f} {100 * d['ci_2bps'][2]:+7.3f} {d['months_pos_2bps']:2d}/{len(d['months']):<2d} {f(d['sharpe_daily']):>6s} {100 * d['max_dd']:7.2f} "
                  f"{f(d['calmar']):>7s} {r.get('target', 0)}/{r.get('stop', 0)}/{r.get('time', 0):<5} {f(d['mean_w']):>5s} {'MET' if d['target_met'] else 'not met'}")
        for nm in cands:
            Q(f"monthly net at 2 bps, {nm}/FULL PESS (%): " + " ".join(f"{m[2:]}:{100 * x:+.1f}" for m, x in zip(res[(sp, f'{nm}/FULL', 'PESS')]['months'], res[(sp, f'{nm}/FULL', 'PESS')]['monthly_2bps'])))
    Q("\nNote: span C's first month (2025-10) has no previous month inside the forward file, so a vol-targeted size uses w = 1 there. NO_DV for ROBUST equals FULL (its dV gate is already off).")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


def curve(tf, cost=2.0):
    if len(tf) == 0:
        return np.array([]), np.array([]), np.array([])
    t = tf["exit_time"].values; c = np.cumsum(tf["w"].values * (tf["ret_gross"].values - cost * 1e-4)); dd = np.maximum.accumulate(np.maximum(c, 0)) - c
    return t, c, dd


def make_figures(trades):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "src"))
    from neural_trade.visualization import theme as TH
    keys = [("PRIMARY/FULL", "PESS", "-"), ("PRIMARY/FULL", "orig", ":"), ("ROBUST/FULL", "PESS", "-"), ("ROBUST/FULL", "orig", ":"), ("PRIMARY/SHUF", "PESS", "-"), ("ROBUST/SHUF", "PESS", "-"), ("BASE", "PESS", "-")]
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), facecolor=TH.PAPER)
    for j, sp in enumerate(("B", "C", "A")):
        for key, fn, ls in keys:
            t, c, dd = curve(trades[sp][(key, fn)])
            if len(t):
                lw = 1.7 if key.endswith("FULL") else 1.0
                ax[0, j].plot(t, 100 * c, ls, color=COL[key], lw=lw, label=f"{key} {fn}"); ax[1, j].plot(t, -100 * dd, ls, color=COL[key], lw=lw)
        ax[0, j].set_title(SPAN_NAMES[sp], color=TH.INK, fontsize=10)
        ax[0, j].set_ylabel("cumulative net return, 2 bps (% of capital)", color=TH.MUTED); ax[1, j].set_ylabel("drawdown (% of capital)", color=TH.MUTED)
        for r in (0, 1):
            a = ax[r, j]; a.set_facecolor(TH.SURFACE); a.grid(color=TH.GRID, lw=0.6); a.tick_params(colors=TH.MUTED); a.axhline(0, color=TH.NEUTRAL, lw=0.8)
            for sp_ in a.spines.values():
                sp_.set_color(TH.AXIS)
    ax[0, 0].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK, fontsize=7)
    fig.suptitle("O9 descriptive: equity and drawdown (w x position returns in order, no compounding). Solid = PESS fills, dotted = original fills", color=TH.INK)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "equity_dd.png"), dpi=110, facecolor=TH.PAPER); plt.close(fig)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pf = make_subplots(rows=2, cols=3, subplot_titles=[SPAN_NAMES[x] for x in ("B", "C", "A")] + [""] * 3)
    for j, sp in enumerate(("B", "C", "A"), 1):
        for key, fn, ls in keys:
            t, c, dd = curve(trades[sp][(key, fn)])
            if len(t):
                kw = dict(x=t, mode="lines", line=dict(color=COL[key], dash="solid" if ls == "-" else "dot"), name=f"{key} {fn}", legendgroup=f"{key}{fn}", showlegend=(j == 1))
                pf.add_trace(go.Scatter(y=100 * c, **kw), row=1, col=j); kw["showlegend"] = False; pf.add_trace(go.Scatter(y=-100 * dd, **kw), row=2, col=j)
    pf.update_layout(template="plotly_dark", paper_bgcolor=TH.PAPER, plot_bgcolor=TH.SURFACE, height=700, title="O9 descriptive equity (top, % of capital, 2 bps) and drawdown (bottom)")
    open(os.path.join(HERE, "equity_dd.plotly.json"), "w", encoding="utf-8").write(pf.to_json())


def make_windows(trades, frames, cands):
    """Eight real example trades for the report (a winner and a loser per candidate and cut-file span; PESS, FULL arms) from cut-file spans only (bars <= 2025-07-20)."""
    out = {"note": "cut-file bars only (<= 2025-07-20)", "examples": []}
    rng = np.random.default_rng(7)
    for nm in ("PRIMARY", "ROBUST"):
        for sp, win in (("B", True), ("B", False), ("A", True), ("A", False)):
            tf = trades[sp][(f"{nm}/FULL", "PESS")]; s, DF = frames[sp]
            if len(tf) == 0:
                continue
            net = tf["w"].values * (tf["ret_gross"].values - 2e-4); want = (net > 0) if win else (net <= 0)
            idx = np.where(want)[0]
            if len(idx) == 0:
                continue
            i = int(rng.choice(idx)); r = tf.iloc[i]
            O, Hi, Lo, Cl = s["open"].values, s["high"].values, s["low"].values, s["close"].values
            d = int(s.index.get_loc(r["decision_time"])); e = int(s.index.get_loc(r["entry_time"])); x = int(s.index.get_loc(r["exit_time"]))
            m = float(O[d]); sg = float(r["sigma"]); c = cands[nm]; side = int(r["side"])
            out["examples"].append({"candidate": nm, "span": sp, "decision_time": str(r["decision_time"]), "m": m, "sigma": sg, "p_up": float(r["p_up"]), "dV": float(r["dV"]), "w": float(r["w"]),
                                    "buy_stop_U": m * (1 + c["k"] * sg), "sell_stop_L": m * (1 - c["k"] * sg), "side": side, "entry_time": str(r["entry_time"]), "entry_px": float(r["entry_px"]),
                                    "stop_px": float(r["entry_px"]) * (1 - side * sg), "target_px": None if c["tau"] is None else float(r["entry_px"]) * (1 + side * c["tau"] * sg),
                                    "exit_time": str(r["exit_time"]), "exit_px": float(r["exit_px"]), "reason": str(r["reason"]), "gross_ret": float(r["ret_gross"]),
                                    "bars": [{"t": str(s.index[j]), "o": float(O[j]), "h": float(Hi[j]), "l": float(Lo[j]), "c": float(Cl[j])} for j in range(max(0, d - 8), min(len(O) - 1, x + 12) + 1)]})
    json.dump(out, open(os.path.join(HERE, "windows.json"), "w"), indent=1)


def freeze():
    choice_gate(); nice()
    out_p = os.path.join(HERE, "frozen_P.json")
    if os.path.exists(out_p):
        raise SystemExit("REFUSED: frozen_P.json exists; the freeze is made once.")
    bars = P5.load_freeze_bars(); assert bars.index.max() <= P5.B_HI and bars.index.min() >= P5.B_LO
    log("freeze bars", len(bars), bars.index.min(), bars.index.max())
    art = P2.freeze_fit(bars, P5.B_M0); json.dump(art, open(out_p, "w"), indent=1)
    open(os.path.join(HERE, "frozen_P.sha256"), "w").write(art["sha256"] + "\n")
    log("FROZEN sha256", art["sha256"], "sigma n_tr", art["sigma"]["n_tr"], "dv n_tr", art["dv"]["n_tr"], "pup n_tr", art["pup"]["n_tr"], "last label ends", art["sigma"]["last_label_end"], art["dv"]["last_label_end"], art["pup"]["last_label_end"])



if __name__ == "__main__":
    {"select": select, "describe": describe, "freeze": freeze}[sys.argv[1]]()
