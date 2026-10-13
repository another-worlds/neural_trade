"""O10 runner (SPEC.md). usage: run_o10.py [select|describe|freeze].
  select   : the 14 exit configurations on 2021-01..2024-06 (PESS fills primary, original fills beside; costs 0/2/5 bps) -> selection.jsonl,
             tables_selection.txt, choice.json (max t-statistic of the monthly net at 2 bps, PESS).
  describe : A, B, C descriptive runs for the choice, E0 (ROBUST's own exit, T 60), E0 at the choice's T, and SHUF. choice.json must be committed and unchanged.
  freeze   : all models fitted once on the Binance file to 2026-09-30 23:59 (O9's models are reused, identical fit) + sigma_15 + E4 -> frozen_P.json + sha256.
"""
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o10_lib as O  # noqa: E402
L9 = O.L9; L8 = O.L8; P5 = O.P5; P2 = O.P2; O3 = O.O3
LOG = os.path.join(HERE, "run.log"); SEL = os.path.join(HERE, "selection.jsonl"); CHOICE = os.path.join(HERE, "choice.json"); TABLE = os.path.join(HERE, "tables_selection.txt")
OUT = HERE                                                    # where tables, figures and windows are written (tests redirect it)
O9SCR = L9.SCRATCH
FWD_END = pd.Timestamp("2026-10-12 23:59")
SPAN_NAMES = {"A": "A 2018-07..2020-12 (walk-forward)", "B": "B 2024-07..2025-07-20 (walk-forward)", "C": "C 2025-10-05..2026-10-12 (frozen fit to 2025-07-20)"}
FILLS = {"PESS": True, "orig": False}
SIGNAL_REASONS = ("trail", "ema", "geo", "vol", "learn")


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
    X = L9.load_train_extras(TF, s); DX = L9.load_dec_extras(DF, s)
    return s, DF, TF, X, DX


def cached(path, fn):
    os.makedirs(O.SCRATCH, exist_ok=True)
    if os.path.exists(path):
        z = np.load(path); return {k: z[k] for k in z.files}
    out = fn(); np.savez(path, **out); return out


def sel_signals(TF, X, DF, DX):
    z = np.load(os.path.join(O9SCR, "sig_select.npz")); S = {"sig": z["sigma"], "dv": z["dv"], "p": z["p"]}      # O9's cache: the same walk-forward signals
    S["s15"] = cached(os.path.join(O.SCRATCH, "sig15_select.npz"), lambda: {"s15": O.sigma15_walk(TF, DF, O.SIG_MONTHS)})["s15"]
    return S


# ------------------------------------------------------------------ selection
def sel_row(c, sp_, so_):
    r = {"config": O.cfg_name(c), **c, **sp_, "orig": {k: so_[k] for k in ("n_trades", "mean_m_2bps", "t_2bps", "months_pos_2bps", "hit_2bps", "max_dd_2bps", "mean_m_0bps", "mean_m_5bps")}}
    return r


def select():
    nice(); t0 = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    s, DF, TF, X, DX = load_all(); S = sel_signals(TF, X, DF, DX); log("signals", int(np.isfinite(S["sig"]).sum()), "valid", int(np.isfinite(S["s15"]).sum()), "s15", f"{time.time() - t0:.0f}s")
    ctx = O.make_ctx(s, DF, S, O.gcols()); mo_of = ctx["mo_row"]; months = O.SEL_MONTHS
    rows = np.where(np.isin(mo_of, months))[0]; pool_rows = np.where(np.isin(mo_of, O.POOL_MONTHS))[0]
    assert pd.Timestamp(DF["ta"][rows].max()) < pd.Timestamp("2024-07-01")
    aL, aS = L9.gate_masks(S["p"], S["dv"], O.ENTRY["delta"], O.ENTRY["g"])
    recs = {}; models = {}
    for Tt in O.TS:                                           # E4's training checkpoints: E0 positions of the pool months (2019-01..2024-06), PESS
        tr = O.simulate10(ctx, pool_rows, O.spec_of({"exit": "E0", "T": Tt}), S["sig"], S["sig"], aL, aS, pess=True)
        recs[Tt] = O.e4_records(ctx, tr); np.savez(os.path.join(O.SCRATCH, f"recs_sel_T{Tt}.npz"), **recs[Tt])
        models[Tt] = O.e4_models(recs[Tt], months); log("E4 pool T", Tt, "E0 positions", len(tr["dec"]), "checkpoints", len(recs[Tt]["y"]), "models", len(models[Tt]), f"{time.time() - t0:.0f}s")
    if os.path.exists(SEL):
        os.remove(SEL)
    res = []
    for c in O.grid():
        out = {}
        for fn, pess in FILLS.items():
            sp = O.spec_of(c, models=models[c["T"]] if c["exit"] == "E4" else None)
            out[fn] = O.trades10(ctx, rows, sp, S, S["sig"], aL, aS, pess)[0]
        r = sel_row(c, O.summarise10(out["PESS"], months), O.summarise10(out["orig"], months)); r["eligible"] = True
        res.append(r)
        with open(SEL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        log(r["config"], "trades/mo", f"{r['trades_per_month']:.1f}", "m2%", f"{100 * r['mean_m_2bps']:+.3f}", "t", f"{r['t_2bps']:+.2f}", f"{time.time() - t0:.0f}s")
    best = max(res, key=lambda r: r["t_2bps"])
    ch = {"rule": "highest t-statistic of the monthly net return at 2 bps (PESS fills) on 2021-01..2024-06 among the 14 configurations (SPEC.md)", "config": best["config"],
          "cfg": {"exit": best["exit"], "T": best["T"]}, "entry": O.ENTRY, "t_2bps": best["t_2bps"], "mean_m_2bps": best["mean_m_2bps"], "trades_per_month": best["trades_per_month"],
          "n_trades": best["n_trades"], "n_months": best["n_months"], "n_configs": len(res), "fills": "PESS", "selection_months": [months[0], months[-1]]}
    tables(res, ch)
    json.dump(ch, open(CHOICE, "w"), indent=1); log("CHOICE", ch); log("total", f"{time.time() - t0:.0f}s")
    json.dump({"runtime_s": time.time() - t0}, open(os.path.join(HERE, "runtime_select.json"), "w"))


def tables(res, ch):
    o = ["O10 selection: 14 exit configurations on ROBUST's entry, 42 months 2021-01..2024-06, PESS fills primary. Returns: mean monthly net return in % of capital (sum of position log returns, no compounding).",
         "Rank: t-statistic of the monthly net at 2 bps. win/loss = average net trade (2 bps) in bps; hold = mean bars from entry to exit. orig = original fills (O8's primary variant)."]
    o.append(f"{'config':10s} {'tr/mo':>6s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t(2bp)':>7s} {'mo+':>5s} {'hit%':>6s} {'win':>6s} {'loss':>6s} {'hold':>6s} {'maxDD%':>7s} {'reasons':>34s} | {'orig m2%':>8s} {'orig t':>7s} {'orig tr':>7s} {'orig hit':>8s}")
    f1 = lambda x, p=1: "n/a" if x is None else f"{x:.{p}f}"
    for r in sorted(res, key=lambda r: -r["t_2bps"]):
        og = r["orig"]; rs = " ".join(f"{k}:{v}" for k, v in sorted(r["reasons"].items()))
        o.append(f"{r['config']:10s} {r['trades_per_month']:6.1f} {100 * r['mean_m_0bps']:+7.3f} {100 * r['mean_m_2bps']:+7.3f} {100 * r['mean_m_5bps']:+7.3f} {r['t_2bps']:+7.2f} "
                 f"{r['months_pos_2bps']:2d}/{r['n_months']:<2d} {f1(None if r['hit_2bps'] is None else 100 * r['hit_2bps'])}{'':>0s} {f1(r['avg_win_bps']):>6s} {f1(r['avg_loss_bps']):>6s} {f1(r['mean_hold_min']):>6s} "
                 f"{100 * r['max_dd_2bps']:7.2f} {rs:>34s} | {100 * og['mean_m_2bps']:+8.3f} {og['t_2bps']:+7.2f} {og['n_trades']:7d} {f1(None if og['hit_2bps'] is None else 100 * og['hit_2bps']):>8s}")
    o.append(f"\nCHOICE: {ch['config']}  t={ch['t_2bps']:+.2f}  mean monthly net {100 * ch['mean_m_2bps']:+.3f}%  {ch['trades_per_month']:.1f} trades/month  ({ch['n_configs']} configurations)")
    by = {r["config"]: r for r in res}
    o.append("\nPaired against E0 at the same T (descriptive; mean monthly net at 2 bps and t, PESS)")
    for Tt in O.TS:
        b = by[f"E0_T{Tt}"]
        o.append(f"  T={Tt}: E0 m2 {100 * b['mean_m_2bps']:+.3f}% t {b['t_2bps']:+.2f} | " + " | ".join(
            f"{e} {100 * (by[f'{e}_T{Tt}']['mean_m_2bps'] - b['mean_m_2bps']):+.3f}%" for e in O.EXITS[1:]))
    open(TABLE, "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


# ------------------------------------------------------------------ the gate of PHASE 2
def git(*args):
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True)


def choice_gate():
    """Refuse PHASE 2 unless choice.json is tracked at HEAD and unchanged in the working tree. Returns it."""
    nm = os.path.basename(CHOICE)
    if git("ls-files", "--error-unmatch", "--", CHOICE).returncode != 0:
        raise SystemExit(f"REFUSED: {nm} is not tracked in git. Commit it before any other span is computed.")
    if git("diff", "--quiet", "HEAD", "--", CHOICE).returncode != 0:
        raise SystemExit(f"REFUSED: {nm} differs from HEAD (changed after its commit).")
    if git("cat-file", "-e", f"HEAD:./{nm}").returncode != 0:
        raise SystemExit(f"REFUSED: {nm} is not in HEAD.")
    return json.load(open(CHOICE, encoding="utf-8"))


# ------------------------------------------------------------------ PHASE 2
def span_months(sp):
    return L8.SPANS[sp] if sp in L8.SPANS else [str(m) for m in pd.period_range("2025-10", "2026-10", freq="M")]


def prepare_span(sp, TF, X, cut):
    """bars, decision frame and the four signals (sig, dv, p, s15) of a span; A, B: monthly walk-forward on the cut file; C: one fit at the cut on forward-file bars only."""
    months = span_months(sp); prev = str(pd.Period(months[0], freq="M") - 1); allm = [prev] + months if sp != "C" else months
    if sp == "C":
        s = L8.T.load_bars(L8.T.FWD_CSV, L8.T.FWD_START, FWD_END)                      # refuses a bar of 2025-07-21..2025-09-30
        assert s.index.min() >= L8.T.FWD_START and s.index.max() <= FWD_END
        DF, DX = P2.make_dframe_for(s, P5.FREEZE_FIRST)
        z = cached(os.path.join(O.SCRATCH, "sig_C.npz"), lambda: P2.signals_arms(TF, X, DF, DX, allm, frozen_m0=O3.TRAIN_END))
        s15 = cached(os.path.join(O.SCRATCH, "sig15_C.npz"), lambda: {"s15": O.sigma15_walk(TF, DF, allm, frozen_m0=O3.TRAIN_END)})["s15"]
    else:
        if "cut" not in cut:
            s = L8.load_bars_cut(); DF = L8.load_dframe(); cut["cut"] = (s, DF, L9.load_dec_extras(DF, s))
        s, DF, DX = cut["cut"]
        z = cached(os.path.join(O9SCR, f"sig_{sp}.npz"), lambda: P2.signals_arms(TF, X, DF, DX, allm))
        s15 = cached(os.path.join(O.SCRATCH, f"sig15_{sp}.npz"), lambda: {"s15": O.sigma15_walk(TF, DF, allm)})["s15"]
    S = {"sig": z["L1"], "dv": z["dv"], "p": z["p"], "s15": s15}
    return s, DF, S, months, allm


def descr10(tf, months, s):
    d = O.summarise10(tf, months); d["months"] = list(months)
    for c in L8.COSTS_BPS:
        d[f"monthly_{int(c)}bps"] = P2.monthly_tf(tf, months, c).tolist()
    m2 = np.array(d["monthly_2bps"]); mu, lo, hi, _ = L8.T.ci(m2); d["ci_2bps"] = [mu, lo, hi]
    net = tf["ret_gross"].values - 2e-4 if len(tf) else np.array([])
    d["mean_trade_bps"] = float(net.mean() * 1e4) if len(net) else None
    lo_ = P5.month_bounds(months[0])[0]; hi_ = min(P5.month_bounds(months[-1])[1], s.index.max() + pd.Timedelta(minutes=1))
    days = pd.date_range(lo_.normalize(), (hi_ - pd.Timedelta(minutes=1)).normalize(), freq="D"); dr = P2.daily_net(tf, days, 2.0)
    d["sharpe_daily"] = float(dr.mean() / dr.std(ddof=1) * np.sqrt(365)) if dr.std(ddof=1) > 0 else None
    yrs = len(days) / 365.25; d["total_net"] = float(net.sum()); d["ann_net"] = float(net.sum() / yrs); d["calmar"] = float(d["ann_net"] / d["max_dd_2bps"]) if d["max_dd_2bps"] > 0 else None
    d["target_met"] = bool(d["hit_2bps"] is not None and d["hit_2bps"] > 0.60 and d["max_dd_2bps"] < 0.05)
    return d


def buy_hold(s, months):
    lo = P5.month_bounds(months[0])[0]; hi = P5.month_bounds(months[-1])[1]
    x = s[(s.index >= lo) & (s.index < hi)]
    cum = np.log(x["close"].resample("D").last().dropna().values) - np.log(x["open"].iloc[0]); z = np.concatenate([[0.0], cum])
    return {"log_return": float(np.log(x["close"].iloc[-1] / x["open"].iloc[0])), "max_dd_daily": float((np.maximum.accumulate(z) - z).max())}


def run_arms(ctx, rows, S, months, allm, full, shuf_of, models_t):
    """The arms of one span (O.arm_configs), both fills. models_t: {T: E4 models}. A SHUF arm permutes the exit signal of its base arm within the month
    (E4: draws from the base arm's probabilities of the same month). Returns {(arm, fills): (trade frame, raw trades)}."""
    aL, aS = L9.gate_masks(S["p"], S["dv"], O.ENTRY["delta"], O.ENTRY["g"]); out = {}
    for fn, pess in FILLS.items():
        plogs = {}
        for name, c in full.items():
            plogs[name] = []
            out[(name, fn)] = O.trades10(ctx, rows, O.spec_of(c, models=models_t.get(c["T"], {}) if c["exit"] == "E4" else None, plog=plogs[name]), S, S["sig"], aL, aS, pess)
        for name, base in shuf_of.items():
            c = full[base]; kind = O.spec_of(c)["kind"]; c2, sx = O.make_shuf(ctx, S["sig"], kind, allm); sh = None
            if kind == "E4":
                sh = {}
                for mo, P in plogs[base]:
                    sh.setdefault(mo, []).append(P)
                sh = {k: np.array(v) for k, v in sh.items()}
            out[(name, fn)] = O.trades10(c2, rows, O.spec_of(c, models=models_t.get(c["T"], {}) if c["exit"] == "E4" else None, shuf=sh), S, sx, aL, aS, pess)
    return out


def e4_models_for(sp, Tt, ctx, rows, S, months, recs_sel, recs_b):
    """E4's models for a span and T: A trains on its own past; B on the selection pool + its own past; C one fit at the cut on the selection pool + B."""
    aL, aS = L9.gate_masks(S["p"], S["dv"], O.ENTRY["delta"], O.ENTRY["g"])
    if sp == "C":
        m = O.e4_fit(O.cat_rec([recs_sel[Tt], recs_b[Tt]]), O3.TRAIN_END); assert m is not None; return {"*": m}
    own = O.e4_records(ctx, O.simulate10(ctx, rows, O.spec_of({"exit": "E0", "T": Tt}), S["sig"], S["sig"], aL, aS, pess=True))
    return O.e4_models(O.cat_rec([own] if sp == "A" else [recs_sel[Tt], own]), months)


def describe():
    ch = choice_gate(); cfg = ch["cfg"]; gv = O.gv_config(SEL); full, shuf_of, pairs = O.arm_configs(cfg, gv)
    nice(); t0 = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    out_p = os.path.join(HERE, "describe.jsonl")
    if os.path.exists(out_p):
        raise SystemExit("REFUSED: describe.jsonl exists; the descriptive run is made once.")
    log("arms", full, "SHUF of", shuf_of, "pairs", pairs)
    TF = L8.load_train_frame(); s_cut = L8.load_bars_cut(); X = L9.load_train_extras(TF, s_cut); cut = {}
    recs_sel = {Tt: dict(np.load(os.path.join(O.SCRATCH, f"recs_sel_T{Tt}.npz"))) for Tt in O.TS}; recs_b = {}
    e4_ts = sorted({c["T"] for c in full.values() if c["exit"] == "E4"})
    rec = []; res = {}; trades = {}; bh = {}; frames = {}; gc = O.gcols(); aL = aS = None
    for sp in ("B", "C", "A"):
        s, DF, S, months, allm = prepare_span(sp, TF, X, cut); ctx = O.make_ctx(s, DF, S, gc)
        rows = np.where(np.isin(ctx["mo_row"], months) & np.isfinite(S["sig"]))[0]
        if sp == "B":                                                         # B's own E0 checkpoints (both T) feed C's single fit
            aL, aS = L9.gate_masks(S["p"], S["dv"], O.ENTRY["delta"], O.ENTRY["g"])
            for Tt in O.TS:
                recs_b[Tt] = O.e4_records(ctx, O.simulate10(ctx, rows, O.spec_of({"exit": "E0", "T": Tt}), S["sig"], S["sig"], aL, aS, pess=True))
        models_t = {Tt: e4_models_for(sp, Tt, ctx, rows, S, months, recs_sel, recs_b) for Tt in e4_ts}
        log("span", sp, "valid decisions", len(rows), "E4 models", {Tt: len(m) for Tt, m in models_t.items()}, f"{time.time() - t0:.0f}s")
        arms = run_arms(ctx, rows, S, months, allm, full, shuf_of, models_t); bh[sp] = buy_hold(s, months); trades[sp] = arms; allf = []
        for (arm, fn), (tf, tr) in arms.items():
            d = descr10(tf, months, s); d.update({"span": sp, "arm": arm, "fills": fn}); res[(sp, arm, fn)] = d; rec.append(d); allf.append(tf.assign(arm=arm, fills=fn))
        pd.concat(allf).to_csv(os.path.join(HERE, f"trades_{sp}.csv.gz"), index=False)
        if sp in ("A", "B"):
            frames[sp] = (ctx, S, models_t)
    paired = {sp: {f"{a} - {b}": O.paired_monthly(res[(sp, a, "PESS")]["monthly_2bps"], res[(sp, b, "PESS")]["monthly_2bps"]) for a, b in pairs} for sp in ("B", "C", "A")}
    with open(out_p, "w", encoding="utf-8") as f:
        for d in rec:
            f.write(json.dumps(d) + "\n")
        f.write(json.dumps({"verdict": "descriptive", "buy_hold": bh, "choice": ch, "arms": full, "shuf_of": shuf_of, "paired_monthly_2bps_PESS": paired}) + "\n")
    write_tables(res, bh, ch, paired, full, shuf_of)
    make_figures(trades, full)
    make_windows(trades, frames, full)
    log("describe done", f"{time.time() - t0:.0f}s"); json.dump({"runtime_s": time.time() - t0}, open(os.path.join(HERE, "runtime_describe.json"), "w"))


ORDER = ("CHOICE", "E0", "E0_240", "SHUF", "GV", "GV_SHUF")
COL = {"CHOICE": "#d55181", "E0": "#9a9890", "E0_240": "#c98500", "SHUF": "#9085e9", "GV": "#2a9d8f", "GV_SHUF": "#6b8fb5"}


def write_tables(res, bh, ch, paired, full, shuf_of):
    o = []; Q = o.append; f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"
    Q("O10 DESCRIPTIVE results on spans already seen by O8/O9 (A, B, C). No verdict: these spans informed the design (SPEC \"Spans and selection\").")
    Q(f"entry (all arms): ROBUST {ch['entry']}. Arms: " + "; ".join(f"{k} = {O.cfg_name(c)}" for k, c in full.items()) + "; " + "; ".join(f"{k} = {b}'s exit signal permuted within the month" for k, b in shuf_of.items()))
    Q("CHOICE = the pre-registered choice (max t over the 14 configurations); E0 = ROBUST's own exit (target 2 sigma, T 60); GV = the best geometry/volatility exit of the selection span (max t among the non-E0 configurations; an added arm, named from selection numbers only).")
    Q("SHUF for E0-type exits (price levels only): the sigma_60 that sizes the stop and the target is permuted within the month; the entries keep the real sigma. For E1/E2a/E2b/E3 the exit signal series is permuted; for E4 its probability is drawn from the month's probabilities of the base arm.")
    Q("Returns in % of capital per month: sum of position log returns, no compounding. t = t-statistic of the monthly net at 2 bps; CI = 95% t-interval of the monthly mean.")
    for sp in ("B", "C", "A"):
        any_ = res[(sp, "CHOICE", "PESS")]
        Q(f"\n===== {SPAN_NAMES[sp]}; {len(any_['months'])} months; buy-and-hold log return {100 * bh[sp]['log_return']:+.1f}% (daily-close max DD {100 * bh[sp]['max_dd_daily']:.1f}%) =====")
        Q(f"{'arm':8s} {'fills':5s} {'trades':>6s} {'hit%':>5s} {'win':>6s} {'loss':>6s} {'hold':>6s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t2':>6s} {'CI2 lo':>7s} {'CI2 hi':>7s} {'mo+':>6s} {'Sharpe':>6s} {'maxDD%':>7s} {'Calmar':>7s} reasons | owner target")
        for key in [k for k in ORDER if (sp, k, "PESS") in res]:
            for fn in FILLS:
                d = res[(sp, key, fn)]; rs = " ".join(f"{k}:{v}" for k, v in sorted(d["reasons"].items()))
                Q(f"{key:8s} {fn:5s} {d['n_trades']:6d} {f(None if d['hit_2bps'] is None else 100 * d['hit_2bps'], 1):>5s} {f(d['avg_win_bps'], 1):>6s} {f(d['avg_loss_bps'], 1):>6s} {f(d['mean_hold_min'], 1):>6s} "
                  f"{100 * d['mean_m_0bps']:+7.3f} {100 * d['mean_m_2bps']:+7.3f} {100 * d['mean_m_5bps']:+7.3f} {d['t_2bps']:+6.2f} {100 * d['ci_2bps'][1]:+7.3f} {100 * d['ci_2bps'][2]:+7.3f} "
                  f"{d['months_pos_2bps']:2d}/{len(d['months']):<2d} {f(d['sharpe_daily']):>6s} {100 * d['max_dd_2bps']:7.2f} {f(d['calmar']):>7s} {rs} | {'MET' if d['target_met'] else 'not met'}")
        for name, pr in paired[sp].items():
            Q(f"paired monthly net at 2 bps, PESS: {name}: mean {100 * pr['mean']:+.3f}%  95% t-interval [{100 * pr['lo']:+.3f}, {100 * pr['hi']:+.3f}]  t {pr['t']:+.2f}  first better in {pr['months_better']}/{pr['n']} months")
        Q("monthly net at 2 bps, CHOICE PESS (%): " + " ".join(f"{m[2:]}:{100 * x:+.1f}" for m, x in zip(any_["months"], any_["monthly_2bps"])))
    Q("\nNote: E4's models: A trains on A's own earlier checkpoints, B on the selection pool + B's own, C one fit at 2025-07-21 on the selection pool + B. A month without a model leaves E4 holding (stop and time stop only).")
    open(os.path.join(OUT, "tables.txt"), "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


def curve(tf, cost=2.0):
    if len(tf) == 0:
        return np.array([]), np.array([]), np.array([])
    t = tf["exit_time"].values; c = np.cumsum(tf["ret_gross"].values - cost * 1e-4); dd = np.maximum.accumulate(np.maximum(c, 0)) - c
    return t, c, dd


def make_figures(trades, full):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "src"))
    from neural_trade.visualization import theme as TH
    present = [k for k in ORDER if (k, "PESS") in trades["B"]]
    keys = [(k, "PESS", "-") for k in present] + [(k, "orig", ":") for k in ("CHOICE", "E0", "GV") if k in present]
    nm = "  ".join(f"{k} = {O.cfg_name(c)}" for k, c in full.items())

    def style(a):
        a.set_facecolor(TH.SURFACE); a.grid(color=TH.GRID, lw=0.6); a.tick_params(colors=TH.MUTED); a.axhline(0, color=TH.NEUTRAL, lw=0.8)
        for sp_ in a.spines.values():
            sp_.set_color(TH.AXIS)
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), facecolor=TH.PAPER)
    for j, sp in enumerate(("B", "C", "A")):
        for key, fn, ls in keys:
            t, c, dd = curve(trades[sp][(key, fn)][0])
            if len(t):
                lw = 1.7 if key in ("CHOICE", "GV") else 1.0
                ax[0, j].plot(t, 100 * c, ls, color=COL[key], lw=lw, label=f"{key} {fn}"); ax[1, j].plot(t, -100 * dd, ls, color=COL[key], lw=lw)
        ax[0, j].set_title(SPAN_NAMES[sp], color=TH.INK, fontsize=10)
        ax[0, j].set_ylabel("cumulative net return, 2 bps (% of capital)", color=TH.MUTED); ax[1, j].set_ylabel("drawdown (% of capital)", color=TH.MUTED)
        style(ax[0, j]); style(ax[1, j])
    ax[0, 0].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK, fontsize=7)
    fig.suptitle(f"O10 descriptive: equity and drawdown (position returns in order, no compounding). Solid = PESS fills, dotted = original fills.  {nm}", color=TH.INK, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_dd.png"), dpi=110, facecolor=TH.PAPER); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), facecolor=TH.PAPER); dist = {}
    dkeys = [k for k in ("CHOICE", "E0", "GV", "SHUF", "GV_SHUF") if k in present]
    for j, sp in enumerate(("B", "C", "A")):
        for key in dkeys:
            tf = trades[sp][(key, "PESS")][0]
            if len(tf) == 0:
                continue
            hold = tf["hold"].values.astype(float); R = tf["ret_gross"].values / tf["sigma"].values; dist[(sp, key)] = (hold, R)
            ax[0, j].hist(hold, bins=np.linspace(0, max(240, hold.max()), 49), histtype="step", color=COL[key], lw=1.5, label=f"{key} (mean {hold.mean():.0f} min)", density=True)
            ax[1, j].hist(np.clip(R, -4, 6), bins=np.linspace(-4, 6, 51), histtype="step", color=COL[key], lw=1.5, label=f"{key} (mean {R.mean():+.2f} R)", density=True)
        ax[0, j].set_title(SPAN_NAMES[sp], color=TH.INK, fontsize=10); ax[0, j].set_xlabel("holding time (minutes, entry bar to exit bar)", color=TH.MUTED); ax[1, j].set_xlabel("exit R-multiple (gross log return / entry sigma_60, clipped to [-4, 6])", color=TH.MUTED)
        for r in (0, 1):
            style(ax[r, j]); ax[r, j].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK, fontsize=7); ax[r, j].set_ylabel("density", color=TH.MUTED)
    fig.suptitle(f"O10 descriptive: distribution of holding times (top) and exit R-multiples (bottom), PESS fills.  {nm}", color=TH.INK, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hold_R.png"), dpi=110, facecolor=TH.PAPER); plt.close(fig)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pf = make_subplots(rows=2, cols=3, subplot_titles=[SPAN_NAMES[x] for x in ("B", "C", "A")] + [""] * 3)
    for j, sp in enumerate(("B", "C", "A"), 1):
        for key, fn, ls in keys:
            t, c, dd = curve(trades[sp][(key, fn)][0])
            if len(t):
                kw = dict(x=t, mode="lines", line=dict(color=COL[key], dash="solid" if ls == "-" else "dot"), name=f"{key} {fn}", legendgroup=f"{key}{fn}", showlegend=(j == 1))
                pf.add_trace(go.Scatter(y=100 * c, **kw), row=1, col=j); kw["showlegend"] = False; pf.add_trace(go.Scatter(y=-100 * dd, **kw), row=2, col=j)
    pf.update_layout(template="plotly_dark", paper_bgcolor=TH.PAPER, plot_bgcolor=TH.SURFACE, height=700, title=f"O10 descriptive equity (top, % of capital, 2 bps) and drawdown (bottom). {nm}")
    open(os.path.join(OUT, "equity_dd.plotly.json"), "w", encoding="utf-8").write(pf.to_json())
    ph = make_subplots(rows=2, cols=3, subplot_titles=[SPAN_NAMES[x] for x in ("B", "C", "A")] + [""] * 3)
    for j, sp in enumerate(("B", "C", "A"), 1):
        for key in dkeys:
            if (sp, key) in dist:
                hold, R = dist[(sp, key)]
                ph.add_trace(go.Histogram(x=hold, name=key, marker_color=COL[key], opacity=0.55, histnorm="probability density", legendgroup=key, showlegend=(j == 1)), row=1, col=j)
                ph.add_trace(go.Histogram(x=np.clip(R, -4, 6), name=key, marker_color=COL[key], opacity=0.55, histnorm="probability density", legendgroup=key, showlegend=False), row=2, col=j)
    ph.update_layout(template="plotly_dark", paper_bgcolor=TH.PAPER, plot_bgcolor=TH.SURFACE, height=700, barmode="overlay", title=f"O10 holding time (min, top) and exit R-multiple (bottom). {nm}")
    open(os.path.join(OUT, "hold_R.plotly.json"), "w", encoding="utf-8").write(ph.to_json())


def make_windows(trades, frames, full):
    """Real example trades (PESS; spans A and B, cut-file bars only, <= 2025-07-20) with the exit drawn: the CHOICE (one winner and one loser per span) and the GV arm
    (two winners and one loser in B, one winner and two losers in A). The signal: the 1-minute EMA20 (all), the stop and target (E0), the trailing level (E1) or the
    decision-time values of slope3 / dV / sigma_15 / P (E2b, E3, E4); trades that exited on the signal are preferred."""
    out = {"note": "cut-file bars only (<= 2025-07-20)", "arms": {k: O.cfg_name(c) for k, c in full.items()}, "examples": []}; rng = np.random.default_rng(7)
    for name, counts in (("CHOICE", {("B", True): 1, ("B", False): 1, ("A", True): 1, ("A", False): 1}), ("GV", {("B", True): 2, ("B", False): 1, ("A", True): 1, ("A", False): 2})):
        if name not in full:
            continue
        cfg = full[name]; sp_ = O.spec_of(cfg)
        for (sp, win), nsel in counts.items():
            tf, tr = trades[sp][(name, "PESS")]; ctx, S, models_t = frames[sp]
            net = tf["ret_gross"].values - 2e-4; want = (net > 0) if win else (net <= 0); idx = np.where(want)[0]
            sig = np.array([r in SIGNAL_REASONS for r in tf["reason"].values]); pref = idx[sig[idx]] if sig[idx].any() else idx
            for i in rng.choice(pref, size=min(nsel, len(pref)), replace=False):
                e = int(tr["ebar"][i]); x = int(tr["xbar"][i]); d = int(tr["dec"][i]); side = int(tr["side"][i]); p0 = float(tr["epx"][i]); s0 = float(tr["sig"][i])
                lo = max(0, d - 8); hi = min(ctx["n"] - 1, x + 12)
                ex = {"arm": name, "config": O.cfg_name(cfg), "span": sp, "decision_time": str(ctx["idx"][d]), "entry_time": str(ctx["idx"][e]), "exit_time": str(ctx["idx"][x]), "side": side,
                      "entry_px": p0, "exit_px": float(tr["xpx"][i]), "reason": str(tr["reason"][i]), "gross_ret": float(tr["ret"][i]), "sigma_60": s0, "stop_px": p0 * (1 - side * s0),
                      "target_px": p0 * (1 + side * O.TAU * s0) if sp_["kind"] == "E0" else None, "p_up": float(S["p"][tr["row"][i]]),
                      "bars": [{"t": str(ctx["idx"][j]), "o": float(ctx["O"][j]), "h": float(ctx["Hi"][j]), "l": float(ctx["Lo"][j]), "c": float(ctx["Cl"][j]), "ema20": float(ctx["ema"][j])} for j in range(lo, hi + 1)]}
                if sp_["kind"] == "E1":
                    m = max(min(x, e + cfg["T"] - 1) - e, 1); ext = (ctx["Hi"] if side == 1 else ctx["Lo"])[e + 1:e + m]
                    lv = O.trail_levels(side, p0, ex["stop_px"], ext, O.sigma15_path(ctx, e, m), sp_["c"]); ex["trail"] = [{"t": str(ctx["idx"][e + 1 + q]), "level": float(v)} for q, v in enumerate(lv)]
                if sp_["kind"] in ("E2b", "E3", "E4"):
                    dec = []
                    for a in range(e + 1, x + 1):
                        r = int(ctx["barrow"][a])
                        if r >= 0:
                            P = O.e4_prob(ctx, O.spec_of(cfg, models=models_t.get(cfg["T"], {})), r, a, e, side, p0, s0) if sp_["kind"] == "E4" else None
                            dec.append({"t": str(ctx["idx"][a]), "slope3_15m_ema20": float(ctx["slope"][r]), "dV": float(ctx["dvx"][r]), "sigma_15": float(ctx["s15"][r]), "P_hold": None if P is None else float(P)})
                    ex["decisions"] = dec
                out["examples"].append(ex)
    json.dump(out, open(os.path.join(OUT, "windows.json"), "w"), indent=1)


# ------------------------------------------------------------------ freeze
def freeze():
    ch = choice_gate(); nice()
    out_p = os.path.join(HERE, "frozen_P.json")
    if os.path.exists(out_p):
        raise SystemExit("REFUSED: frozen_P.json exists; the freeze is made once.")
    art9 = json.load(open(os.path.join(O.O9DIR, "frozen_P.json")))
    bars = P5.load_freeze_bars(); assert bars.index.max() <= P5.B_HI and bars.index.min() >= P5.B_LO
    log("freeze bars", len(bars), bars.index.min(), bars.index.max())
    art = O.freeze_fit10(bars, P5.B_M0, ch["config"], art9=art9, log=log); json.dump(art, open(out_p, "w"), indent=1)
    open(os.path.join(HERE, "frozen_P.sha256"), "w").write(art["sha256"] + "\n")
    log("FROZEN sha256", art["sha256"], "O9 models sha256", art["o9_frozen_sha256"], "sigma15 n_tr", art["sigma15"]["n_tr"], "E4", {t: (j or {}).get("n") for t, j in art["e4"].items()})


if __name__ == "__main__":
    {"select": select, "describe": describe, "freeze": freeze}[sys.argv[1]]()
