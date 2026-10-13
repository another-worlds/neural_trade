"""O8 runner (SPEC.md). usage: run_o8.py [frame|select|test].
  frame  : bars cache and the decision-frame cache (scratch).
  select : PHASE 1. All 24 configurations, L1 arm, on the selection months 2021-01..2024-06 -> selection.jsonl, tables_selection.txt, choice.json.
  test   : PHASE 2: the chosen configuration on spans B, C, A, all arms and both fill variants. Refuses unless choice.json is tracked at HEAD and unchanged;
           refuses to run twice (tests.jsonl exists).
"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o8_lib as L  # noqa: E402
LOG = os.path.join(HERE, "run.log"); SEL = os.path.join(HERE, "selection.jsonl"); CHOICE = os.path.join(HERE, "choice.json")
TABLE = os.path.join(HERE, "tables_selection.txt")


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


def frame():
    nice(); assert free_gb() >= 8, "less than 8 GB free RAM"
    t0 = time.time(); s = L.load_bars_cut()
    gaps = s.index.to_series().diff().dropna()
    log("bars", len(s), s.index.min(), s.index.max(), "non-1-minute steps", int((gaps != pd.Timedelta(minutes=1)).sum()), f"{time.time() - t0:.0f}s")
    D = L.load_dframe(); log("decision frame", len(D["ta"]), "valid", int(D["ok"].sum()), pd.Timestamp(D["ta"][0]), pd.Timestamp(D["ta"][-1]), f"{time.time() - t0:.0f}s")
    return s, D


def arrays(s):
    return (s["open"].values, s["high"].values, s["low"].values, s["close"].values)


def run_config(cfg, sig, s_idx, O, Hi, Lo, Cl, DF, rows):
    """One configuration on the decision rows `rows` (a sorted index array into the decision frame) with sigma `sig` (aligned to the frame)."""
    tr = L.simulate(O, Hi, Lo, Cl, DF["a"][rows], DF["m"][rows], sig[rows], cfg["family"], cfg["k"], cfg["H"], cfg["stop"], tmin=s_idx.values.astype("datetime64[m]").astype(np.int64))
    em = s_idx[tr["ebar"]].to_period("M").astype(str).values if len(tr["dec"]) else np.array([], dtype=str)
    return tr, em


def select():
    nice(); t_all = time.time(); s, DF = frame(); O, Hi, Lo, Cl = arrays(s); TF = L.load_train_frame()
    months = L.SPANS["sel"]; lo = L.P5.month_bounds(months[0])[0]; hi = L.P5.month_bounds(months[-1])[1]
    rows = np.where((DF["ta"] >= np.datetime64(lo, "ns")) & (DF["ta"] < np.datetime64(hi, "ns")))[0]
    assert pd.Timestamp(DF["ta"][rows].max()) < pd.Timestamp("2024-07-01") and pd.Timestamp(DF["ta"][rows].min()) >= pd.Timestamp("2021-01-01")
    sig = {}
    for H in L.HZ:
        arms, _ = L.sigma_arms(TF, DF, months, H); sig[H] = arms["L1"]
        log("sigma L1 H", H, "valid decisions", int(np.isfinite(arms["L1"]).sum()), "median sigma", float(np.nanmedian(arms["L1"])), f"{time.time() - t_all:.0f}s")
    if os.path.exists(SEL):
        os.remove(SEL)
    res = []
    for cfg in L.config_grid():
        tr, em = run_config(cfg, sig[cfg["H"]], s.index, O, Hi, Lo, Cl, DF, rows)
        r = {"config": L.cfg_name(cfg), **cfg, "arm": "L1", **L.summarise(tr, em, months)}; res.append(r)
        with open(SEL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        log(r["config"], "trades", r["n_trades"], "mean% 2bps", f"{100 * r['mean_m_2bps']:+.3f}", "t", f"{r['t_2bps']:+.2f}", f"{time.time() - t_all:.0f}s")
    best = max(res, key=lambda r: r["t_2bps"])
    ch = {"rule": "highest t-statistic of the monthly net return at 2 bps on 2021-01..2024-06, L1 arm (SPEC.md)", "config": best["config"],
          **{k: best[k] for k in ("family", "k", "H", "stop")}, "t_2bps": best["t_2bps"], "mean_m_2bps": best["mean_m_2bps"], "n_trades": best["n_trades"],
          "n_months": best["n_months"], "selection_months": [months[0], months[-1]]}
    tables(res, ch)
    json.dump(ch, open(CHOICE, "w"), indent=1); log("CHOICE", ch); log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime_select.json"), "w"))


def tables(res, ch):
    out = ["O8 selection: 24 configurations, L1 arm, 42 months 2021-01..2024-06. Returns: mean monthly net return in % of capital (sum of position log returns, 1x, no compounding)."]
    out.append(f"{'config':22s} {'trades':>6s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t(2bp)':>7s} {'mo+':>5s} {'hit%':>6s} {'maxDD%':>7s} {'long%':>6s} {'tgt/stop/time':>16s} {'both':>5s}")
    for r in sorted(res, key=lambda r: -r["t_2bps"]):
        rs = r["reasons"]; ex = f"{rs.get('target', 0)}/{rs.get('stop', 0)}/{rs.get('time', 0)}"
        hit = "n/a" if r["hit_2bps"] is None else f"{100 * r['hit_2bps']:.1f}"
        lg = "n/a" if r["long_share"] is None else f"{100 * r['long_share']:.0f}"
        out.append(f"{r['config']:22s} {r['n_trades']:6d} {100 * r['mean_m_0bps']:+7.3f} {100 * r['mean_m_2bps']:+7.3f} {100 * r['mean_m_5bps']:+7.3f} {r['t_2bps']:+7.2f} "
                   f"{r['months_pos_2bps']:2d}/{r['n_months']:<2d} {hit:>6s} {100 * r['max_dd_2bps']:7.2f} {lg:>6s} {ex:>16s} {r['n_both_same_bar']:5d}")
    out.append(f"\nCHOICE (max t at 2 bps): {ch['config']}  t={ch['t_2bps']:+.2f}  mean monthly net {100 * ch['mean_m_2bps']:+.3f}%  trades {ch['n_trades']}")
    open(TABLE, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))


# ------------------------------------------------------------------ the gate of PHASE 2
def git(*args):
    return subprocess.run(["git", "-C", HERE, *args], capture_output=True, text=True)


def choice_gate():
    """Refuse the test unless choice.json is tracked at HEAD, unchanged in the working tree, and the commit that added it is an ancestor of HEAD."""
    if git("ls-files", "--error-unmatch", "--", CHOICE).returncode != 0:
        raise SystemExit("REFUSED: choice.json is not tracked in git. Commit it before any test is computed.")
    if git("diff", "--quiet", "HEAD", "--", CHOICE).returncode != 0:
        raise SystemExit("REFUSED: choice.json differs from HEAD (changed after its commit).")
    if git("cat-file", "-e", "HEAD:./choice.json").returncode != 0:
        raise SystemExit("REFUSED: choice.json is not in HEAD.")
    return json.load(open(CHOICE, encoding="utf-8"))


VARIANTS = {"primary": False, "PESS": True}
TESTS = os.path.join(HERE, "tests.jsonl")
FWD_END = pd.Timestamp("2026-10-12 23:59")
SPAN_NAMES = {"A": "A 2018-07..2020-12 (walk-forward)", "B": "B 2024-07..2025-07-20 (walk-forward)", "C": "C 2025-10-05..2026-10-12 (frozen fit)"}
ARM_COLORS = {"L1": "#d55181", "R0": "#c98500", "TRAIL": "#9a9890", "SHUF": "#9085e9"}
WARM_C = L.P5.FREEZE_FIRST


def span_months(sp):
    return L.SPANS[sp] if sp in L.SPANS else [str(m) for m in pd.period_range("2025-10", "2026-10", freq="M")]


def prepare_span(sp, ctx):
    """bars s, decision frame DF, sigma {arm: array} for the chosen H. A, B: monthly walk-forward on the cut file; C: frozen fit at TRAIN_END
    applied to features computed from forward-file bars only."""
    months = span_months(sp); H = ctx["cfg"]["H"]
    if sp == "C":
        if "s_fwd" not in ctx:
            ctx["s_fwd"] = L.T.load_bars(L.T.FWD_CSV, L.T.FWD_START, FWD_END)          # refuses a bar of 2025-07-21..2025-09-30
            assert ctx["s_fwd"].index.min() >= L.T.FWD_START
            D_ = L.build_dframe(ctx["s_fwd"], WARM_C)
            ctx["DF_fwd"] = {"XB": D_["XB"], "G": D_["G"], "ok": D_["ok"], "a": D_["a"], "ta": D_["ta"], "m": D_["m"], **{f"trail{H_}": D_["trail"][H_] for H_ in L.HZ}}
        s, DF = ctx["s_fwd"], ctx["DF_fwd"]
        arms, st = L.sigma_arms(ctx["TF"], DF, months, H, frozen_m0=L.O3.TRAIN_END)
        assert pd.Timestamp(st["frozen"]["last_label_end"]) <= L.O3.TRAIN_END
    else:
        if "s_cut" not in ctx:
            ctx["s_cut"] = L.load_bars_cut(); ctx["DF_cut"] = L.load_dframe()
        s, DF = ctx["s_cut"], ctx["DF_cut"]
        arms, st = L.sigma_arms(ctx["TF"], DF, months, H)
    return s, DF, arms


def run_span(sp, ctx):
    s, DF, arms = prepare_span(sp, ctx); cfg = ctx["cfg"]; months = span_months(sp)
    O, Hi, Lo, Cl = arrays(s); tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    out = {}
    for arm in L.ARMS:
        rows = np.where(np.isfinite(arms[arm]))[0]
        for vn, pess in VARIANTS.items():
            out[(arm, vn)] = L.simulate(O, Hi, Lo, Cl, DF["a"][rows], DF["m"][rows], arms[arm][rows], cfg["family"], cfg["k"], cfg["H"], cfg["stop"], tmin=tmin, pess=pess)
    return s, DF, arms, out, months


def trade_frame(tr, s):
    if len(tr["dec"]) == 0:
        return pd.DataFrame({c: [] for c in ("entry_time", "exit_time", "side", "entry_px", "exit_px", "reason", "ret_gross", "sigma", "both_same_bar")})
    return pd.DataFrame({"entry_time": s.index[tr["ebar"]], "exit_time": s.index[tr["xbar"]], "side": tr["side"], "entry_px": tr["epx"], "exit_px": tr["xpx"],
                         "reason": tr["reason"], "ret_gross": tr["ret"], "sigma": tr["sig"], "both_same_bar": tr["both"]})


def descr(tf, months, s, cost=2.0):
    """The descriptive block of one (span, arm, variant): trades, hit, trade bps, monthly returns, daily Sharpe, drawdown, Calmar, split, reasons."""
    d = {"n_trades": int(len(tf))}
    em = tf["entry_time"].dt.to_period("M").astype(str).values if len(tf) else np.array([], dtype=str)
    tr = {"ret": tf["ret_gross"].values if len(tf) else np.array([])}
    for c in L.COSTS_BPS:
        mr = L.monthly_returns(tr, em, months, c); d[f"monthly_{int(c)}bps"] = mr.tolist(); d[f"mean_m_{int(c)}bps"] = float(mr.mean())
    d["months"] = list(months); d["t_2bps"] = L.t_stat(np.array(d["monthly_2bps"]))
    net = tr["ret"] - cost * 1e-4
    d["hit"] = float((net > 0).mean()) if len(net) else None
    d["mean_trade_bps"] = float(net.mean() * 1e4) if len(net) else None; d["median_trade_bps"] = float(np.median(net) * 1e4) if len(net) else None
    d["max_dd"] = L.max_drawdown(net)
    lo = L.P5.month_bounds(months[0])[0]; hi = min(L.P5.month_bounds(months[-1])[1], s.index.max() + pd.Timedelta(minutes=1))
    days = pd.date_range(lo.normalize(), (hi - pd.Timedelta(minutes=1)).normalize(), freq="D")
    if len(tf):
        dr = pd.Series(net, index=pd.DatetimeIndex(tf["exit_time"]).normalize()).groupby(level=0).sum().reindex(days, fill_value=0.0)
    else:
        dr = pd.Series(0.0, index=days)
    d["sharpe_daily"] = float(dr.mean() / dr.std(ddof=1) * np.sqrt(365)) if dr.std(ddof=1) > 0 else None
    yrs = len(days) / 365.25; d["total_net"] = float(net.sum()); d["ann_net"] = float(net.sum() / yrs)
    d["calmar"] = float(d["ann_net"] / d["max_dd"]) if d["max_dd"] > 0 else None
    d["long_share"] = float((tf["side"] == 1).mean()) if len(tf) else None
    d["reasons"] = tf["reason"].value_counts().to_dict() if len(tf) else {}
    d["both_same_bar"] = int(tf["both_same_bar"].sum()) if len(tf) else 0
    d["target_met"] = bool(d["hit"] is not None and d["hit"] > 0.60 and d["max_dd"] < 0.05)
    return d


def buy_hold(s, months):
    lo = L.P5.month_bounds(months[0])[0]; hi = L.P5.month_bounds(months[-1])[1]
    x = s[(s.index >= lo) & (s.index < hi)]
    cum = np.log(x["close"].resample("D").last().dropna().values) - np.log(x["open"].iloc[0]); z = np.concatenate([[0.0], cum])
    return {"log_return": float(np.log(x["close"].iloc[-1] / x["open"].iloc[0])), "max_dd_daily": float((np.maximum.accumulate(z) - z).max()),
            "from": str(x.index[0]), "to": str(x.index[-1])}


def ci_block(x):
    x = np.asarray(x, float); mu, lo, hi, t = L.T.ci(x)
    return {"mean": mu, "lo": lo, "hi": hi, "t": t, "n": int(len(x)), "pos": int((x > 0).sum())}


def test():
    ch = choice_gate()
    if os.path.exists(TESTS):
        raise SystemExit("REFUSED: tests.jsonl exists; the tests are judged once.")
    nice(); t_all = time.time(); assert free_gb() >= 8, "less than 8 GB free RAM"
    cfg = {k: ch[k] for k in ("family", "k", "H", "stop")}; assert L.cfg_name(cfg) == ch["config"]
    ctx = {"cfg": cfg, "TF": L.load_train_frame()}; rec = []; res = {}; trades = {}; bh = {}
    for sp in ("B", "C", "A"):
        s, DF, arms, out, months = run_span(sp, ctx); log("span", sp, "decisions", int(np.isfinite(arms["L1"]).sum()), f"{time.time() - t_all:.0f}s")
        bh[sp] = buy_hold(s, months); frames = []; trades[sp] = {}
        for (arm, vn), tr in out.items():
            tf = trade_frame(tr, s); d = descr(tf, months, s); d.update({"span": sp, "arm": arm, "variant": vn, "config": ch["config"]})
            res[(sp, arm, vn)] = d; rec.append(d); trades[sp][(arm, vn)] = tf; frames.append(tf.assign(arm=arm, variant=vn))
        pd.concat(frames).to_csv(os.path.join(HERE, f"trades_{sp}.csv.gz"), index=False)
        if sp == "B":
            ctx["ex_B"] = (out[("L1", "primary")],)
    mo = lambda sp, arm, vn: np.array(res[(sp, arm, vn)]["monthly_2bps"])
    pool = lambda arm, vn: np.concatenate([mo("B", arm, vn), mo("C", arm, vn)])
    crit = {"c1_primary": ci_block(pool("L1", "primary")), "c1_PESS": ci_block(pool("L1", "PESS")),
            "c2_L1_minus_SHUF": ci_block(pool("L1", "primary") - pool("SHUF", "primary")), "c3_L1_minus_R0": ci_block(pool("L1", "primary") - pool("R0", "primary")),
            "c2_L1_minus_SHUF_PESS": ci_block(pool("L1", "PESS") - pool("SHUF", "PESS")), "c3_L1_minus_R0_PESS": ci_block(pool("L1", "PESS") - pool("R0", "PESS")),
            "L1_minus_TRAIL_primary(descriptive)": ci_block(pool("L1", "primary") - pool("TRAIL", "primary"))}
    for sp in ("B", "C", "A"):
        crit[f"{sp}_c1_primary"] = ci_block(mo(sp, "L1", "primary")); crit[f"{sp}_c1_PESS"] = ci_block(mo(sp, "L1", "PESS"))
        crit[f"{sp}_L1_minus_SHUF"] = ci_block(mo(sp, "L1", "primary") - mo(sp, "SHUF", "primary"))
        crit[f"{sp}_L1_minus_R0"] = ci_block(mo(sp, "L1", "primary") - mo(sp, "R0", "primary"))
    pos = lambda k: crit[k]["mean"] > 0 and crit[k]["lo"] > 0
    works = pos("c1_primary") and pos("c1_PESS") and pos("c2_L1_minus_SHUF"); adds = pos("c3_L1_minus_R0")
    verdict = {"works": bool(works), "adds": bool(adds), "c1_primary": bool(pos("c1_primary")), "c1_PESS": bool(pos("c1_PESS")),
               "c2": bool(pos("c2_L1_minus_SHUF")), "c3": bool(pos("c3_L1_minus_R0"))}
    with open(TESTS, "w", encoding="utf-8") as f:
        for d in rec:
            f.write(json.dumps(d) + "\n")
        f.write(json.dumps({"criteria": crit, "verdict": verdict, "buy_hold": bh, "config": ch["config"]}) + "\n")
    write_tables(res, crit, verdict, bh, ch)
    make_figures(trades)
    make_windows(ctx, trades, cfg)
    log("VERDICT", verdict); log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime_test.json"), "w"))


def fmt_ci(c):
    return f"{100 * c['mean']:+.3f}% [{100 * c['lo']:+.3f}, {100 * c['hi']:+.3f}]  t={c['t']:+.2f}  n={c['n']}  +{c['pos']}"


def write_tables(res, crit, verdict, bh, ch):
    o = []; Q = o.append
    Q(f"strategy works on independent data: {'yes' if verdict['works'] else 'no'}")
    Q(f"geometry adds to the strategy: {'yes' if verdict['adds'] else 'no'}")
    Q(f"\nO8 tests, configuration {ch['config']} chosen on 2021-01..2024-06 (choice.json 3253588). Returns in % of capital per month (sum of position log returns, 1x).")
    Q("\n===== PRE-REGISTERED CRITERIA, pooled B + C (months as units, 2 bps, 95% t-interval: mean [lo, hi]) =====")
    for k_, lab in (("c1_primary", "1 L1 mean monthly net, primary fills"), ("c1_PESS", "1 L1 mean monthly net, PESS fills"),
                    ("c2_L1_minus_SHUF", "2 L1 - SHUF (paired by month), primary"), ("c3_L1_minus_R0", "3 L1 - R0 (paired by month), primary"),
                    ("c2_L1_minus_SHUF_PESS", "  (extra) L1 - SHUF, PESS"), ("c3_L1_minus_R0_PESS", "  (extra) L1 - R0, PESS"),
                    ("L1_minus_TRAIL_primary(descriptive)", "  (descr.) L1 - TRAIL, primary (TRAIL not scale-matched)")):
        Q(f"  {lab:58s} {fmt_ci(crit[k_])}  -> {'PASS' if crit[k_]['mean'] > 0 and crit[k_]['lo'] > 0 else 'fail'}")
    Q(f"  criterion flags: {verdict}")
    Q("\n===== the same, per span =====")
    for sp in ("B", "C", "A"):
        for k_ in ("c1_primary", "c1_PESS", "L1_minus_SHUF", "L1_minus_R0"):
            Q(f"  {sp} {k_:16s} {fmt_ci(crit[f'{sp}_{k_}'])}")
    f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"
    for sp in ("B", "C", "A"):
        Q(f"\n===== {SPAN_NAMES[sp]}; buy-and-hold log return {100 * bh[sp]['log_return']:+.1f}% (daily-close max DD {100 * bh[sp]['max_dd_daily']:.1f}%), "
          f"{len(res[(sp, 'L1', 'primary')]['months'])} months =====")
        Q(f"{'arm':6s} {'fills':8s} {'trades':>6s} {'hit%':>5s} {'mean bp':>8s} {'med bp':>7s} {'m0bp%':>7s} {'m2bp%':>7s} {'m5bp%':>7s} {'t2':>6s} {'Sharpe':>6s} "
          f"{'maxDD%':>7s} {'Calmar':>7s} {'long%':>5s} {'tgt/stop/time':>14s} owner target")
        for arm in L.ARMS:
            for vn in VARIANTS:
                d = res[(sp, arm, vn)]; r = d["reasons"]
                Q(f"{arm:6s} {vn:8s} {d['n_trades']:6d} {f(None if d['hit'] is None else 100 * d['hit'], 1):>5s} {f(d['mean_trade_bps']):>8s} {f(d['median_trade_bps']):>7s} "
                  f"{100 * d['mean_m_0bps']:+7.3f} {100 * d['mean_m_2bps']:+7.3f} {100 * d['mean_m_5bps']:+7.3f} {d['t_2bps']:+6.2f} {f(d['sharpe_daily']):>6s} "
                  f"{100 * d['max_dd']:7.2f} {f(d['calmar']):>7s} {f(None if d['long_share'] is None else 100 * d['long_share'], 0):>5s} "
                  f"{r.get('target', 0)}/{r.get('stop', 0)}/{r.get('time', 0):<6} {'MET' if d['target_met'] else 'not met'}")
        Q("monthly net returns at 2 bps, L1 primary (%): " + " ".join(f"{m}:{100 * x:+.1f}" for m, x in zip(res[(sp, 'L1', 'primary')]["months"], res[(sp, 'L1', 'primary')]["monthly_2bps"])))
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o))


def curve(tf, cost=2.0):
    if len(tf) == 0:
        return np.array([]), np.array([]), np.array([])
    t = tf["exit_time"].values; c = np.cumsum(tf["ret_gross"].values - cost * 1e-4); dd = np.maximum.accumulate(np.maximum(c, 0)) - c
    return t, c, dd


def make_figures(trades):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "src"))
    from neural_trade.visualization import theme as TH
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), facecolor=TH.PAPER)
    for j, sp in enumerate(("B", "C", "A")):
        for arm in L.ARMS:
            for vn, ls in (("primary", "-"), ("PESS", "--")):
                if arm != "L1" and vn == "PESS":
                    continue
                t, c, dd = curve(trades[sp][(arm, vn)])
                if len(t):
                    lw = 1.6 if arm == "L1" else 1.0
                    ax[0, j].plot(t, 100 * c, ls, color=ARM_COLORS[arm], lw=lw, label=f"{arm} {vn}"); ax[1, j].plot(t, -100 * dd, ls, color=ARM_COLORS[arm], lw=lw)
        ax[0, j].set_title(SPAN_NAMES[sp], color=TH.INK, fontsize=10)
        ax[0, j].set_ylabel("cumulative net return, 2 bps (% of capital)", color=TH.MUTED); ax[1, j].set_ylabel("drawdown (% of capital)", color=TH.MUTED)
        for r in (0, 1):
            a = ax[r, j]; a.set_facecolor(TH.SURFACE); a.grid(color=TH.GRID, lw=0.6); a.tick_params(colors=TH.MUTED); a.axhline(0, color=TH.NEUTRAL, lw=0.8)
            for sp_ in a.spines.values():
                sp_.set_color(TH.AXIS)
    ax[0, 0].legend(facecolor=TH.SURFACE, edgecolor=TH.AXIS, labelcolor=TH.INK, fontsize=8)
    fig.suptitle("O8 BRK_k1.5_H60_s1.0: equity and drawdown by arm (position returns in order, constant 1x capital, no compounding)", color=TH.INK)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "equity_dd.png"), dpi=110, facecolor=TH.PAPER); plt.close(fig)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pf = make_subplots(rows=2, cols=3, subplot_titles=[SPAN_NAMES[x] for x in ("B", "C", "A")] + [""] * 3)
    for j, sp in enumerate(("B", "C", "A"), 1):
        for arm in L.ARMS:
            for vn, dash in (("primary", "solid"), ("PESS", "dash")):
                if arm != "L1" and vn == "PESS":
                    continue
                t, c, dd = curve(trades[sp][(arm, vn)])
                if len(t):
                    kw = dict(x=t, mode="lines", line=dict(color=ARM_COLORS[arm], dash=dash), name=f"{arm} {vn}", legendgroup=f"{arm}{vn}", showlegend=(j == 1))
                    pf.add_trace(go.Scatter(y=100 * c, **kw), row=1, col=j); kw["showlegend"] = False
                    pf.add_trace(go.Scatter(y=-100 * dd, **kw), row=2, col=j)
    pf.update_layout(template="plotly_dark", paper_bgcolor=TH.PAPER, plot_bgcolor=TH.SURFACE, height=700, title="O8 equity (top, % of capital, 2 bps) and drawdown (bottom)")
    open(os.path.join(HERE, "equity_dd.plotly.json"), "w", encoding="utf-8").write(pf.to_json())


def make_windows(ctx, trades, cfg):
    """~6 real example trades for the report: orders, fill, exit and the OHLC around them. Drawn bars are cut-file bars (<= 2025-07-20) only;
    forward-file trades are listed as numbers without bars."""
    s = ctx["s_cut"]; DF = ctx["DF_cut"]; O, Hi, Lo, Cl = arrays(s)
    out = {"config": L.cfg_name(cfg), "note": "OHLC rows are cut-file bars only; forward-file trades are numbers without bars", "examples": []}
    arms, _ = L.sigma_arms(ctx["TF"], DF, L.SPANS["sel"], cfg["H"]); rows = np.where(np.isfinite(arms["L1"]))[0]
    tm = s.index.values.astype("datetime64[m]").astype(np.int64)
    sel = L.simulate(O, Hi, Lo, Cl, DF["a"][rows], DF["m"][rows], arms["L1"][rows], cfg["family"], cfg["k"], cfg["H"], cfg["stop"], tmin=tm)
    Btr = ctx["ex_B"][0]

    def pick(tr, seed):
        rng = np.random.default_rng(seed); chosen = []; net = tr["ret"] - 2e-4
        for side, why, win in ((1, "time", True), (-1, "stop", False), (-1, "time", True), (1, "stop", False)):
            idx = np.where((tr["side"] == side) & (tr["reason"] == why) & ((net > 0) == win))[0]
            if len(idx):
                chosen.append(int(rng.choice(idx)))
        return chosen

    for tr, span, seed in ((sel, "selection 2021-01..2024-06", 1), (Btr, "test B 2024-07..2025-07-20", 2)):
        for i in pick(tr, seed)[:3]:
            d, e, x = int(tr["dec"][i]), int(tr["ebar"][i]), int(tr["xbar"][i]); sg = float(tr["sig"][i]); m = float(O[d]); side = int(tr["side"][i]); ep = float(tr["epx"][i])
            lo_, hi_ = max(0, d - 8), min(len(O) - 1, x + 12)
            bars_ = [{"t": str(s.index[j]), "o": float(O[j]), "h": float(Hi[j]), "l": float(Lo[j]), "c": float(Cl[j])} for j in range(lo_, hi_ + 1)]
            out["examples"].append({"span": span, "decision_time": str(s.index[d]), "m": m, "sigma": sg, "buy_stop_U": m * (1 + cfg["k"] * sg), "sell_stop_L": m * (1 - cfg["k"] * sg),
                                    "side": side, "entry_time": str(s.index[e]), "entry_px": ep, "stop_px": ep * (1 - side * cfg["stop"] * sg) if cfg["stop"] else None,
                                    "exit_time": str(s.index[x]), "exit_px": float(tr["xpx"][i]), "reason": str(tr["reason"][i]), "gross_ret": float(tr["ret"][i]),
                                    "time_stop_time": str(s.index[min(e + cfg["H"], len(s) - 1)]), "bars": bars_})
    tf = trades["C"][("L1", "primary")]
    fw = []
    for r in (tf.iloc[[len(tf) // 3, 2 * len(tf) // 3]].to_dict("records") if len(tf) > 2 else []):
        fw.append({k_: (str(v) if "time" in k_ else (v.item() if hasattr(v, "item") else v)) for k_, v in r.items()})
    out["forward_numbers_only"] = fw
    json.dump(out, open(os.path.join(HERE, "windows.json"), "w"), indent=1)


if __name__ == "__main__":
    {"frame": frame, "select": select, "test": test}[sys.argv[1]]()
