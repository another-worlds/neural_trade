"""O8 runner (SPEC.md). usage: run_o8.py [frame|select|test].
  frame  : bars cache and the decision-frame cache (scratch).
  select : PHASE 1. All 24 configurations, L1 arm, on the selection months 2021-01..2024-06 -> selection.jsonl, tables_selection.txt, choice.json.
  test   : PHASE 2 (not implemented here beyond its gate): refuses to run unless choice.json is tracked in git at HEAD and unchanged.
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


def test():
    ch = choice_gate()
    raise SystemExit(f"PHASE 2 is not implemented in this phase (choice {ch['config']} verified committed).")


if __name__ == "__main__":
    {"frame": frame, "select": select, "test": test}[sys.argv[1]]()
