"""Part 2 (SPEC.md): one regime filter chosen on the hourly dev folds, applied once to the held-out period."""
import argparse, importlib.util, itertools, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import regime_vars as R  # noqa: E402

spec = importlib.util.spec_from_file_location("hourly", "D:/nt/nt_tactical/runs/tactical/hourly/hourly.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)      # chdirs to D:/nt/nt_tactical; read-only use
OUT = os.path.join(HERE, "results.jsonl")


def minutes():
    s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "close"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.sort_index(); return s.index, s["close"].values


def null95(D, idx, rng):
    k = len(idx); a, b = D["pl_long"][idx], D["pl_short"][idx]
    return float(np.quantile([np.mean(np.where(rng.random(k) < 0.5, a, b)) for _ in range(200)], 0.95) * 1e4)


def t_interval(v):
    n = len(v); m = float(np.mean(v)); se = np.std(v, ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return [round(m - t * se, 3), round(m + t * se, 3)]


def candidates():
    out = []
    for v in ("vol", "vtrend", "trend", "session"):
        for r in (1, 2):
            out += [(v, frozenset(c)) for c in itertools.combinations(range(3), r)]
    return out + [("weekend", frozenset([0])), ("weekend", frozenset([1]))]


def name(f): return f"{f[0]}:" + "+".join(R.CELLS[f[0]][j] for j in sorted(f[1]))


def main():
    t0 = time.time(); a = argparse.Namespace(T=12, tp=1.5, sl=1.5, q=0.2, feat="rich", primary="hgb", target="barrier", meta="mag")
    ranking = json.load(open("D:/nt/nt_tactical/runs/tactical/hourly/ranking.json"))
    assert ranking["top3"][0] == "T12_tp1.5_sl1.5_rich_hgb_barrier_mag", ranking["top3"][0]
    D = H.build(a); idx = np.arange(len(D["F"])); dev = idx[D["t_exit"] < H.HOLD]; held = idx[D["t_entry"] >= H.HOLD]
    mi, mc = minutes(); te_all = pd.DatetimeIndex(D["t_entry"])
    mpos = mi.searchsorted(te_all + pd.Timedelta("59min"), side="right") - 1   # last available minute of the entry bar (<= :59)
    assert (mpos >= 0).all() and (mi[mpos] >= te_all).all()

    def codes_for(rows, train_rows, cut_fit=None):
        c, cf = R.cell_codes(mc, mpos[rows], te_all[rows] + pd.Timedelta("1h"), train_pos=mpos[train_rows] if cut_fit is None else None, cut_fit=cut_fit)
        return c, cf

    def trades(r, tr, te):
        tt = pd.DatetimeIndex(r["meta"]["_t"]); rows = te_all.get_indexer(tt)
        c, cf = codes_for(rows, tr); return rows, r["meta"]["_pl"], c

    rng = np.random.default_rng(0); edges = np.linspace(len(dev) * 0.3, len(dev), 9).astype(int); folds = []
    for f in range(8):
        tr = dev[:max(1, edges[f] - 2 * a.T)]; te = dev[edges[f]:edges[f + 1]]
        r = H.run_split(D, a, tr, te, rng); rows, pl, c = trades(r, tr, te); folds.append(dict(rows=rows, pl=pl, c=c)); print("fold", f, len(rows), flush=True)
    pooled = {k: np.concatenate([f["c"][k] for f in folds]) for k in R.CELLS}; npool = len(pooled["vol"])
    share = {(k, j): float(np.mean(pooled[k] == j)) for k in R.CELLS for j in range(len(R.CELLS[k]))}

    def score(filt, rng_):
        ex = []
        for f in folds:
            keep = np.isin(f["c"][filt[0]], list(filt[1])) if filt else np.ones(len(f["rows"]), bool)
            if keep.sum() < 10:
                continue
            ex.append(float(np.mean(f["pl"][keep]) * 1e4 - null95(D, f["rows"][keep], rng_)))
        return (float(np.mean(ex)) if len(ex) >= 6 else None), len(ex), ex
    srng = np.random.default_rng(0); table = []
    base = score(None, srng)
    for filt in candidates():
        ok = all(share[(filt[0], j)] >= 0.20 for j in filt[1])
        s = score(filt, srng) if ok else (None, 0, [])
        table.append({"filter": name(filt), "share": round(sum(share[(filt[0], j)] for j in filt[1]), 3), "eligible": ok, "dev_excess": s[0], "n_folds": s[1], "per_fold": s[2]})
    elig = [(t["dev_excess"], i) for i, t in enumerate(table) if t["dev_excess"] is not None]
    best = max(elig)[1]; chosen = candidates()[best]; print("chosen", name(chosen), table[best]["dev_excess"], "unfiltered", base[0], flush=True)

    # ---- the one held-out look
    tr = dev[D["t_exit"][dev] < H.HOLD - pd.Timedelta(hours=2 * a.T)]
    r = H.run_split(D, a, tr, held, np.random.default_rng(0))
    tt = pd.DatetimeIndex(r["meta"]["_t"]); rows = te_all.get_indexer(tt); pl = r["meta"]["_pl"]
    c, _ = codes_for(rows, tr)
    days = max(1, (D["t_entry"][held[-1]] - D["t_entry"][held[0]]).days); hrng = np.random.default_rng(0); res = {}
    for label, keep in (("unfiltered", np.ones(len(rows), bool)), ("filtered", np.isin(c[chosen[0]], list(chosen[1])))):
        months = pd.Series(pl[keep], index=tt[keep]).groupby(pd.Grouper(freq="M")).mean().dropna() * 1e4
        res[label] = {"n": int(keep.sum()), "trades_per_day": float(keep.sum() / days), "hit": float(np.mean(pl[keep] > 0)), "bps": float(np.mean(pl[keep]) * 1e4),
                      "null95": null95(D, rows[keep], hrng), "monthly_bps_ci": t_interval(months.values), "n_months": int(len(months)),
                      "months_pos": int((months > 0).sum())}
        res[label]["success"] = bool(res[label]["bps"] > res[label]["null95"] and res[label]["monthly_bps_ci"][0] > 0)
    rec = {"part": 2, "config": "T12_tp1.5_sl1.5_rich_hgb_barrier_mag", "unfiltered_dev_excess": base[0], "dev_table": table, "chosen": name(chosen),
           "chosen_dev_excess": table[best]["dev_excess"], "held_out": res, "seconds": round(time.time() - t0, 1)}
    open(OUT, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    print(json.dumps(res, indent=1))
    for t in sorted(table, key=lambda t: -(t["dev_excess"] if t["dev_excess"] is not None else -1e9)):
        print(t["filter"], t["share"], t["eligible"], t["dev_excess"], t["n_folds"])


if __name__ == "__main__":
    main()
