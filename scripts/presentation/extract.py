"""Collect the data of the leader-model presentation (docs/presentation/) from committed runs.

    CUDA_VISIBLE_DEVICES=-1 python scripts/presentation/extract.py      # writes docs/presentation/data.json

Everything is recomputed from stored files, on CPU, at zero trading costs (D-044) unless a field says otherwise:
the leader is candidate C1 (configs/candidates/manifest.json): the 360-day model of long_360d_stab, fold -2,
seed 0, calibrated_quantile entry_quantile 0.9, size 1.0. Only dev folds (-3, -2) are read; fold -1 (the test
fold) is never touched (D-020).
"""
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

import neural_trade  # noqa: F401  (CUDA DLLs; before anything imports tensorflow)
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block

REPO = Path(__file__).resolve().parents[2]
STAB = REPO / "runs/scenarios/long_360d_stab"
LEADER = STAB / "20260930T094257Z-dce15ed-e3669618-default__f-2__s0"
REFERENCE = REPO / "runs/scenarios/reference_default"
OUT = REPO / "docs/presentation/data.json"
ZERO = {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0}
C1 = {"entry_quantile": 0.9, "size": 1.0}
HZ = ("h0", "h1", "h2")
THIN = 15  # bars per point of the per-bar series


def r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, n)


def costs(round_trip_bps: float) -> dict:
    return {"fee_bps": round_trip_bps / 2.0, "half_spread_bps": 0.0, "slippage_bps": 0.0}


def load_cell(d: Path):
    cal, _, _ = load_block(d / "predictions_cal.npz")
    oos, bars, extra = load_block(d / "predictions_oos.npz")
    return BlockSignals.build(cal, oos), bars, extra, oos


def run(sig, bars, bar_minutes, params=C1, cost_rt=0.0, seeds=0, strategy="calibrated_quantile"):
    bt, strat = fit_and_backtest(sig, bars, strategy=strategy, strategy_params=params,
                                 backtest_params={**costs(cost_rt), "random_seeds": seeds},
                                 bar_minutes=bar_minutes)
    return bt, strat


def summary(bt) -> dict:
    s = bt.summary
    return {"ret": r(s["total_return"]), "sharpe": r(s["sharpe_net"], 2), "win": r(s["hit_rate"]),
            "dd": r(s["max_drawdown"]), "trades": int(s["n_trades"]),
            "edge_bps": r(s.get("gross_edge_per_trade_bps"), 3), "exposure": r(s.get("exposure")),
            "bh": r(bt.baselines["buy_and_hold"]["total_return"])}


def training(d: Path) -> dict:
    rows = list(csv.DictReader(open(d / "training_log.csv", encoding="utf-8")))
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))

    def col(name):
        return [r(row.get(name) or "nan", 5) for row in rows]

    terms = ["point_loss", "trend_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss", "vol_loss",
             "t_perp_loss", "casimir_loss", "vac_loss", "hd_loss", "ife_loss", "vac_overflow_loss", "reg_loss"]
    out = {"epoch": [int(float(row["epoch"])) for row in rows], "loss": col("loss"), "val_loss": col("val_loss"),
           "grad_norm": col("grad_global_norm"), "lr": col("lr_used"), "lr_indicator": col("lr_indicator_used"),
           "served_epoch": status.get("weights_epoch"), "sec_per_step": r(status.get("sec_per_step"), 4),
           "elapsed_s": r(status.get("elapsed_seconds"), 1),
           "terms": {t: {"train": col(t), "val": col("val_" + t)} for t in terms}}
    for m in ("dir_mcc", "dir_bal_acc", "dir_brier", "dir_ece", "mean_dir_prob", "pred_up_rate"):
        out["val_" + m] = {h: col(f"val_{m}_{h}") for h in HZ}
    return out


def indicators(d: Path) -> dict:
    rows = list(csv.DictReader(open(d / "indicator_params_history.csv", encoding="utf-8")))
    init = json.loads((d / "period_init.json").read_text(encoding="utf-8"))["periods"]
    names = [k for k in rows[0] if k.startswith(("ma_", "macd_", "rsi_", "bb_"))]
    return {"names": names, "init": {k: r(init.get(k), 3) for k in names},
            "epoch": [int(float(x["epoch"])) for x in rows],
            "history": {k: [r(x[k], 3) for x in rows] for k in names}}


def fit(d: Path, oos, sig) -> dict:
    rep = json.loads((d / "eval_report_dev.json").read_text(encoding="utf-8"))
    out = {"n": rep["model"]["horizons"]["h0"]["n"], "n_eff": {}, "metrics": {}, "reliability": {}, "sigma_cal": {},
           "pit": {}, "p_hist": {}}
    for h in HZ:
        m = rep["model"]["horizons"][h]
        lr = rep["baselines"]["logreg_lags"]["horizons"][h]
        cp = rep["baselines"]["class_prior"]["horizons"][h]
        cv = rep["baselines"]["const_var"]["horizons"][h]
        out["n_eff"][h] = m.get("n_eff")
        out["metrics"][h] = {
            "auc": r(m["direction"]["auc"]), "auc_logreg": r(lr["direction"]["auc"]),
            "auc_prior": r(cp["direction"]["auc"]), "brier": r(m["direction"]["brier"]),
            "brier_logreg": r(lr["direction"]["brier"]), "ece": r(m["direction"]["ece_pos"]),
            "acc": r(m["direction"]["acc"]), "mcc": r(m["direction"]["mcc"]),
            "crps": r(m["variance"]["crps"], 2), "crps_const": r(cv["variance"]["crps"], 2),
            "crpss": r(m["variance"]["crpss"]), "coverage90": r(m["variance"]["coverage90"]),
            "width90": r(m["variance"]["width90"], 1), "spearman_var_err2": r(m["variance"]["corr_var_err2_spearman"]),
            "delta_skill": r(m["delta"]["skill_vs_zero"], 5)}
        for key in ("logreg_lags", "class_prior"):
            for mk in ("direction/auc",):
                z = rep["beats_baseline"][key][mk]
                out["metrics"][h][f"beats_{key}"] = bool(z[h]) if isinstance(z, dict) and h in z else None
    # reliability of the calibrated P(up), sigma calibration and PIT, from the stored arrays
    k = {"h0": 0, "h1": 1, "h2": 2}
    for h in HZ:
        i = k[h]
        y = np.asarray(oos.y[h] if isinstance(oos.y, dict) else oos.y[:, i], dtype=float)
        p = np.asarray(sig.oos.p[:, i], dtype=float)
        sd = np.asarray(sig.oos.sigma[:, i], dtype=float)
        ok = np.isfinite(y) & np.isfinite(p) & (y != 0)
        y, p, sdv = y[ok], p[ok], sd[ok]
        hb = int(oos.horizon_steps[i]) if hasattr(oos, "horizon_steps") else (10, 15, 20)[i]
        edges = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        rel = []
        for b in range(10):
            s = idx == b
            n = int(s.sum())
            up = float(np.mean(y[s] > 0))
            ne = max(n // hb, 1)
            rel.append({"p": r(np.mean(p[s])), "up": r(up), "n": n, "se": r(math.sqrt(up * (1 - up) / ne))})
        out["reliability"][h] = rel
        e2 = np.quantile(sdv, np.linspace(0, 1, 11))
        j = np.clip(np.searchsorted(e2, sdv, side="right") - 1, 0, 9)
        out["sigma_cal"][h] = [{"sigma": r(np.mean(sdv[j == b]), 2), "rms": r(np.sqrt(np.mean(y[j == b] ** 2)), 2),
                                "n": int((j == b).sum())} for b in range(10)]
        from scipy.stats import norm
        pit = norm.cdf(y / np.maximum(sdv, 1e-9))
        out["pit"][h] = np.histogram(pit, bins=20, range=(0, 1))[0].tolist()
        out["p_hist"][h] = {"counts": np.histogram(p, bins=40, range=(0.4, 0.6))[0].tolist(), "lo": 0.4, "hi": 0.6}
    return out


def backtest_view(sig, bars, extra, bt, strat) -> dict:
    close = np.asarray(bars.close, dtype=float)
    ts = np.asarray(extra["anchor_timestamp"]).astype(str)
    eq = np.asarray(bt.equity, dtype=float)[1:]
    peak = np.maximum.accumulate(eq)
    dd = 1 - eq / peak
    n = len(close)
    pick = np.arange(0, n, THIN)
    dd_b = [float(dd[a:a + THIN].max()) for a in pick]
    wd = np.asarray(sig.oos.weighted_direction, dtype=float)
    sret = np.asarray(sig.oos.sigma_ret[:, 1], dtype=float)
    tf = bt.trades_frame()
    side = np.where(tf["side"].astype(str).str.upper().str.startswith("L"), 1, -1)
    eb = tf["entry_bar"].to_numpy(int)
    xb = tf["exit_bar"].to_numpy(int)
    ret_bps = tf["return_pct"].to_numpy(float) * 100.0  # return_pct is in percent
    hours = np.array([int(t[11:13]) if len(t) >= 13 else -1 for t in ts])
    params = {}
    for key in ("long_above", "short_below", "median"):
        v = getattr(strat, key, None)
        if v is None and hasattr(strat, "params"):
            v = strat.params.get(key) if isinstance(strat.params, dict) else getattr(strat.params, key, None)
        params[key] = r(v, 5)
    return {
        "t": [ts[i][:16] for i in pick], "close": [r(close[i], 1) for i in pick],
        # every bar's close, drawn at the bar's END (t0 + (i + 1) minutes): a fill at the next bar's open then sits on it
        "close_full": [r(x, 2) for x in close], "t0": ts[0][:16], "equity_full": [r(x, 2) for x in eq],
        "equity": [r(eq[i], 2) for i in pick], "bh": [r(10000 * close[i] / close[0], 2) for i in pick],
        "dd": [r(x, 5) for x in dd_b], "wd": [r(wd[i], 4) for i in pick],
        "thresholds": params,
        "trades": {"entry": eb.tolist(), "exit": xb.tolist(), "side": side.tolist(),
                   "ret_bps": [r(x, 2) for x in ret_bps], "wd": [r(wd[i], 4) for i in eb],
                   "sigma_bps": [r(1e4 * sret[i], 2) for i in eb], "hour": [int(hours[i]) for i in eb],
                   "reason": tf["exit_reason"].astype(str).tolist(), "bars": tf["bars_held"].astype(int).tolist(),
                   "t_entry": [ts[i][:16] for i in eb], "t_exit": [ts[min(i, n - 1)][:16] for i in xb],
                   "px": [r(x, 1) for x in tf["entry_price"].to_numpy(float)]},
        "n_bars": int(n), "thin": THIN, "first": ts[0][:16], "last": ts[-1][:16],
    }


def whatif(sig, bars, bar_minutes) -> dict:
    qs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
    cs = [0, 0.5, 1, 2, 4, 6, 10, 16, 26]
    grid = {}
    for q in qs:
        for c in cs:
            bt, _ = run(sig, bars, bar_minutes, {"entry_quantile": q, "size": 1.0}, c)
            grid[f"{q}|{c}"] = summary(bt)
    return {"q": qs, "cost": cs, "grid": grid}


def main() -> None:
    data = {"leader": LEADER.name}
    sig, bars, extra, oos = load_cell(LEADER)
    bar_minutes = float(extra["bar_minutes"])
    bt, strat = run(sig, bars, bar_minutes, C1, 0.0, seeds=100)
    data["summary"] = summary(bt)
    rnd = bt.baselines.get("random_same_freq", {})
    data["random_null"] = {k: r(v) for k, v in rnd.items() if isinstance(v, (int, float))}
    bt26, _ = run(sig, bars, bar_minutes, C1, 26.0, seeds=0)
    data["summary_26bps"] = summary(bt26)
    data["gross_equity_26bps"] = [r(x, 2) for x in np.asarray(bt26.equity, dtype=float)[1::THIN]]
    data["backtest"] = backtest_view(sig, bars, extra, bt, strat)
    data["training"] = training(LEADER)
    data["indicators"] = indicators(LEADER)
    data["fit"] = fit(LEADER, oos, sig)
    data["whatif"] = whatif(sig, bars, bar_minutes)
    rep = json.loads((LEADER / "eval_report_dev.json").read_text(encoding="utf-8"))
    data["blocks"] = rep["meta"]["blocks"]

    # the six 360-day cells (folds -3 / -2 x seeds 0-2): training curves, periods, C1/C2 at zero cost, cost curves
    cells = []
    for d in sorted(Path(p) for p in glob.glob(str(STAB / "2026*"))):
        s2, b2, e2, _ = load_cell(d)
        bm2 = float(e2["bar_minutes"])
        c1, _ = run(s2, b2, bm2, C1)
        c2, _ = run(s2, b2, bm2, {"entry_quantile": 0.95, "size": 1.0})
        c1s, _ = run(s2, b2, bm2, {"entry_quantile": 0.9, "size": 0.7})
        eq = np.asarray(c1.equity, dtype=float)[1::THIN * 2]
        cost_curve = {c: summary(run(s2, b2, bm2, C1, c)[0])["ret"] for c in (0, 0.5, 1, 1.5, 2, 3, 4)}
        ind = indicators(d)
        cells.append({"name": d.name, "fold": d.name.split("__")[1], "seed": d.name.split("__")[2],
                      "c1": summary(c1), "c2": summary(c2), "c1_size07": summary(c1s),
                      "equity": [r(x, 1) for x in eq], "cost_curve": cost_curve,
                      "training": {k: v for k, v in training(d).items() if k in ("epoch", "loss", "val_loss", "served_epoch")},
                      "final_periods": {k: v[-1] for k, v in ind["history"].items()}})
    data["cells"] = cells
    data["manifest"] = json.loads((REPO / "configs/candidates/manifest.json").read_text(encoding="utf-8"))["candidates"]
    for c in data["manifest"].values():
        c.pop("files", None)

    # the 7-day reference model on its dev cells (f-3, f-2; never f-1), the same strategy, zero cost and 26 bps
    ref = []
    for d in sorted(Path(p) for p in glob.glob(str(REFERENCE / "2026*__f-[23]__s*"))):
        if not (d / "predictions_oos.npz").exists():
            continue
        s3, b3, e3, _ = load_cell(d)
        bm3 = float(e3["bar_minutes"])
        ref.append({"name": d.name, "zero": summary(run(s3, b3, bm3, C1)[0]),
                    "c26": summary(run(s3, b3, bm3, C1, 26.0)[0])})
    data["reference_7d"] = ref

    # every stored model x strategy at zero cost (Z0), dev cells only
    zc = REPO / "runs/experiments/micro_loop_v1/zero_cost_status.csv"
    data["zero_cost_status"] = [{k: (r(v) if k in ("ret", "ret_min", "sharpe", "hit", "dd", "dd_max", "trades", "bh")
                                     and v not in ("", None) else v) for k, v in row.items()}
                                for row in csv.DictReader(open(zc, encoding="utf-8"))]

    # screen campaign level 1 (maths and stability): per block, trials and survivors
    screens = {}
    for d in sorted(glob.glob(str(REPO / "runs/screens/l1_*"))):
        rows = []
        for f in glob.glob(str(Path(d) / "results.shard-*.jsonl")):
            rows += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        surv = sum(1 for x in rows if x.get("passed") or x.get("survived") or x.get("verdict") == "pass")
        screens[Path(d).name] = {"trials": len(rows), "survivors": surv,
                                 "keys": sorted(rows[0].keys())[:40] if rows else []}
    data["screens"] = screens

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
    print("leader", data["summary"], "26bps", data["summary_26bps"])
    print("ref cells", len(ref), "screens", {k: (v["trials"], v["survivors"]) for k, v in screens.items()})
    print("screen keys", next(iter(screens.values()))["keys"] if screens else None)


if __name__ == "__main__":
    main()
