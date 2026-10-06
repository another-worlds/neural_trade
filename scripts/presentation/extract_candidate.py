"""Data for the candidate presentations #3 (seed ensemble, C3) and #4 (the gated TA rule on the 10-day clip100 model).

    CUDA_VISIBLE_DEVICES=-1 python scripts/presentation/extract_candidate.py ensemble   # -> docs/presentation/data_ensemble.json
    CUDA_VISIBLE_DEVICES=-1 python scripts/presentation/extract_candidate.py ta         # -> docs/presentation/data_ta.json

Zero trading costs (D-044) unless a field says otherwise; dev folds only, fold -1 is never read (D-020).
#3: per dev fold, the mean of three seeds' stored predictions (every head, calibration and dev blocks) of the 360-day
model (runs/scenarios/long_360d_stab), traded by calibrated_quantile 0.9 at position size 0.7 (configs/candidates, C3).
#4: the micro_l2 scenario's clip100 variant (10-day training, gradient clip 100; three seeds on dev fold -2), traded by
gated_ta: an SMA 20/50 cross of the close, entered only when the model's predicted sigma is at or above its calibration
80th percentile (configs/strategy_studies/zero_cost_status.yaml, id gt).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pres_extract", HERE / "extract.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)  # noqa: E305  (imports neural_trade first)

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame  # noqa: E402
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block  # noqa: E402

REPO = X.REPO
r = X.r
THIN = X.THIN


def bt_run(sig, bars, bar_minutes, strategy, params, cost_rt=0.0, seeds=0):
    return fit_and_backtest(sig, bars, strategy=strategy, strategy_params=params,
                            backtest_params={**X.costs(cost_rt), "random_seeds": seeds},
                            bar_minutes=bar_minutes)


def summ(bt) -> dict:
    out = X.summary(bt)
    tf = bt.trades_frame()
    if len(tf):
        rb = tf["return_pct"].to_numpy(float) * 100
        out.update(avg_win_bps=r(rb[rb > 0].mean() if (rb > 0).any() else None, 2),
                   avg_loss_bps=r(rb[rb <= 0].mean() if (rb <= 0).any() else None, 2),
                   avg_bars=r(tf["bars_held"].mean(), 1))
    return out


def view(bars, extra, bt, sig=None, feature=None) -> dict:
    close = np.asarray(bars.close, dtype=float)
    ts = np.asarray(extra["anchor_timestamp"]).astype(str)
    eq = np.asarray(bt.equity, dtype=float)[1:]
    dd = 1 - eq / np.maximum.accumulate(eq)
    n = len(close)
    pick = np.arange(0, n, THIN)
    tf = bt.trades_frame()
    side = np.where(tf["side"].astype(str).str.upper().str.startswith("L"), 1, -1) if len(tf) else np.array([])
    eb = tf["entry_bar"].to_numpy(int) if len(tf) else np.array([], int)
    xb = tf["exit_bar"].to_numpy(int) if len(tf) else np.array([], int)
    out = {"t": [ts[i][:16] for i in pick], "close": [r(close[i], 1) for i in pick],
           "close_full": [r(x, 2) for x in close], "t0": ts[0][:16], "equity_full": [r(x, 2) for x in eq],
           "equity": [r(eq[i], 2) for i in pick], "bh": [r(10000 * close[i] / close[0], 2) for i in pick],
           "dd": [r(float(dd[a:a + THIN].max()), 5) for a in pick], "first": ts[0][:16], "last": ts[-1][:16],
           "trades": {"side": side.tolist(), "ret_bps": [r(x, 2) for x in tf["return_pct"].to_numpy(float) * 100] if len(tf) else [],
                      "t_entry": [ts[i][:16] for i in eb], "t_exit": [ts[min(i, n - 1)][:16] for i in xb],
                      "px": [r(x, 1) for x in tf["entry_price"].to_numpy(float)] if len(tf) else [],
                      "bars": tf["bars_held"].astype(int).tolist() if len(tf) else [],
                      "reason": tf["exit_reason"].astype(str).tolist() if len(tf) else [],
                      "hour": [int(ts[i][11:13]) for i in eb]}}
    if feature is not None:
        out["trades"]["feature"] = [r(feature[i], 3) for i in eb]
    return out


def report_metrics(d: Path) -> dict:
    rep = json.loads((d / "eval_report_dev.json").read_text(encoding="utf-8"))
    out = {}
    for h in ("h0", "h1", "h2"):
        m, lr = rep["model"]["horizons"][h], rep["baselines"]["logreg_lags"]["horizons"][h]
        out[h] = {"auc": r(m["direction"]["auc"]), "auc_logreg": r(lr["direction"]["auc"]), "ece": r(m["direction"]["ece_pos"]),
                  "crpss": r(m["variance"]["crpss"]), "coverage90": r(m["variance"]["coverage90"]),
                  "spearman": r(m["variance"]["corr_var_err2_spearman"])}
    return out


def up_auc(frame, sig, i) -> float:
    h = HORIZONS[i]
    y = np.asarray(frame.y[h] if isinstance(frame.y, dict) else frame.y[:, i], dtype=float)
    p = np.asarray(sig.oos.p[:, i], dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (y != 0)
    return float(roc_auc_score(y[ok] > 0, p[ok]))


def mean_frame(frames):
    f0 = frames[0]
    avg = lambda a: {h: np.mean([getattr(f, a)[h] for f in frames], axis=0) for h in HORIZONS}  # noqa: E731
    cal = avg("direction_prob_calibrated") if all(f.direction_prob_calibrated is not None for f in frames) else None
    out = PredictionFrame(f0.y, f0.last_close, avg("delta"), avg("direction_prob"), avg("variance_scaled"),
                          f0.pred_scale, f0.pred_mean, f0.horizon_steps, f0.split, cal)
    out.meta = dict(getattr(f0, "meta", {}) or {})
    return out


def ensemble() -> dict:
    cq = "calibrated_quantile"
    data = {"subject": "ensemble", "folds": {}}
    stab = REPO / "runs/scenarios/long_360d_stab"
    for fold in ("f-3", "f-2"):
        dirs = sorted(Path(p) for p in glob.glob(str(stab / f"2026*__{fold}__s*")))
        cals, oos, sigs = [], [], []
        for d in dirs:
            c, _, _ = load_block(d / "predictions_cal.npz")
            o, bars, extra = load_block(d / "predictions_oos.npz")
            cals.append(c)
            oos.append(o)
            sigs.append(BlockSignals.build(c, o))
        esig = BlockSignals.build(mean_frame(cals), mean_frame(oos))
        bar_minutes = float(extra["bar_minutes"])
        c3, _ = bt_run(esig, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 0.7}, seeds=100)
        full, _ = bt_run(esig, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 1.0})
        seeds = [summ(bt_run(s, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 1.0})[0]) for s in sigs]
        seeds07 = [summ(bt_run(s, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 0.7})[0]) for s in sigs]
        wds = [np.asarray(s.oos.weighted_direction, dtype=float) for s in sigs] + [np.asarray(esig.oos.weighted_direction, dtype=float)]
        corr = np.corrcoef(np.vstack(wds)).round(3).tolist()
        aucs = {"seeds": [[r(up_auc(o, s, i)) for i in range(3)] for o, s in zip(oos, sigs)],
                "ensemble": [r(up_auc(oos[0], esig, i)) for i in range(3)]}
        hist = {k: np.histogram(w, bins=50, range=(0.44, 0.56))[0].tolist() for k, w in zip(["s0", "s1", "s2", "ens"], wds)}
        sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        size_curve = [{"size": z, **summ(bt_run(esig, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": z})[0])} for z in sizes]
        qs = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
        q_curve = [{"q": q, "ens": summ(bt_run(esig, bars, bar_minutes, cq, {"entry_quantile": q, "size": 0.7})[0]),
                    "seeds": [summ(bt_run(s, bars, bar_minutes, cq, {"entry_quantile": q, "size": 0.7})[0]) for s in sigs]} for q in qs]
        # entries in common: share of the ensemble's entry bars within 2 bars of a seed's entry
        ent_e = set(t.entry_bar for t in c3.trades)
        overlap = []
        for s in sigs:
            es = set(t.entry_bar for t in bt_run(s, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 1.0})[0].trades)
            near = sum(1 for b in ent_e if any(b + k in es for k in (-2, -1, 0, 1, 2)))
            overlap.append(r(near / max(len(ent_e), 1)))
        cost_curve = {c: summ(bt_run(esig, bars, bar_minutes, cq, {"entry_quantile": 0.9, "size": 0.7}, c)[0])["ret"] for c in (0, 0.5, 1, 1.5, 2, 3)}
        rnd = c3.baselines.get("random_same_freq", {})
        data["folds"][fold] = {
            "cells": [d.name for d in dirs], "c3": summ(c3), "full_size": summ(full), "seeds": seeds, "seeds07": seeds07,
            "corr_labels": ["seed 0", "seed 1", "seed 2", "ensemble"], "corr": corr, "auc": aucs, "wd_hist": hist,
            "size_curve": size_curve, "q_curve": q_curve, "entry_overlap": overlap, "cost_curve": cost_curve,
            "random_null": {k: r(v) for k, v in rnd.items() if isinstance(v, (int, float))},
            "fit": [report_metrics(d) for d in dirs], "view": view(bars, extra, c3),
            "training": [{k: v for k, v in X.training(d).items() if k in ("epoch", "loss", "val_loss", "served_epoch", "val_dir_mcc")} for d in dirs]}
    data["manifest"] = json.loads((REPO / "configs/candidates/manifest.json").read_text(encoding="utf-8"))["candidates"]["C3"]
    return data


def ta() -> dict:
    import pandas as pd

    gt = "gated_ta"
    base = {"primary": "ma_cross", "q": 0.8}
    dirs = sorted(Path(p) for p in glob.glob(str(REPO / "runs/scenarios/micro_l2/2026*-clip100__f-2__s*")))
    data = {"subject": "ta", "cells": []}
    for k, d in enumerate(dirs):
        cal, _, _ = load_block(d / "predictions_cal.npz")
        o, bars, extra = load_block(d / "predictions_oos.npz")
        sig = BlockSignals.build(cal, o)
        bar_minutes = float(extra["bar_minutes"])
        bt, strat = bt_run(sig, bars, bar_minutes, gt, base, seeds=100)
        close = np.asarray(bars.close, dtype=float)
        sh_model = np.asarray(sig.oos.sigma_for("model", -1), dtype=float) / close
        sh_ewma = np.asarray(sig.oos.sigma_for("ewma", -1), dtype=float) / close
        variants = {}
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            variants[f"model|{q}"] = summ(bt_run(sig, bars, bar_minutes, gt, {"primary": "ma_cross", "q": q})[0])
            variants[f"ewma|{q}"] = summ(bt_run(sig, bars, bar_minutes, gt,
                                                {"primary": "ma_cross", "q": q, "sigma_source": "ewma"})[0])
        variants["none|0.001"] = summ(bt_run(sig, bars, bar_minutes, gt, {"primary": "ma_cross", "q": 0.001})[0])
        variants["bollinger|0.8"] = summ(bt_run(sig, bars, bar_minutes, gt, {"primary": "bollinger", "q": 0.8})[0])
        cq = summ(bt_run(sig, bars, bar_minutes, "calibrated_quantile", {"entry_quantile": 0.9, "size": 1.0})[0])
        rnd = bt.baselines.get("random_same_freq", {})
        cell = {"name": d.name, "seed": d.name.split("__")[-1], "gate": r(strat.gate, 7), "summary": summ(bt),
                "variants": variants, "cq09": cq, "fit": report_metrics(d),
                "random_null": {kk: r(v) for kk, v in rnd.items() if isinstance(v, (int, float))},
                "training": {kk: v for kk, v in X.training(d).items() if kk in ("epoch", "loss", "val_loss", "served_epoch", "val_dir_mcc", "grad_norm")},
                "view": view(bars, extra, bt, feature=sh_model * 1e4),
                "cost_curve": {c: summ(bt_run(sig, bars, bar_minutes, gt, base, c)[0])["ret"] for c in (0, 1, 2, 4, 6, 10, 16, 26)}}
        if k == 0:
            s = pd.Series(close)
            pick = np.arange(0, len(close), 1)
            ts = np.asarray(extra["anchor_timestamp"]).astype(str)
            cell["gate_view"] = {"t": [ts[i][:16] for i in pick], "close": [r(close[i], 1) for i in pick],
                                 "sma20": [r(x, 1) for x in s.rolling(20).mean().to_numpy()[pick]],
                                 "sma50": [r(x, 1) for x in s.rolling(50).mean().to_numpy()[pick]],
                                 "sigma_model_bps": [r(x * 1e4, 3) for x in sh_model[pick]],
                                 "sigma_ewma_bps": [r(x * 1e4, 3) for x in sh_ewma[pick]]}
            # sigma forecast quality at h2 (the gate's horizon): predicted sigma vs realised RMS move, by decile
            y = np.asarray(o.y["h2"] if isinstance(o.y, dict) else o.y[:, 2], dtype=float)
            sd = np.asarray(sig.oos.sigma[:, 2], dtype=float)
            ok = np.isfinite(y) & np.isfinite(sd)
            y, sd = y[ok], sd[ok]
            e = np.quantile(sd, np.linspace(0, 1, 11))
            j = np.clip(np.searchsorted(e, sd, side="right") - 1, 0, 9)
            cell["sigma_cal"] = [{"sigma": r(sd[j == b].mean(), 2), "rms": r(np.sqrt(np.mean(y[j == b] ** 2)), 2)} for b in range(10)]
            cell["gate_share"] = r(float(np.mean(sh_model >= strat.gate)))
            cfg = (d / "config.yaml").read_text(encoding="utf-8")
            cell["config_lines"] = [ln.strip() for ln in cfg.splitlines()
                                    if ln.split(":")[0].strip() in ("GRAD_CLIP_NORM", "BATCH_SIZE", "LEARNING_RATE", "EPOCHS",
                                                                    "LOOKBACK", "HORIZONS", "MAX_SEQUENCE_COUNT", "N_FOLDS")]
        rep = json.loads((d / "eval_report_dev.json").read_text(encoding="utf-8"))
        cell["blocks"] = rep["meta"]["blocks"]
        data["cells"].append(cell)
    return data


def main() -> None:
    which = sys.argv[1]
    data = ensemble() if which == "ensemble" else ta()
    out = REPO / f"docs/presentation/data_{which}.json"
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
    if which == "ensemble":
        for f, v in data["folds"].items():
            print(f, "c3", v["c3"], "seeds", [s["ret"] for s in v["seeds"]], "win", [s["win"] for s in v["seeds"]], v["c3"]["win"])
            print("  corr", v["corr"][0], "auc", v["auc"], "overlap", v["entry_overlap"])
    else:
        for c in data["cells"]:
            print(c["seed"], c["summary"], "none", c["variants"]["none|0.001"]["ret"], "ewma.8", c["variants"]["ewma|0.8"]["ret"],
                  "boll", c["variants"]["bollinger|0.8"]["ret"])
        print(data["cells"][0].get("config_lines"), data["cells"][0]["gate_share"])


if __name__ == "__main__":
    main()
