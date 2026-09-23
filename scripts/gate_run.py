"""Run one milestone-gate training run in its own directory and record everything the
plan's M1-M4 stop/go gates need.

    python scripts/gate_run.py --name m1a --epochs 2 --physics-off
    python scripts/gate_run.py --name m1b --epochs 2
    python scripts/gate_run.py --name m2  --epochs 5
    python scripts/gate_run.py --name m3  --epochs 20

Each run writes to runs/gates/<name>/ (created fresh):
    training_log.csv               Keras CSVLogger, one row per epoch (all train/val metrics)
    indicator_params_history.csv   ParamsLogger, learned periods per epoch
    analytics.json                 test-set analytics (EV, ROC-AUC, variance-error correlation,
                                   variance dispersion, conformal coverage) + run facts
    predictions_test.npz           raw test predictions, for later evaluation and backtesting
    predictions_cal.npz            the calibration block's predictions
    meta.json                      overrides, git sha, timings, lambdas
Then judge it with scripts/check_gates.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PHYSICS_LAMBDAS = ("LAMBDA_T_PERP", "LAMBDA_CASIMIR", "LAMBDA_HD", "LAMBDA_IFE", "LAMBDA_VAC_OVERFLOW", "LAMBDA_VAC")


def _parse_set(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        out[key.strip()] = val
    return out


def _git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _analytics(result, epochs_requested):
    import numpy as np
    from scipy.special import log_ndtr
    from sklearn.metrics import explained_variance_score, roc_auc_score

    cfg = result.config
    preds = result.predictions
    y = np.asarray(result.y_test, float)
    lc = np.asarray(result.last_close_test, float).reshape(-1)
    scale = float(result.target_scaler.scale_[0])
    mean = float(result.target_scaler.mean_[0])
    d = float(getattr(cfg, "DIR_DEADBAND_BPS", 0.0)) / 1e4

    per_h = {}
    for i, h in enumerate(("h0", "h1", "h2")):
        yt = y[:, i]
        dp = np.asarray(preds["delta"][h], float)
        pp = np.asarray(preds["direction_prob"][h], float)
        vs = np.asarray(preds["variance"][h], float)          # scaled units
        ret = yt / (lc + 1e-12)
        up_all = (ret > d).astype(int)                          # notebook convention: all rows
        mask = np.abs(ret) > d

        err2_scaled = ((yt - dp) / scale) ** 2
        corr_var_err2 = float(np.corrcoef(vs, err2_scaled)[0, 1]) if vs.std() > 0 else 0.0

        mu_raw = dp
        sig_raw = np.sqrt(vs) * scale + 1e-8
        gauss = 1.0 / (1.0 + np.exp(-(log_ndtr((mu_raw - d * lc) / sig_raw) - log_ndtr((-mu_raw - d * lc) / sig_raw))))

        def _auc(t, s):
            try:
                return float(roc_auc_score(t, s)) if 0 < t.sum() < len(t) else float("nan")
            except ValueError:
                return float("nan")

        per_h[h] = {
            "ev_delta": float(explained_variance_score(yt, dp)),
            "corr_delta": float(np.corrcoef(yt, dp)[0, 1]) if dp.std() > 0 else 0.0,
            "rmse_delta": float(np.sqrt(np.mean((yt - dp) ** 2))),
            "roc_auc": _auc(up_all, pp),
            "roc_auc_masked": _auc(up_all[mask], pp[mask]),
            "gauss_roc_auc_masked": _auc(up_all[mask], gauss[mask]),
            "pred_up_rate_masked": float(np.mean(pp[mask] > 0.5)) if mask.any() else float("nan"),
            "gauss_pred_up_rate_masked": float(np.mean(gauss[mask] > 0.5)) if mask.any() else float("nan"),
            "corr_var_err2_scaled": corr_var_err2,
            "var_dispersion": float(vs.std() / vs.mean()) if vs.mean() > 0 else 0.0,
            "var_mean_scaled": float(vs.mean()),
            "pred_std_raw": float(dp.std()),
            "true_std_raw": float(yt.std()),
            "n_test": int(len(yt)),
            "n_masked": int(mask.sum()),
        }
        rep = (result.calibration_report or {}).get(h) or {}
        per_h[h]["coverage90"] = rep.get("coverage90")
        per_h[h]["width90"] = rep.get("width90")
        per_h[h]["temperature"] = rep.get("temperature")

    hist = getattr(result.history, "history", {}) or {}
    epochs_run = len(hist.get("loss", []))
    val_loss = hist.get("val_loss", [])
    return {
        "horizons": per_h,
        "epochs_requested": int(epochs_requested),
        "epochs_run": int(epochs_run),
        "early_stopped": bool(epochs_run < epochs_requested),
        "best_epoch": int(np.argmin(val_loss)) if val_loss else None,
        "calibration_fitted": result.calibration_pipeline is not None,
        "calibration_lambdas_completed": result.calibration_lambdas is not None,
        "pred_scale": scale, "pred_mean": mean,
        "total_trades": None,
        "total_trades_note": "computed by the backtest engine from predictions_test.npz (Phase C5)",
    }


def _save_predictions(path, preds, y, lc, extra=None):
    import numpy as np
    arrays = {"y": np.asarray(y, float), "last_close": np.asarray(lc, float).reshape(-1)}
    for kind in ("delta", "direction_prob", "variance"):
        for h, v in (preds or {}).get(kind, {}).items():
            arrays[f"{kind}_{h}"] = np.asarray(v, float)
    for k, v in (extra or {}).items():
        arrays[k] = np.asarray(v, float)
    np.savez_compressed(path, **arrays)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--physics-off", action="store_true", help="override the six physics lambdas to 0")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="extra Config override (JSON value)")
    ap.add_argument("--out", default=str(REPO / "runs" / "gates"))
    ap.add_argument("--no-calibrate", action="store_true", help="skip the pre-training lambda calibration pass")
    args = ap.parse_args(argv)

    run_dir = Path(args.out) / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    overrides = {"MODEL_PATH": str(run_dir / "weights.h5"), "SCALER_PATH": str(run_dir / "scaler.joblib")}
    if args.physics_off:
        overrides.update({k: 0.0 for k in PHYSICS_LAMBDAS})
    overrides.update(_parse_set(args.set))

    os.chdir(run_dir)  # CSVLogger / ParamsLogger write relative paths
    sys.path.insert(0, str(REPO))
    t0 = time.time()
    from neural_trade.training.trainer import train_and_evaluate  # noqa: E402  (after chdir so relative outputs land here)

    result = train_and_evaluate(
        csv_path=str(REPO / "binance_btcusdt_1min_ccxt.csv"),
        config_overrides=overrides, epochs=args.epochs, force=True,
        calibrate=not args.no_calibrate, fit_calibration=True,
    )
    wall = time.time() - t0

    analytics = _analytics(result, args.epochs)
    analytics["wall_clock_s"] = wall
    (run_dir / "analytics.json").write_text(json.dumps(analytics, indent=2, default=float), encoding="utf-8")

    extra = {}
    cal = result.predictions_calibrated or {}
    for h, v in (cal.get("direction_prob") or {}).items():
        extra[f"calibrated_direction_prob_{h}"] = v
    for h, (lo, hi) in (cal.get("intervals") or {}).items():
        extra[f"interval90_lo_{h}"], extra[f"interval90_hi_{h}"] = lo, hi
    _save_predictions(run_dir / "predictions_test.npz", result.predictions, result.y_test,
                      result.last_close_test, extra)
    if result.predictions_cal is not None:
        _save_predictions(run_dir / "predictions_cal.npz", result.predictions_cal, result.y_cal,
                          result.last_close_cal)

    meta = {
        "name": args.name, "epochs": args.epochs, "physics_off": args.physics_off,
        "overrides": {k: v for k, v in overrides.items() if not k.endswith("_PATH")},
        "git": _git_sha(), "wall_clock_s": wall,
        "ewma_impl": getattr(result.config, "EWMA_IMPL", "scan"),
        "lambda_values_final": result.model.get_lambda_values(),
        "calibration_lambdas": result.calibration_lambdas,
        "fold": {"train": len(result.fold.train), "val": len(result.fold.val), "cal": len(result.fold.cal),
                 "test": len(result.fold.test), "gap": result.fold.gap} if result.fold else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"\n[gate_run] {args.name}: {analytics['epochs_run']}/{args.epochs} epochs in {wall/60:.1f} min -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
