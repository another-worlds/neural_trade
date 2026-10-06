"""Record or verify a small deterministic training run, to prove a refactor changed no behaviour.

    python scripts/golden_run.py record OUT.npz [--repo PATH]
    python scripts/golden_run.py verify OUT.npz [--repo PATH] [--atol 1e-6]

Runs train_and_evaluate on CPU with TF_DETERMINISTIC_OPS, 3,000 sequences and 2 epochs
(lambda calibration pass included), then stores the per-epoch history, calibrated
lambdas, learned periods and the test predictions. `verify` re-runs and compares every
array. A pure code move reproduces the numbers to float32 round-off or better; anything
larger means the move changed behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

OVERRIDES = {"MAX_SEQUENCE_COUNT": 3000, "BATCH_SIZE": 64}
EPOCHS = 2


def _train_and_evaluate():
    try:
        from neural_trade.training.trainer import train_and_evaluate  # after Phase B
    except ImportError:
        from model import train_and_evaluate  # Phase A layout
    return train_and_evaluate


def run(repo: Path) -> dict:
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    train_and_evaluate = _train_and_evaluate()
    work = Path(tempfile.mkdtemp(prefix="golden_"))
    os.chdir(work)
    overrides = dict(OVERRIDES, MODEL_PATH=str(work / "w.h5"), SCALER_PATH=str(work / "s.joblib"))
    res = train_and_evaluate(csv_path=str(repo / "binance_btcusdt_1min_ccxt.csv"),
                             config_overrides=overrides, epochs=EPOCHS, force=True,
                             calibrate=True, fit_calibration=True)
    import numpy as np
    out = {}
    hist = res.history.history
    for k in sorted(hist):
        out[f"hist/{k}"] = np.asarray(hist[k], dtype=np.float64)
    for kind in ("delta", "direction_prob", "variance"):
        for h, v in res.predictions[kind].items():
            out[f"pred/{kind}/{h}"] = np.asarray(v, dtype=np.float64)
    for k, v in sorted((res.calibration_lambdas or {}).items()):
        out[f"calib_lambda/{k}"] = np.asarray([v], dtype=np.float64)
    layer = res.model._indicator_layer
    for k, v in sorted(layer.get_learned_parameters().items()):
        out[f"period/{k}"] = np.asarray([v], dtype=np.float64)
    # NT-124: the fitted temperature per horizon and the calibrated P(up) on the test block.
    pipe = res.calibration_pipeline
    if pipe is not None:
        for h, t in sorted(pipe.temperature_scaler.temperatures.items()):
            out[f"temperature/{h}"] = np.asarray([t], dtype=np.float64)
    for h, v in sorted(((res.predictions_calibrated or {}).get("direction_prob") or {}).items()):
        out[f"calibrated/direction_prob/{h}"] = np.asarray(v, dtype=np.float64)
    for h, rep in sorted((res.calibration_report or {}).items()):
        out[f"coverage/{h}"] = np.asarray([rep["coverage90"], rep["width90"]], dtype=np.float64)
    return out


def _compare(a, b, atol: float, rtol: float):
    """(excess, message) for one array pair; excess > 0 means FAIL, inf for any non-finite or shape mismatch.

    Equal: same shape, and at every position either both NaN, or both the same infinity, or both finite
    within atol + rtol * |recorded|. A NaN or an infinity on one side only is a failure, never ignored.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), f"shape {a.shape} vs {b.shape}"
    same = (np.isnan(a) & np.isnan(b)) | ((np.isinf(a) | np.isinf(b)) & (a == b))
    fin = np.isfinite(a) & np.isfinite(b)
    bad_nonfinite = ~(same | fin)
    diff = np.zeros(a.shape)
    excess = np.zeros(a.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        diff[fin] = np.abs(a[fin] - b[fin])
        excess[fin] = diff[fin] - (atol + rtol * np.abs(a[fin]))
    n_nf = int(bad_nonfinite.sum())
    worst = float(excess.max()) if excess.size else 0.0
    if n_nf:
        return float("inf"), f"{n_nf} non-finite mismatch(es), max|diff|(finite)={diff.max() if diff.size else 0:.3g}"
    return worst, f"max|diff|={diff.max() if diff.size else 0:.3g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "verify"])
    ap.add_argument("path")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--skip", action="append", default=[], metavar="PREFIX",
                    help="ignore arrays whose key starts with PREFIX (e.g. hist/ after a logging change)")
    args = ap.parse_args(argv)
    path = Path(args.path).resolve()
    out = run(Path(args.repo).resolve())

    import numpy as np
    if args.mode == "record":
        np.savez_compressed(path, **out)
        print(f"[golden] recorded {len(out)} arrays -> {path}")
        return 0

    ref = {k: v for k, v in np.load(path).items() if not any(k.startswith(p) for p in args.skip)}
    out = {k: v for k, v in out.items() if not any(k.startswith(p) for p in args.skip)}
    missing = sorted(set(ref) - set(out))
    extra = sorted(set(out) - set(ref))
    worst = []
    for k in sorted(set(ref) & set(out)):
        excess, msg = _compare(ref[k], out[k], args.atol, args.rtol)
        worst.append((excess, k, msg))
    worst.sort(key=lambda w: (-w[0], w[1]))
    bad = [w for w in worst if w[0] > 0]
    print(f"[golden] compared {len(worst)} arrays; missing={missing} extra={extra}")
    for excess, k, msg in worst[:12]:
        print(f"   {'FAIL' if excess > 0 else 'ok  '} {k:45s} {msg}")
    by_prefix = {}
    for excess, k, _ in worst:
        pre = k.split("/")[0]
        by_prefix.setdefault(pre, [0, 0])[0 if excess <= 0 else 1] += 1
    print("   by prefix (ok, fail): " + ", ".join(f"{p}={v[0]}/{v[1]}" for p, v in sorted(by_prefix.items())))
    ok = not bad and not missing
    print(json.dumps({"golden_equal": ok, "n_fail": len(bad), "n_missing": len(missing)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
