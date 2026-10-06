"""NT-035 Part B harness: one training run, with an explicit opt-in for the deterministic mode.

Mirrors ``neural_trade.cli.cmd_train`` exactly (same Config load, RunContext, train_and_evaluate
call), because ``neural-trade train`` has no switch for ``seed_everything(seed, deterministic=True)``
(neither ``train_and_evaluate`` nor the CLI exposes a ``deterministic`` kwarg: trainer.py always calls
``seed_everything(cfg.SEED)`` with the default ``deterministic=False``). This script adds exactly one
extra call, ``seed_everything(seed, deterministic=True)``, made *before* ``train_and_evaluate`` for the
"det_on" arm; TensorFlow's ``enable_op_determinism()`` is a one-way, process-global switch in TF 2.10,
so trainer.py's own later reseed (without ``deterministic=True``) cannot undo it.

It does not touch src/ or tests/: it only imports and calls existing public functions
(neural_trade.core.config.Config, neural_trade.experiments.run_context.RunContext,
neural_trade.training.trainer.train_and_evaluate, neural_trade.utils.seeding.seed_everything).

Usage (run from the pinned worktree, with PYTHONPATH set to its src/):
    python harness_train_once.py --config configs/default.yaml --fold-index -3 --epochs 3 \
        --seed 777 --deterministic 1 --runs-dir runs/experiments/gpu_measurements_v1/determinism \
        --name det_on_r1
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold-index", type=int, default=-3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deterministic", type=int, choices=[0, 1], required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--override", action="append", default=[],
                     help="KEY=VALUE Config override (YAML-parsed value), repeatable; NT-074 bisection probe")
    ap.add_argument("--force-gru-unroll", type=int, choices=[0, 1], default=0,
                     help="NT-074 probe only: monkeypatch tf.keras.layers.GRU(unroll=True) before model "
                          "build, to force the generic (non-cuDNN-fused) RNN kernel with identical math, "
                          "and see whether the cuDNN GRU kernel is the nondeterminism source. Does not "
                          "touch src/.")
    ap.add_argument("--cpu-only", type=int, choices=[0, 1], default=0,
                     help="NT-074 probe only: CUDA_VISIBLE_DEVICES=-1 (set before TF import) to run the "
                          "identical graph on CPU only, isolating GPU-kernel-specific nondeterminism.")
    args = ap.parse_args()

    if args.cpu_only:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import neural_trade  # noqa: F401 - sets up CUDA DLL PATH and TF_DETERMINISTIC_OPS (setdefault)

    if args.force_gru_unroll:
        import tensorflow as tf
        _OrigGRU = tf.keras.layers.GRU

        class _UnrolledGRU(_OrigGRU):  # NT-074 probe: forces unroll=True, disabling the cuDNN fused path
            def __init__(self, *a, **kw):
                kw["unroll"] = True
                super().__init__(*a, **kw)

        tf.keras.layers.GRU = _UnrolledGRU
        print(json.dumps({"probe": "force_gru_unroll", "patched_GRU": True}), file=sys.stderr)

    from neural_trade.core.config import Config
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.training.trainer import train_and_evaluate
    from neural_trade.utils.seeding import seed_everything
    from neural_trade.utils.env import fingerprint

    # NT-074 GPU check (lead's mid-task note): prove every process imports the PINNED worktree's
    # src/, not the editable install's D:/neural_trade/src (which may be on a different branch/sha).
    print(json.dumps({"neural_trade_file": neural_trade.__file__}), file=sys.stderr)

    print("TF_DETERMINISTIC_OPS env at import time recorded by fingerprint below", file=sys.stderr)
    fp = fingerprint(include_devices=True)
    print(json.dumps({"env_fingerprint": fp}), file=sys.stderr)

    cfg = Config.from_yaml(args.config)
    cfg.override(FOLD_INDEX=args.fold_index, SEED=args.seed)
    if args.override:
        import yaml
        extra = {}
        for kv in args.override:
            k, _, v = kv.partition("=")
            extra[k] = yaml.safe_load(v)
        cfg.override(**extra)
        print(json.dumps({"extra_overrides": extra}), file=sys.stderr)

    if args.deterministic:
        seed_everything(args.seed, deterministic=True)  # D-025's opt-in mode, called once, up front

    from neural_trade.registries import load_all

    load_all(cfg, plugins_dir=getattr(cfg, "PLUGINS_DIR", None), strict=False)
    ctx = RunContext.create(cfg, root=args.runs_dir, seed=args.seed, tags=["nt035", "determinism"],
                             name=args.name)
    t0 = time.perf_counter()
    result = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=args.epochs, force=True,
                                 calibrate=False, fit_calibration=True, save_artifacts=True)
    wall = time.perf_counter() - t0
    print(json.dumps({"run_dir": str(ctx.run_dir), "wall_s": wall, "deterministic": bool(args.deterministic)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
