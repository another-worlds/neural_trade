"""NT-114 CPU step-time check: DETERMINISTIC_GRU off vs on.

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/src python runs/experiments/nt114_cpu_check/time_step.py

Per model and per seed: build the model off and on (same weights), run 5 warm-up and 30 timed
forward+backward steps (tf.function, training=True, a scalar of every output head as the loss) at the
production input shape (BATCH x LOOKBACK x 5), off and on alternated within each seed. Writes raw.json
next to this file. CPU only: the GPU sec_per_step is the experimenter's measurement (D-018).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import neural_trade  # noqa: E402,F401
from neural_trade.core.config import Config  # noqa: E402
from neural_trade.models.registry import Models  # noqa: E402

BATCH = int(os.environ.get("NT114_BATCH", 256))
LOOKBACK = int(os.environ.get("NT114_LOOKBACK", 60))
WARMUP, TIMED, SEEDS = 5, 30, (0, 1, 2)


def _step_times(model, x):
    @tf.function
    def step(batch):
        with tf.GradientTape() as tape:
            outs = model(batch, training=True)
            loss = tf.add_n([tf.reduce_mean(tf.square(o)) for o in outs])
        grads = tape.gradient(loss, model.trainable_variables)
        return loss, grads

    for _ in range(WARMUP):
        step(x)
    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        loss, grads = step(x)
        float(loss)  # force completion
        times.append(time.perf_counter() - t0)
    return times


def main() -> int:
    out = {"batch": BATCH, "lookback": LOOKBACK, "warmup": WARMUP, "timed_per_seed": TIMED, "seeds": list(SEEDS)}
    for name in ("gru_attention", "gru_small"):
        samples = {"off": [], "on": []}
        for seed in SEEDS:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            off = Models.build(name, Config(LOOKBACK=LOOKBACK))
            on = Models.build(name, Config(LOOKBACK=LOOKBACK, DETERMINISTIC_GRU=True))
            on.set_weights(off.get_weights())
            x = tf.constant(np.random.default_rng(seed).normal(
                size=(BATCH,) + tuple(off.input_shape[1:])).astype("float32"))
            for label, model in (("off", off), ("on", on)) if seed % 2 == 0 else (("on", on), ("off", off)):
                samples[label] += _step_times(model, x)
        r = {}
        for label, v in samples.items():
            r[f"{label}_mean_s"] = statistics.fmean(v)
            r[f"{label}_median_s"] = statistics.median(v)
        r["ratio_on_over_off_mean"] = r["on_mean_s"] / r["off_mean_s"]
        r["ratio_on_over_off_median"] = r["on_median_s"] / r["off_median_s"]
        r["n_steps"] = len(samples["off"])
        out[name] = r
        print(name, json.dumps(r), file=sys.stderr)
    Path(__file__).with_name("raw.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
