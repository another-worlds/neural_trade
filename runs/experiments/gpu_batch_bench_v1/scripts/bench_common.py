"""Shared timing callback for the benchmark drivers (new and old code). Not part of the repo."""
import json
import os
import subprocess
import sys
import time


def gpu_peak_mb():
    try:
        import tensorflow as tf
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {k: round(v / 2**20, 1) for k, v in info.items()}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def make_timer(tf):
    class Timer(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epochs = []
            self._t_step = None
            self._steps = []
            self.t_train_begin = None
            self.t_train_end = None

        def on_train_begin(self, logs=None):
            self.t_train_begin = time.time()

        def on_train_end(self, logs=None):
            self.t_train_end = time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self._steps = []
            self._epoch = {"epoch": epoch, "t_begin": time.time()}

        def on_train_batch_begin(self, batch, logs=None):
            self._t_step = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            self._steps.append(time.perf_counter() - self._t_step)

        def on_test_begin(self, logs=None):
            self._epoch["t_val_begin"] = time.time()
            self._epoch["t_train_end"] = time.time()

        def on_test_end(self, logs=None):
            self._epoch["t_val_end"] = time.time()

        def on_epoch_end(self, epoch, logs=None):
            e = self._epoch
            e["t_end"] = time.time()
            steps = self._steps
            e["n_steps"] = len(steps)
            e["train_seconds"] = e.get("t_train_end", e["t_end"]) - e["t_begin"]
            e["val_seconds"] = (e.get("t_val_end", 0) - e.get("t_val_begin", 0)) if "t_val_begin" in e else None
            e["step_sum_seconds"] = sum(steps)
            if len(steps) > 4:
                s = sorted(steps)
                e["step_median"] = s[len(s) // 2]
                e["step_mean_excl_first2"] = sum(steps[2:]) / (len(steps) - 2)
                e["step_max"] = s[-1]
            e["gpu_mem_mb"] = gpu_peak_mb()
            self.epochs.append(e)

    return Timer()


class Dmon:
    """nvidia-smi dmon -s um in the background, one-second samples with timestamps."""

    def __init__(self, path):
        self.path = path
        self.proc = None

    def start(self):
        self.f = open(self.path, "w")
        self.proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-o", "T"], stdout=self.f,
                                     stderr=subprocess.STDOUT)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.f.close()


def summarize(tag, timer, t0, t_pre_train, t_after_train_eval, t_end, extra=None):
    ep = timer.epochs
    steady = [e for e in ep if e["epoch"] >= 1]
    out = {
        "tag": tag,
        "epochs": ep,
        "setup_seconds(before fit)": timer.t_train_begin - t0,
        "fit_seconds": timer.t_train_end - timer.t_train_begin,
        "post_fit_seconds(serve+calibration+predict)": t_after_train_eval - timer.t_train_end,
        "eval_report_seconds": t_end - t_after_train_eval,
        "total_seconds": t_end - t0,
        "steady_sec_per_step(mean of epochs>=2, per-step mean excl first 2)":
            (sum(e["step_mean_excl_first2"] for e in steady) / len(steady)) if steady else None,
        "steady_sec_per_step_median": (sum(e["step_median"] for e in steady) / len(steady)) if steady else None,
        "steady_train_sec_per_epoch": (sum(e["train_seconds"] for e in steady) / len(steady)) if steady else None,
        "steady_val_sec_per_epoch": (sum(e["val_seconds"] or 0 for e in steady) / len(steady)) if steady else None,
        "epoch1_train_seconds(includes tracing)": ep[0]["train_seconds"] if ep else None,
        "n_steps_per_epoch": ep[0]["n_steps"] if ep else None,
        "peak_gpu_mem_mb": max((e["gpu_mem_mb"].get("peak", 0) for e in ep), default=None),
    }
    if extra:
        out.update(extra)
    return out
