"""Append-only, never-raising per-epoch telemetry (plan section C3).

One JSON object per epoch in ``metrics.jsonl``: the Keras epoch logs (epoch aggregates, see
CustomTrainModel), the current learning rates, every loss weight and every learned indicator
period, plus wall-clock timing. ``status.json`` is rewritten each epoch with progress,
seconds per step and the number of telemetry errors.

Telemetry must never stop training: every write is wrapped, failures are counted and logged.
The old ParamsLogger rewrote a growing CSV every epoch (O(n^2)) and a CSV held open in Excel
raised PermissionError mid-training.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

_log = logging.getLogger(__name__)


def _plain(v: Any):
    """JSON-safe scalar: numpy/tensor -> float; NaN/inf -> None."""
    try:
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, (np.generic,)):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return float(v)
    except Exception:
        return None


class JsonlEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, out_dir, indicator_layer=None, run_id: Optional[str] = None):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.indicator_layer = indicator_layer
        self.run_id = run_id
        self.n_errors = 0
        self._t_train = self._t_epoch = None
        self._steps = 0

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / "metrics.jsonl"

    @property
    def status_path(self) -> Path:
        return self.out_dir / "status.json"

    def _safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception:  # telemetry never takes training down
            self.n_errors += 1
            _log.warning("JsonlEpochLogger: telemetry write failed", exc_info=True)
            return None

    def on_train_begin(self, logs=None):
        self._t_train = time.time()
        self._safe(self.out_dir.mkdir, 0o777, True, True)

    def on_epoch_begin(self, epoch, logs=None):
        self._t_epoch = time.time()
        self._steps = 0

    def on_train_batch_end(self, batch, logs=None):
        self._steps += 1

    def _record(self, epoch, logs) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"epoch": int(epoch), "run_id": self.run_id, "time": time.time()}
        rec.update({k: _plain(v) for k, v in (logs or {}).items()})
        model = self.model
        for name, opt in (("lr", getattr(model, "optimizer", None)),
                          ("lr_indicator", getattr(model, "indicator_optimizer", None))):
            if opt is not None:
                rec[name] = _plain(tf.keras.backend.get_value(opt.learning_rate))
        if hasattr(model, "get_lambda_values"):
            rec.update({k: _plain(v) for k, v in model.get_lambda_values().items()})
        layer = self.indicator_layer or getattr(model, "_indicator_layer", None)
        if layer is not None and hasattr(layer, "get_learned_parameters"):
            rec.update({f"period/{k}": _plain(v) for k, v in layer.get_learned_parameters().items()})
        dt = time.time() - (self._t_epoch or time.time())
        rec["epoch_seconds"] = dt
        rec["sec_per_step"] = dt / self._steps if self._steps else None
        return rec

    def _write(self, rec):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")

    def _write_status(self, rec, done=False):
        status = {"run_id": self.run_id, "epochs_completed": rec.get("epoch", -1) + 1, "done": done,
                  "sec_per_step": rec.get("sec_per_step"), "n_errors": self.n_errors,
                  "elapsed_seconds": time.time() - (self._t_train or time.time()),
                  "val_loss": rec.get("val_loss")}
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def on_epoch_end(self, epoch, logs=None):
        rec = self._safe(self._record, epoch, logs) or {"epoch": int(epoch)}
        self._safe(self._write, rec)
        self._safe(self._write_status, rec)
        self._last = rec

    def on_train_end(self, logs=None):
        self._safe(self._write_status, getattr(self, "_last", {"epoch": -1}), True)


def read_metrics(path) -> list:
    """Load a metrics.jsonl into a list of dicts (skips malformed lines)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
