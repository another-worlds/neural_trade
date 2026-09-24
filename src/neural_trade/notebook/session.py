"""TrainingSession: train in a background thread with Pause / Resume / Stop and live curves.

Jupyter processes widget events (button clicks) only while no cell is executing, so a training
loop that blocks its cell can never be paused or stopped from a button. The session runs
``train_and_evaluate`` in a thread and returns control to the notebook at once; a Keras callback
checks the pause/stop events after every batch and redraws the curves after every epoch.

    session = TrainingSession(cfg, run_context=ctx, epochs=20)
    display(session.widget())
    session.start()
    ...
    result = session.wait()          # blocks until training (and evaluation) finished

Stop ends training after the current batch; evaluation, calibration and the artifact bundle
still run, so a stopped run is a complete, usable run.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from neural_trade.notebook._display import show

logger = logging.getLogger(__name__)

_QUIET_CALLBACKS = ("tqdm_progress",)   # console progress bars would print into whichever cell is active


class _ThreadLogHandler(logging.Handler):
    """Routes one thread's log records to a widget; other threads are unaffected."""

    def __init__(self, sink, thread_ident_getter):
        super().__init__()
        self._sink, self._ident = sink, thread_ident_getter
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        if record.thread == self._ident():
            try:
                self._sink(self.format(record) + "\n")
            except Exception:  # never let the UI break training
                self.handleError(record)


class _NotThreadFilter(logging.Filter):
    def __init__(self, thread_ident_getter):
        super().__init__()
        self._ident = thread_ident_getter

    def filter(self, record):
        return record.thread != self._ident()


class TrainingSession:
    def __init__(self, config, *, run_context=None, epochs: Optional[int] = None, calibrate: bool = True,
                 fit_calibration: bool = True, save_artifacts: bool = True, extra_callbacks=None,
                 redraw_every_batches: int = 25):
        self.config = config
        self.run_context = run_context
        self.epochs = int(epochs if epochs is not None else config.EPOCHS)
        self.calibrate, self.fit_calibration, self.save_artifacts = calibrate, fit_calibration, save_artifacts
        self.extra_callbacks = list(extra_callbacks or [])
        self.redraw_every = max(1, int(redraw_every_batches))
        self.history: List[Dict[str, float]] = []
        self.batch_points: List[tuple] = []      # (fractional epoch, running-mean train loss)
        self.val_points: List[tuple] = []        # (epoch end, val loss)
        self.status = "ready"
        self.result = None
        self.error: Optional[BaseException] = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._w = None

    # ------------------------------------------------------------------ controls (safe to call from anywhere)
    def pause(self):
        if self.status == "training":
            self._pause.set()
            self._set_status("paused")

    def resume(self):
        if self._pause.is_set():
            self._pause.clear()
            self._set_status("training")

    def stop(self):
        self._stop.set()
        self._pause.clear()
        if self.status in ("training", "paused", "ready"):
            self._set_status("stopping")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "TrainingSession":
        if self.running:
            raise RuntimeError("this session is already training")
        self._thread = threading.Thread(target=self._run, name="neural_trade-training", daemon=True)
        self._thread.start()
        return self

    def wait(self, timeout: Optional[float] = None):
        """Block until the run finished; returns the TrainResult (re-raises a training error)."""
        self._done.wait(timeout)
        if self.error is not None:
            raise self.error
        return self.result

    def _run(self):
        from neural_trade.training.trainer import train_and_evaluate

        pkg_logger = logging.getLogger("neural_trade")
        ident = lambda: self._thread.ident if self._thread else None  # noqa: E731
        handler = _ThreadLogHandler(self._log, ident)
        quiet = _NotThreadFilter(ident)
        pkg_logger.addHandler(handler)
        for h in pkg_logger.handlers:
            if h is not handler:
                h.addFilter(quiet)
        cfg = self.config
        names = [c for c in (getattr(cfg, "CALLBACKS", None) or []) if c not in _QUIET_CALLBACKS]
        try:
            cfg = cfg.copy().override(CALLBACKS=names) if names != list(cfg.CALLBACKS) else cfg
            if self.run_context is not None:
                self.run_context.config.override(CALLBACKS=names)
                cfg = self.run_context.config
            self._set_status("training")
            self.result = train_and_evaluate(
                config=cfg, run_context=self.run_context, epochs=self.epochs, force=True,
                calibrate=self.calibrate, fit_calibration=self.fit_calibration,
                save_artifacts=self.save_artifacts, extra_callbacks=[self._callback()] + self.extra_callbacks)
            self._set_status("stopped early - evaluated" if self._stop.is_set() else "finished")
        except BaseException as exc:  # surfaced by wait()
            self.error = exc
            self._set_status(f"failed: {exc!r}"[:200])
            logger.exception("training failed")
        finally:
            pkg_logger.removeHandler(handler)
            for h in pkg_logger.handlers:
                h.removeFilter(quiet)
            self._done.set()
            self._redraw()

    # ------------------------------------------------------------------ the Keras hook
    def _callback(self):
        import tensorflow as tf

        session = self

        class _SessionCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self._supports_tf_logs = True
                self._t0 = time.time()

            def on_epoch_begin(self, epoch, logs=None):
                self._t0 = time.time()
                session._progress(epoch, 0, (self.params or {}).get("steps"))

            def on_train_batch_end(self, batch, logs=None):
                while session._pause.is_set() and not session._stop.is_set():
                    time.sleep(0.2)
                if session._stop.is_set():
                    self.model.stop_training = True
                if batch % session.redraw_every == 0:
                    steps = (self.params or {}).get("steps")
                    loss = (logs or {}).get("loss")
                    if loss is not None and steps:
                        session.batch_points.append((len(session.history) + (batch + 1) / steps, float(loss)))
                    session._progress(len(session.history), batch + 1, steps)
                    session._redraw_batches()

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": epoch, "seconds": time.time() - self._t0}
                row.update({k: float(v) for k, v in (logs or {}).items()})
                session.history.append(row)
                if "val_loss" in row:
                    session.val_points.append((epoch + 1, row["val_loss"]))
                session._redraw()

        return _SessionCallback()

    def epoch_rows(self) -> List[Dict[str, float]]:
        """Per-epoch rows: the session's Keras logs merged with the run's metrics.jsonl (which adds
        the learning rates, the learned periods and the epoch timing) when the run logs one."""
        rows = [dict(r) for r in self.history]
        path = self.run_context.path("metrics.jsonl") if self.run_context is not None else None
        if path is not None and path.exists():
            from neural_trade.telemetry.epoch_logger import read_metrics

            try:
                logged = {int(r["epoch"]): r for r in read_metrics(path) if "epoch" in r}
            except (OSError, ValueError):
                logged = {}
            rows = [{**r, **logged.get(int(r["epoch"]), {})} for r in rows]
        return rows

    def curves_figure(self):
        """The training dashboard as a plain figure (for a static cell output after training)."""
        from neural_trade.visualization.training_dashboard import training_dashboard_figure

        return training_dashboard_figure(self.epoch_rows(), self._config_for_display())

    def health_html(self) -> str:
        """The training-health tiles (convergence, patience, collapse, gradients ...) as HTML."""
        from neural_trade.visualization.training_dashboard import training_health_html

        return training_health_html(self.epoch_rows(), self._config_for_display())

    def _config_for_display(self):
        cfg = self.config.copy() if hasattr(self.config, "copy") else self.config
        if hasattr(cfg, "override"):
            cfg.override(EPOCHS=self.epochs)
        return cfg

    def history_frame(self):
        """One row per epoch with the headline train/val metrics."""
        import pandas as pd

        cols = ["epoch", "seconds", "loss", "val_loss", "val_dir_mcc_h1", "val_gauss_dir_mcc_h1", "val_pit_ks_h1",
                "val_crps_loss", "val_nll_loss", "nonfinite_grad_steps"]
        df = pd.DataFrame(self.history)
        return df[[c for c in cols if c in df.columns]]

    # ------------------------------------------------------------------ widgets
    def widget(self):
        """Buttons, status, progress, live curves and the log (build once, display anywhere)."""
        import ipywidgets as w

        if self._w is not None:
            return self._w["box"]
        pause = w.Button(description="Pause", icon="pause", layout=w.Layout(width="110px"))
        resume = w.Button(description="Resume", icon="play", layout=w.Layout(width="110px"))
        stop = w.Button(description="Stop", icon="stop", button_style="danger", layout=w.Layout(width="110px"))
        pause.on_click(lambda _: self.pause())
        resume.on_click(lambda _: self.resume())
        stop.on_click(lambda _: self.stop())
        status = w.HTML()
        epochs = w.IntProgress(min=0, max=self.epochs, description="Epochs", layout=w.Layout(width="45%"))
        batches = w.IntProgress(min=0, max=1, description="Batch", layout=w.Layout(width="45%"))
        batch_curve = w.Output(layout=w.Layout(min_height="240px"))
        curves = w.Output(layout=w.Layout(min_height="420px"))
        table = w.HTML()
        log = w.Output(layout=w.Layout(max_height="220px", overflow="auto", border="1px solid #8884"))
        box = w.VBox([w.HBox([pause, resume, stop, status]), w.HBox([epochs, batches]), table, batch_curve, curves,
                      w.Accordion(children=[log], titles=("Log",))])
        self._w = dict(box=box, status=status, epochs=epochs, batches=batches, curves=curves, table=table, log=log,
                       batch_curve=batch_curve, last_batch_draw=0.0)
        self._set_status(self.status)
        return box

    def _set_status(self, text):
        self.status = text
        if self._w is not None:
            color = {"training": "#15803d", "paused": "#b45309", "stopping": "#b91c1c"}.get(text, "#6b7280")
            self._w["status"].value = f"<b style='color:{color};margin-left:12px'>{text}</b>"

    def _log(self, text):
        if self._w is not None:
            self._w["log"].append_stdout(text)

    def _progress(self, epoch, batch, steps):
        if self._w is None:
            return
        self._w["epochs"].value = min(epoch, self.epochs)
        if steps:
            self._w["batches"].max = int(steps)
        self._w["batches"].value = int(batch)

    def _redraw(self):
        if self._w is None or not self.history:
            return
        self._w["epochs"].value = min(len(self.history), self.epochs)
        # state updates: never published to the executing cell (see notebook._display)
        show(self._w["curves"], self.curves_figure())
        self._w["table"].value = self.health_html()
        self._redraw_batches(force=True)

    def _redraw_batches(self, force: bool = False):
        """The batch-by-batch loss strip; at most every 2 s so drawing never slows training."""
        if self._w is None or not self.batch_points:
            return
        now = time.time()
        if not force and now - self._w["last_batch_draw"] < 2.0:
            return
        from neural_trade.visualization.training_dashboard import batch_loss_figure

        self._w["last_batch_draw"] = now
        show(self._w["batch_curve"], batch_loss_figure(self.batch_points, self.val_points))
