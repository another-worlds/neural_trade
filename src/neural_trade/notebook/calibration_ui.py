"""CalibrationExplorer: refit the calibration pipeline with other options and compare on test.

The run's saved pipeline was fit on its calibration block. Here the same raw calibration-block
predictions are refit with any conformal scale, delta shrinkage on/off and miscoverage level,
then applied to the raw test predictions: per-horizon temperature, delta scale (beta), test
coverage, mean interval width and served-delta EV, plus the reliability diagram and rolling
coverage. Nothing is written back to the run.

    ex = CalibrationExplorer.from_run("runs/<id>", csv_path=...)
    display(ex.widget())
    table = ex.refit("sigma", shrink_delta=False)          # programmatic use
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from neural_trade.evaluation.frame import HORIZONS
from neural_trade.notebook._display import show


def _ev(y, d) -> float:
    """Explained variance of the realised deltas ``y`` by the predicted ``d``."""
    return float(1 - np.var(y - d) / np.var(y))


class CalibrationExplorer:
    def __init__(self, blocks: Dict[str, Any]):
        self.blocks = blocks
        self.config = blocks["config"]
        self.pred_scale = float(blocks["predictor"].bundle.pred_scale)
        self.saved = blocks["predictor"].bundle.calibration_pipeline
        self.pipeline = None
        self.last_table = None
        self._w = None

    @classmethod
    def from_run(cls, run_dir, csv_path: Optional[str] = None) -> "CalibrationExplorer":
        from neural_trade.notebook.backtest_ui import load_run_blocks

        return cls(load_run_blocks(run_dir, csv_path))

    @staticmethod
    def _preds(frame):
        return {"delta": frame.delta, "direction_prob": frame.direction_prob, "variance": frame.variance_scaled}

    def refit(self, conformal_scale: str = "realized_vol", shrink_delta: bool = True, alpha: float = 0.1) -> pd.DataFrame:
        """Fit on the calibration block, apply to test; one row per horizon."""
        from neural_trade.calibration import CalibrationPipeline
        from neural_trade.metrics.direction_labels import direction_labels_np
        from neural_trade.metrics.numpy_metrics import ece_pos

        cal, test = self.blocks["cal_raw"], self.blocks["test_raw"]
        cal_log = logging.getLogger("neural_trade.calibration")   # the fit narrates every step at INFO
        level = cal_log.level
        cal_log.setLevel(logging.WARNING)
        try:
            pipe = CalibrationPipeline(conformal_scale=conformal_scale, shrink_delta=shrink_delta).fit_from_arrays(
                self._preds(cal), cal.y, cal.last_close, deadband_bps=float(self.config.DIR_DEADBAND_BPS),
                conformal_alpha=alpha, windows=cal.X_raw, pred_scale=self.pred_scale,
                horizon_steps=tuple(self.config.HORIZON_STEPS))
        finally:
            cal_log.setLevel(level)
        out = pipe.apply(self._preds(test), alpha=alpha, windows=test.X_raw)
        labels = direction_labels_np(test.y, test.last_close, float(self.config.DIR_DEADBAND_BPS))
        rows = {}
        for i, h in enumerate(HORIZONS):
            y = test.y[:, i]
            lo, hi = out["intervals"][h]
            lab, mask = labels[h]
            rows[h] = {
                "temperature": pipe.temperature_scaler.temperatures.get(h, 1.0),
                "delta beta": pipe.delta_scale[h],
                "coverage": float(np.mean((y >= lo) & (y <= hi))),
                "target": 1 - alpha,
                "mean width $": float(np.mean(hi - lo)),
                "EV raw delta": _ev(y, test.delta[h]),
                "EV served delta": _ev(y, out["delta"][h]),
                "ECE raw": ece_pos(lab[mask], test.direction_prob[h][mask]),
                "ECE calibrated": ece_pos(lab[mask], out["direction_prob"][h][mask]),
            }
        self.pipeline, self._out = pipe, out
        self.last_table = pd.DataFrame(rows).T
        return self.last_table

    def figures(self, horizon: str = "h1", window: int = 500):
        """(reliability, coverage-over-time) figures for ``horizon`` from the last refit."""
        from neural_trade.metrics.direction_labels import direction_labels_np
        from neural_trade.visualization.calibration_plots import coverage_over_time_figure, reliability_figure

        if self.pipeline is None:
            self.refit()
        test, i = self.blocks["test_raw"], HORIZONS.index(horizon)
        lab, mask = direction_labels_np(test.y, test.last_close, float(self.config.DIR_DEADBAND_BPS))[horizon]
        rel = reliability_figure(lab[mask], test.direction_prob[horizon][mask],
                                 self._out["direction_prob"][horizon][mask], title=f"Direction reliability {horizon}")
        lo, hi = self._out["intervals"][horizon]
        cov = coverage_over_time_figure(test.y[:, i], lo, hi, window=window,
                                        title=f"{horizon}: interval coverage on the test block")
        return rel, cov

    def widget(self):
        import ipywidgets as w

        if self._w is not None:
            return self._w["box"]
        saved_scale = getattr(self.saved, "conformal_scale", "realized_vol") if self.saved else "realized_vol"
        saved_shrink = bool(getattr(self.saved, "shrink_delta", True)) if self.saved else True
        scale = w.Dropdown(options=["realized_vol", "sigma", "none"], value=saved_scale, description="interval scale")
        shrink = w.Checkbox(value=saved_shrink, description="delta shrinkage")
        alpha = w.FloatSlider(value=0.1, min=0.02, max=0.4, step=0.01, description="miscoverage")
        horizon = w.ToggleButtons(options=list(HORIZONS), value="h1")
        refit = w.Button(description="Refit on cal, score on test", icon="refresh", button_style="primary",
                         layout=w.Layout(width="240px"))
        status = w.HTML(f"<i>run's saved settings: scale={saved_scale}, shrinkage={saved_shrink}</i>")
        table, rel, cov = w.Output(), w.Output(), w.Output()

        def do_refit(_=None):
            status.value = "<i>fitting...</i>"
            frame = self.refit(scale.value, shrink.value, alpha.value)
            show(table, frame.round(4))
            draw()
            status.value = f"refit: scale={scale.value}, shrinkage={shrink.value}, alpha={alpha.value:.2f}"

        def draw(_=None):
            if self.pipeline is None:
                return
            r, c = self.figures(horizon.value)
            show(rel, r)
            show(cov, c)

        refit.on_click(do_refit)
        horizon.observe(draw, names="value")
        box = w.VBox([w.HBox([scale, shrink, alpha]), w.HBox([refit, status]), table, horizon, w.HBox([rel, cov])])
        self._w = {"box": box, "do_refit": do_refit}
        return box

    def click_refit(self):
        self.widget()
        self._w["do_refit"]()
