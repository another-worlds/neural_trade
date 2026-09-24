"""CalibrationExplorer: refit the calibration pipeline with other options and compare on test.

The run's saved pipeline was fit on its calibration block. Here the same raw calibration-block
predictions are refit with any conformal scale, delta shrinkage on/off and miscoverage level,
then applied to the raw test predictions: per-horizon temperature, delta scale (beta), test
coverage, mean interval width and served-delta EV, next to the same numbers for the run's saved
(served) pipeline, the in-sample fit on the calibration block and each block's up-rate, plus the
reliability diagram and the coverage over time (with the saved pipeline drawn dash-dotted when the
refit settings differ from it). Nothing is written back to the run.

    ex = CalibrationExplorer.from_run("runs/<id>", csv_path=...)
    display(ex.widget())
    table = ex.refit("sigma", shrink_delta=False)          # programmatic use
    ex.comparison_table()                                   # refit and saved, per horizon
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from neural_trade.evaluation.frame import HORIZONS
from neural_trade.notebook._display import show

# load_run_blocks serves the run with Predictor.predict's default miscoverage (90% intervals)
SAVED_ALPHA = 0.1


def _ev(y, d) -> float:
    """Explained variance of the realised deltas ``y`` by the predicted ``d``."""
    return float(1 - np.var(y - d) / np.var(y))


def _inside(y, lo, hi) -> np.ndarray:
    return (np.asarray(y) >= np.asarray(lo)) & (np.asarray(y) <= np.asarray(hi))


class CalibrationExplorer:
    def __init__(self, blocks: Dict[str, Any]):
        self.blocks = blocks
        self.config = blocks["config"]
        self.pred_scale = float(blocks["predictor"].bundle.pred_scale)
        self.saved = blocks["predictor"].bundle.calibration_pipeline
        self.pipeline = None
        self.last_table = None
        self._alpha = SAVED_ALPHA
        self._settings: Optional[Tuple[str, bool, float]] = None
        self._out = self._out_cal = None
        self._w = None
        self.saved_table = self._saved_scores()

    @classmethod
    def from_run(cls, run_dir, csv_path: Optional[str] = None) -> "CalibrationExplorer":
        from neural_trade.notebook.backtest_ui import load_run_blocks

        return cls(load_run_blocks(run_dir, csv_path))

    @staticmethod
    def _preds(frame):
        return {"delta": frame.delta, "direction_prob": frame.direction_prob, "variance": frame.variance_scaled}

    # ------------------------------------------------------------------ scoring
    def saved_settings(self) -> Optional[Tuple[str, bool, float]]:
        """(conformal scale, delta shrinkage, alpha) of the run's saved pipeline, or None without one."""
        if self.saved is None:
            return None
        return (str(getattr(self.saved, "conformal_scale", "realized_vol")),
                bool(getattr(self.saved, "shrink_delta", True)), SAVED_ALPHA)

    def matches_saved(self) -> bool:
        """True when the last refit used the saved pipeline's settings (then it reproduces it)."""
        s = self.saved_settings()
        return s is not None and self._settings is not None and (
            s[0], s[1]) == self._settings[:2] and abs(s[2] - self._settings[2]) < 1e-9

    def _score(self, test_out, cal_out, temperatures, betas, alpha) -> pd.DataFrame:
        """One row per horizon: test scores of a served output, plus its in-sample fit on the cal block."""
        from neural_trade.metrics.direction_labels import direction_labels_np
        from neural_trade.metrics.numpy_metrics import ece_pos

        test, cal = self.blocks["test_raw"], self.blocks["cal_raw"]
        db = float(self.config.DIR_DEADBAND_BPS)
        lab_t = direction_labels_np(test.y, test.last_close, db)
        lab_c = direction_labels_np(cal.y, cal.last_close, db)
        rows = {}
        for i, h in enumerate(HORIZONS):
            y, yc = test.y[:, i], cal.y[:, i]
            lo, hi = test_out["intervals"][h]
            clo, chi = cal_out["intervals"][h]
            (lab, mask), (lc, mc) = lab_t[h], lab_c[h]
            rows[h] = {
                "temperature": float(temperatures.get(h, 1.0)),
                "delta beta": float(betas.get(h, 1.0)),
                "coverage": float(np.mean(_inside(y, lo, hi))),
                "target": 1 - alpha,
                "mean width $": float(np.mean(np.asarray(hi) - np.asarray(lo))),
                "EV raw delta": _ev(y, test.delta[h]),
                "EV served delta": _ev(y, test_out["delta"][h]),
                "ECE raw": ece_pos(lab[mask], test.direction_prob[h][mask]),
                "ECE calibrated": ece_pos(lab[mask], test_out["direction_prob"][h][mask]),
                "coverage cal (in-sample)": float(np.mean(_inside(yc, clo, chi))),
                "ECE calibrated cal (in-sample)": ece_pos(lc[mc], cal_out["direction_prob"][h][mc]),
                "up-rate cal": float(lc[mc].mean()) if mc.any() else float("nan"),
                "up-rate test": float(lab[mask].mean()) if mask.any() else float("nan"),
            }
        return pd.DataFrame(rows).T

    @staticmethod
    def _served(frame) -> Optional[dict]:
        if frame is None or frame.intervals is None or frame.direction_prob_calibrated is None:
            return None
        return {"delta": frame.delta, "direction_prob": frame.direction_prob_calibrated, "intervals": frame.intervals}

    def _saved_scores(self) -> Optional[pd.DataFrame]:
        """The saved pipeline's numbers from the served test/cal frames (no refit), or None."""
        if self.saved is None:
            return None
        test_out, cal_out = self._served(self.blocks.get("test")), self._served(self.blocks.get("cal"))
        if test_out is None or cal_out is None:
            return None
        temps = getattr(getattr(self.saved, "temperature_scaler", None), "temperatures", {}) or {}
        betas = getattr(self.saved, "delta_scale", {}) or {}
        return self._score(test_out, cal_out, temps, betas, SAVED_ALPHA)

    def refit(self, conformal_scale: str = "realized_vol", shrink_delta: bool = True, alpha: float = 0.1) -> pd.DataFrame:
        """Fit on the calibration block, apply to test (and, for the in-sample fit, to cal); one row per horizon."""
        from neural_trade.calibration import CalibrationPipeline

        cal, test = self.blocks["cal_raw"], self.blocks["test_raw"]
        cal_log = logging.getLogger("neural_trade.calibration")   # the fit narrates every step at INFO
        level = cal_log.level
        cal_log.setLevel(logging.WARNING)
        try:
            pipe = CalibrationPipeline(conformal_scale=conformal_scale, shrink_delta=shrink_delta).fit_from_arrays(
                self._preds(cal), cal.y, cal.last_close, deadband_bps=float(self.config.DIR_DEADBAND_BPS),
                conformal_alpha=alpha, windows=cal.X_raw, pred_scale=self.pred_scale,
                horizon_steps=tuple(self.config.HORIZON_STEPS))
            out = pipe.apply(self._preds(test), alpha=alpha, windows=test.X_raw)
            out_cal = pipe.apply(self._preds(cal), alpha=alpha, windows=cal.X_raw)
        finally:
            cal_log.setLevel(level)
        self.last_table = self._score(out, out_cal, pipe.temperature_scaler.temperatures, pipe.delta_scale, alpha)
        self.pipeline, self._out, self._out_cal = pipe, out, out_cal
        self._alpha = float(alpha)
        self._settings = (str(conformal_scale), bool(shrink_delta), float(alpha))
        return self.last_table

    def comparison_table(self) -> pd.DataFrame:
        """The last refit and the saved pipeline side by side: rows (horizon, 'refit' | 'saved')."""
        if self.last_table is None:
            self.refit()
        if self.saved_table is None:
            return self.last_table
        rows = [(h, name, t.loc[h]) for h in HORIZONS for name, t in (("refit", self.last_table),
                                                                      ("saved", self.saved_table))]
        return pd.DataFrame([r for _, _, r in rows],
                            index=pd.MultiIndex.from_tuples([(h, n) for h, n, _ in rows], names=["horizon", "pipeline"]))

    # ------------------------------------------------------------------ figures
    def figures(self, horizon: str = "h1", window: int = 500):
        """(reliability, coverage-over-time) figures for ``horizon`` from the last refit.

        The coverage target follows the refit's miscoverage (1 - alpha). When the refit settings
        differ from the saved pipeline's, its curves are added dash-dotted for comparison.
        """
        from neural_trade.metrics.direction_labels import direction_labels_np
        from neural_trade.visualization import stats as S
        from neural_trade.visualization.calibration_plots import coverage_over_time_figure, reliability_figure
        from neural_trade.visualization.theme import horizon_label

        if self.pipeline is None:
            self.refit()
        test, cal, i = self.blocks["test_raw"], self.blocks["cal_raw"], HORIZONS.index(horizon)
        steps = S.horizon_steps(self.config, horizon)
        db = float(self.config.DIR_DEADBAND_BPS)
        lab, mask = direction_labels_np(test.y, test.last_close, db)[horizon]
        lc, mc = direction_labels_np(cal.y, cal.last_close, db)[horizon]
        served = self._served(self.blocks.get("test"))
        differs = served is not None and self.saved is not None and not self.matches_saved()
        p_cal = self._out["direction_prob"][horizon][mask]
        lo, hi = self._out["intervals"][horizon]
        # the saved pipeline is drawn only where it differs from the refit (alpha moves the intervals,
        # not P(up); the scale moves the intervals; shrinkage moves the delta they are centred on)
        p_saved = s_int = None
        if differs:
            p_s = served["direction_prob"][horizon][mask]
            p_saved = p_s if np.max(np.abs(p_s - p_cal), initial=0.0) > 1e-4 else None
            s_lo, s_hi = served["intervals"][horizon]
            s_int = None if (np.allclose(s_lo, lo, atol=0.5) and np.allclose(s_hi, hi, atol=0.5)) else (s_lo, s_hi)
        name = horizon_label(horizon, self.config)
        temp = float(self.last_table.loc[horizon, "temperature"])
        rel = reliability_figure(
            lab[mask], test.direction_prob[horizon][mask], p_cal,
            horizon=horizon, horizon_steps=steps, ref_rate=float(lc[mc].mean()) if mc.any() else None,
            p_saved=p_saved, title=f"Direction reliability, {name}",
            note=(f"refit temperature {temp:.2f}, fit on the calibration block: a temperature stretches P(up) "
                  "around 0.5 and cannot follow a shift in the up-rate"
                  + ("; open diamonds on a dash-dot line: the saved pipeline" if p_saved is not None else "")))
        target = 1 - self._alpha
        cov = coverage_over_time_figure(
            test.y[:, i], lo, hi, window=window, target=target, horizon=horizon, horizon_steps=steps,
            saved=s_int, saved_target=1 - SAVED_ALPHA,
            title=f"Interval coverage, {name}, test block (refit alpha {self._alpha:.2f})")
        return rel, cov

    # ------------------------------------------------------------------ widget
    def _status(self) -> str:
        s = self.saved_settings()
        if self._settings is None:
            return "<i>not fitted yet</i>"
        scale, shrink, alpha = self._settings
        txt = f"refit: scale={scale}, shrinkage={shrink}, alpha={alpha:.2f} (target {1 - alpha:.2f})"
        if s is None:
            return txt + "; the run has no saved pipeline"
        return txt + ("; = the run's saved settings (reproduces the served pipeline)" if self.matches_saved()
                      else f"; saved: scale={s[0]}, shrinkage={s[1]}, alpha={s[2]:.2f} (drawn dash-dotted)")

    def widget(self):
        import ipywidgets as w

        if self._w is not None:
            return self._w["box"]
        s = self.saved_settings()
        saved_scale, saved_shrink = (s[0], s[1]) if s else ("realized_vol", True)
        scale = w.Dropdown(options=["realized_vol", "sigma", "none"], value=saved_scale, description="interval scale")
        shrink = w.Checkbox(value=saved_shrink, description="delta shrinkage")
        alpha = w.FloatSlider(value=SAVED_ALPHA, min=0.02, max=0.4, step=0.01, description="miscoverage")
        horizon = w.ToggleButtons(options=list(HORIZONS), value="h1")
        refit = w.Button(description="Refit on cal, score on test", icon="refresh", button_style="primary",
                         layout=w.Layout(width="240px"))
        status = w.HTML(f"<i>run's saved settings: scale={saved_scale}, shrinkage={saved_shrink}, "
                        f"alpha={SAVED_ALPHA:.2f}</i>")
        table, rel, cov = w.Output(), w.Output(), w.Output()

        def do_refit(_=None):
            status.value = "<i>fitting...</i>"
            self.refit(scale.value, shrink.value, alpha.value)
            show(table, self.comparison_table().round(4))
            draw()
            status.value = self._status()

        def draw(_=None):
            if self.pipeline is None:
                return
            r, c = self.figures(horizon.value)
            show(rel, r)
            show(cov, c)

        refit.on_click(do_refit)
        horizon.observe(draw, names="value")
        # figures stacked, each the full cell width: two unsized Plotly outputs side by side in an HBox
        # render at Plotly's 700 px default each and overflow the cell
        box = w.VBox([w.HBox([scale, shrink, alpha]), w.HBox([refit, status]), table, horizon, rel, cov])
        self._w = {"box": box, "do_refit": do_refit, "rel": rel, "cov": cov, "table": table}
        return box

    def click_refit(self):
        self.widget()
        self._w["do_refit"]()
