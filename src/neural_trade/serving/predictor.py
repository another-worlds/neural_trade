"""Predictor: raw bars in, per-horizon forecasts out (plan section B5).

    p = Predictor.from_artifacts("runs/<id>/artifacts")
    p.predict_last(close_series_or_df)   # the newest window -> one Prediction
    p.predict(windows)                   # raw input windows -> PredictionBatch:
                                         #   [N, LOOKBACK] close windows (INPUT_SERIES=['close'])
                                         #   [N, LOOKBACK, C] OHLCV windows otherwise (NT-047)
    p.predict_frame(ohlcv_dataframe)     # every complete window of a raw frame -> DataFrame

For each horizon: the forecast price change in dollars, P(up) from the direction head (raw and
temperature-calibrated), the predicted sigma in dollars, the Gaussian readout P(up | move
leaves the deadband), and the conformal interval for the price change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from neural_trade.metrics.direction_labels import gaussian_up_prob_given_move_np
from neural_trade.serving.postprocess import heads_to_predictions
from neural_trade.training.artifacts import ArtifactBundle

HORIZONS = ("h0", "h1", "h2")


@dataclass
class PredictionBatch:
    """Arrays per horizon, each of length N (one entry per input window)."""

    delta: Dict[str, np.ndarray]
    direction_prob: Dict[str, np.ndarray]
    direction_prob_calibrated: Dict[str, np.ndarray]
    sigma: Dict[str, np.ndarray]
    variance_scaled: Dict[str, np.ndarray]
    gauss_up_prob: Dict[str, np.ndarray]
    interval: Dict[str, tuple]
    last_close: np.ndarray
    horizon_steps: tuple
    # the raw price heads before the calibration's delta shrink (D-051, NT-119); None on legacy batches
    delta_raw: Optional[Dict[str, np.ndarray]] = None
    # {h: "ok" | "none"} from the calibration pipeline (D-066: "none" = the temperature fit ended at a bound);
    # None when no calibration was applied
    direction_signal: Optional[Dict[str, str]] = None

    def as_predictions_dict(self) -> dict:
        """The TrainResult.predictions layout (raw heads), for evaluation and calibration code."""
        return {"delta": self.delta, "direction_prob": self.direction_prob, "variance": self.variance_scaled}

    def to_prediction_frame(self, pred_scale: float, pred_mean: float = 0.0, y=None, split: str = "live"):
        """An evaluation.PredictionFrame (``y`` = realised deltas, NaN when unknown) for strategies."""
        from neural_trade.evaluation.frame import PredictionFrame

        n = len(self.last_close)
        y = np.full((n, 3), np.nan) if y is None else np.asarray(y, float)
        frame = PredictionFrame(y, self.last_close, self.delta, self.direction_prob, self.variance_scaled,
                                pred_scale, pred_mean, tuple(self.horizon_steps), split,
                                self.direction_prob_calibrated, self.interval)
        # D-051: strategies read the coherence flags from the raw heads, the same meta key
        # PredictionFrame.from_result sets on the training/evaluation path
        if self.delta_raw is not None:
            frame.meta["delta_raw"] = {h: np.asarray(self.delta_raw[h], float).reshape(-1)[:n] for h in self.delta_raw}
        if self.direction_signal:
            frame.meta["direction_signal"] = dict(self.direction_signal)
        return frame

    def to_frame(self, index=None) -> pd.DataFrame:
        cols = {"last_close": self.last_close}
        for h, _steps in zip(HORIZONS, self.horizon_steps):
            cols[f"{h}_delta"] = self.delta[h]
            cols[f"{h}_price"] = self.last_close + self.delta[h]
            cols[f"{h}_p_up"] = self.direction_prob[h]
            cols[f"{h}_p_up_calibrated"] = self.direction_prob_calibrated[h]
            cols[f"{h}_gauss_p_up"] = self.gauss_up_prob[h]
            cols[f"{h}_sigma"] = self.sigma[h]
            lo, hi = self.interval[h]
            cols[f"{h}_lo90"] = self.last_close + lo
            cols[f"{h}_hi90"] = self.last_close + hi
            if self.direction_signal:
                # D-066 / NT-204: "none" = no usable direction signal; the calibrated P(up) above is flat
                cols[f"{h}_direction_signal"] = self.direction_signal.get(h, "ok")
        return pd.DataFrame(cols, index=index)


def _tail_batch(b: PredictionBatch) -> PredictionBatch:
    """The last window's PredictionBatch (predict_last on a DataFrame)."""
    def take(d):
        return {h: v[-1:] for h, v in d.items()}

    iv = {h: (lo[-1:], hi[-1:]) for h, (lo, hi) in b.interval.items()}
    return PredictionBatch(take(b.delta), take(b.direction_prob),
                           take(b.direction_prob_calibrated), take(b.sigma),
                           take(b.variance_scaled), take(b.gauss_up_prob), iv,
                           b.last_close[-1:], b.horizon_steps,
                           None if b.delta_raw is None else take(b.delta_raw), b.direction_signal)


class Predictor:
    def __init__(self, bundle: ArtifactBundle, model=None):
        self.bundle = bundle
        self.config = bundle.config
        self.model = model if model is not None else bundle.build_model()

    @classmethod
    def from_artifacts(cls, directory) -> "Predictor":
        return cls(ArtifactBundle.load(directory))

    # ------------------------------------------------------------------ core
    def predict(self, windows, last_close: Optional[np.ndarray] = None, alpha: float = 0.1,
                batch_size: Optional[int] = None, calibrated: bool = True) -> PredictionBatch:
        """Forecast from raw input windows: ``[N, LOOKBACK]`` close windows when the bundle's
        ``Config.INPUT_SERIES`` is close-only, otherwise ``[N, LOOKBACK, len(INPUT_SERIES)]``
        raw OHLCV windows (NT-047; one window may drop its leading batch axis). Each window
        ends at the decision bar; the default ``last_close`` is the close channel's last bar.

        ``calibrated=False`` returns the raw heads (no temperature, delta shrinkage or intervals),
        e.g. to refit the calibration pipeline."""
        import tensorflow as tf

        series = tuple(getattr(self.config, "INPUT_SERIES", None) or ["close"])
        X = np.asarray(windows, dtype="float32")
        want_rank = 2 if len(series) == 1 else 3
        if X.ndim == want_rank - 1:
            X = X[None, ...]
        expect = (self.config.LOOKBACK,) if len(series) == 1 else (self.config.LOOKBACK, len(series))
        if X.ndim != want_rank or X.shape[1:] != expect:
            raise ValueError(f"windows must be [N, {', '.join(str(d) for d in expect)}] for "
                             f"INPUT_SERIES {list(series)}, got {X.shape}")
        X_close = X if X.ndim == 2 else np.ascontiguousarray(X[..., series.index("close")])
        lc = np.asarray(X_close[:, -1] if last_close is None else last_close,
                        dtype="float32").reshape(-1)
        Xn = self.bundle.normalizer.transform(X, lc)
        # the Grappler arithmetic rewrite the bundle was trained and evaluated with (NT-047)
        from neural_trade.utils.seeding import set_arithmetic_rewrite

        set_arithmetic_rewrite(self.config)
        # Same batch size as training by default: GEMM tiling differs by batch shape, and matching it
        # makes served predictions bit-identical to the ones training reported.
        bs = int(batch_size or self.config.BATCH_SIZE)
        heads = self.model.predict(tf.data.Dataset.from_tensor_slices(Xn).batch(bs), verbose=0)
        preds = heads_to_predictions(heads, len(Xn), self.bundle.pred_scale, self.bundle.pred_mean, self.config)

        raw_delta = {h: preds["delta"][h] for h in HORIZONS}
        prob_cal = {h: preds["direction_prob"][h] for h in HORIZONS}
        intervals = {h: (np.full(len(Xn), np.nan), np.full(len(Xn), np.nan)) for h in HORIZONS}
        signal = None
        if calibrated and self.bundle.calibration_pipeline is not None:
            # conformal realized vol reads raw CLOSE windows in every input mode
            cal = self.bundle.calibration_pipeline.apply(preds, alpha=alpha, windows=X_close)
            prob_cal, intervals = cal["direction_prob"], cal["intervals"]
            signal = self.bundle.calibration_pipeline.direction_signal()
            preds = dict(preds, delta=cal["delta"])  # delta shrinkage (identity when not fitted)
        sigma = {h: np.sqrt(preds["variance"][h]) * self.bundle.pred_scale for h in HORIZONS}
        gauss = {h: gaussian_up_prob_given_move_np(preds["delta"][h], preds["variance"][h], lc,
                                                   self.config.DIR_DEADBAND_BPS, self.bundle.pred_scale)
                 for h in HORIZONS}
        return PredictionBatch(preds["delta"], preds["direction_prob"], prob_cal, sigma, preds["variance"],
                               gauss, intervals, lc, tuple(self.config.HORIZON_STEPS), raw_delta, signal)

    def predict_last(self, close, alpha: float = 0.1) -> Dict[str, dict]:
        """Forecast from the newest complete window; one dict per horizon.

        ``close`` is the raw close series (close-only bundles), or - with a multi-series
        ``Config.INPUT_SERIES`` (NT-047) - a raw OHLCV DataFrame, from which the newest
        window over every configured series is taken (a bare close series cannot fill an
        OHLCV window)."""
        if isinstance(close, pd.DataFrame):
            batch, _df, _anchors = self.predict_windows_frame(close, alpha=alpha)
            batch = _tail_batch(batch)
        else:
            if len(self.config.input_series()) > 1:
                raise ValueError("this bundle's INPUT_SERIES needs OHLCV input: pass the raw "
                                 "OHLCV DataFrame to predict_last (or use predict_frame)")
            close = np.asarray(close, dtype="float32").reshape(-1)
            batch = self.predict(close[-self.config.LOOKBACK:][None, :], alpha=alpha)
        row = batch.to_frame().iloc[0]
        # "direction_signal" (D-066, NT-204): "ok" | "none" (no usable direction signal: the calibrated P(up)
        # is flat) | "n/a" when no calibration pipeline was applied
        sig = batch.direction_signal
        return {h: {k.split("_", 1)[1]: float(v) for k, v in row.items()
                    if k.startswith(h + "_") and not k.endswith("_direction_signal")} | {
            "horizon_bars": int(steps), "last_close": float(batch.last_close[0]),
            "direction_signal": str(sig.get(h, "ok")) if sig else "n/a"}
            for h, steps in zip(HORIZONS, batch.horizon_steps)}

    def predict_windows_frame(self, frame: pd.DataFrame, alpha: float = 0.1, batch_size: Optional[int] = None):
        """``(PredictionBatch, preprocessed df, anchor rows)`` for every complete window of a raw frame;
        anchor row i of the df is the last bar of window i (for Bars.from_frame)."""
        from neural_trade.data.loaders import validate_ohlcv_frame
        from neural_trade.data.windowing import first_anchor, make_inference_input_windows
        from neural_trade.registries.preprocessors import run_preprocessors

        df = validate_ohlcv_frame(run_preprocessors(frame.copy(), self.config),
                                  bar_minutes=self.config.RESAMPLE_MINUTES)
        X, lc, _ = make_inference_input_windows(self.config, df)
        start = first_anchor(self.config.LOOKBACK, self.config.EXTENDED_TREND_PERIODS)
        return self.predict(X, lc, alpha=alpha, batch_size=batch_size), df, np.arange(start - 1, len(df))

    def predict_frame(self, frame: pd.DataFrame, alpha: float = 0.1, batch_size: Optional[int] = None) -> pd.DataFrame:
        """``batch_size``: None = the training batch (bit-identical to training); larger (e.g. 4096)
        is much faster on a GPU for bulk scoring and differs only at float32 round-off."""
        """Preprocess a raw OHLCV frame (Config.PREPROCESSORS) and forecast every complete window."""
        from neural_trade.data.loaders import validate_ohlcv_frame
        from neural_trade.data.windowing import first_anchor, make_inference_input_windows
        from neural_trade.registries.preprocessors import run_preprocessors

        df = validate_ohlcv_frame(run_preprocessors(frame.copy(), self.config),
                                  bar_minutes=self.config.RESAMPLE_MINUTES)
        X, lc, _ = make_inference_input_windows(self.config, df)
        start = first_anchor(self.config.LOOKBACK, self.config.EXTENDED_TREND_PERIODS)
        index = pd.DatetimeIndex(df["timestamp"].iloc[start - 1:].to_numpy(), name="timestamp")
        return self.predict(X, lc, alpha=alpha, batch_size=batch_size).to_frame(index=index)
