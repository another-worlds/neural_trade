"""Predictor: raw closes in, per-horizon forecasts out (plan section B5).

    p = Predictor.from_artifacts("runs/<id>/artifacts")
    p.predict_last(close_series)        # the newest window -> one Prediction
    p.predict(windows)                   # [N, LOOKBACK] raw close windows -> PredictionBatch
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

    def as_predictions_dict(self) -> dict:
        """The TrainResult.predictions layout (raw heads), for evaluation and calibration code."""
        return {"delta": self.delta, "direction_prob": self.direction_prob, "variance": self.variance_scaled}

    def to_prediction_frame(self, pred_scale: float, pred_mean: float = 0.0, y=None, split: str = "live"):
        """An evaluation.PredictionFrame (``y`` = realised deltas, NaN when unknown) for strategies."""
        from neural_trade.evaluation.frame import PredictionFrame

        n = len(self.last_close)
        y = np.full((n, 3), np.nan) if y is None else np.asarray(y, float)
        return PredictionFrame(y, self.last_close, self.delta, self.direction_prob, self.variance_scaled,
                               pred_scale, pred_mean, tuple(self.horizon_steps), split,
                               self.direction_prob_calibrated, self.interval)

    def to_frame(self, index=None) -> pd.DataFrame:
        cols = {"last_close": self.last_close}
        for h, steps in zip(HORIZONS, self.horizon_steps):
            cols[f"{h}_delta"] = self.delta[h]
            cols[f"{h}_price"] = self.last_close + self.delta[h]
            cols[f"{h}_p_up"] = self.direction_prob[h]
            cols[f"{h}_p_up_calibrated"] = self.direction_prob_calibrated[h]
            cols[f"{h}_gauss_p_up"] = self.gauss_up_prob[h]
            cols[f"{h}_sigma"] = self.sigma[h]
            lo, hi = self.interval[h]
            cols[f"{h}_lo90"] = self.last_close + lo
            cols[f"{h}_hi90"] = self.last_close + hi
        return pd.DataFrame(cols, index=index)


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
                batch_size: Optional[int] = None) -> PredictionBatch:
        """Forecast from raw close windows ``[N, LOOKBACK]`` (the last column is the last close)."""
        import tensorflow as tf

        X = np.asarray(windows, dtype="float32")
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.config.LOOKBACK:
            raise ValueError(f"windows must have {self.config.LOOKBACK} bars, got {X.shape[1]}")
        lc = np.asarray(X[:, -1] if last_close is None else last_close, dtype="float32").reshape(-1)
        Xn = self.bundle.normalizer.transform(X, lc)
        # Same batch size as training by default: GEMM tiling differs by batch shape, and matching it
        # makes served predictions bit-identical to the ones training reported.
        bs = int(batch_size or self.config.BATCH_SIZE)
        heads = self.model.predict(tf.data.Dataset.from_tensor_slices(Xn).batch(bs), verbose=0)
        preds = heads_to_predictions(heads, len(Xn), self.bundle.pred_scale, self.bundle.pred_mean, self.config)

        calibrated = {h: preds["direction_prob"][h] for h in HORIZONS}
        intervals = {h: (np.full(len(Xn), np.nan), np.full(len(Xn), np.nan)) for h in HORIZONS}
        if self.bundle.calibration_pipeline is not None:
            cal = self.bundle.calibration_pipeline.apply(preds, alpha=alpha, windows=X)
            calibrated, intervals = cal["direction_prob"], cal["intervals"]
        sigma = {h: np.sqrt(preds["variance"][h]) * self.bundle.pred_scale for h in HORIZONS}
        gauss = {h: gaussian_up_prob_given_move_np(preds["delta"][h], preds["variance"][h], lc,
                                                   self.config.DIR_DEADBAND_BPS, self.bundle.pred_scale)
                 for h in HORIZONS}
        return PredictionBatch(preds["delta"], preds["direction_prob"], calibrated, sigma, preds["variance"],
                               gauss, intervals, lc, tuple(self.config.HORIZON_STEPS))

    def predict_last(self, close, alpha: float = 0.1) -> Dict[str, dict]:
        """Forecast from the newest ``LOOKBACK`` closes of ``close``; one dict per horizon."""
        close = np.asarray(close, dtype="float32").reshape(-1)
        batch = self.predict(close[-self.config.LOOKBACK:][None, :], alpha=alpha)
        row = batch.to_frame().iloc[0]
        return {h: {k.split("_", 1)[1]: float(v) for k, v in row.items() if k.startswith(h)} | {
            "horizon_bars": int(steps), "last_close": float(batch.last_close[0])}
            for h, steps in zip(HORIZONS, batch.horizon_steps)}

    def predict_windows_frame(self, frame: pd.DataFrame, alpha: float = 0.1):
        """``(PredictionBatch, preprocessed df, anchor rows)`` for every complete window of a raw frame;
        anchor row i of the df is the last bar of window i (for Bars.from_frame)."""
        from neural_trade.data.loaders import validate_ohlcv_frame
        from neural_trade.data.windowing import make_inference_windows
        from neural_trade.registries.preprocessors import run_preprocessors

        df = validate_ohlcv_frame(run_preprocessors(frame.copy(), self.config))
        close = df["Close"].to_numpy(dtype="float32")
        X, lc, _ = make_inference_windows(close, self.config.LOOKBACK,
                                          extended_trend_periods=self.config.EXTENDED_TREND_PERIODS)
        start = int(max([self.config.LOOKBACK] + list(self.config.EXTENDED_TREND_PERIODS)))
        return self.predict(X, lc, alpha=alpha), df, np.arange(start - 1, len(close))

    def predict_frame(self, frame: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
        """Preprocess a raw OHLCV frame (Config.PREPROCESSORS) and forecast every complete window."""
        from neural_trade.data.loaders import validate_ohlcv_frame
        from neural_trade.data.windowing import make_inference_windows
        from neural_trade.registries.preprocessors import run_preprocessors

        df = validate_ohlcv_frame(run_preprocessors(frame.copy(), self.config))
        close = df["Close"].to_numpy(dtype="float32")
        X, lc, _ = make_inference_windows(close, self.config.LOOKBACK,
                                          extended_trend_periods=self.config.EXTENDED_TREND_PERIODS)
        start = int(max([self.config.LOOKBACK] + list(self.config.EXTENDED_TREND_PERIODS)))
        index = pd.DatetimeIndex(df["timestamp"].iloc[start - 1:].to_numpy(), name="timestamp")
        return self.predict(X, lc, alpha=alpha).to_frame(index=index)
