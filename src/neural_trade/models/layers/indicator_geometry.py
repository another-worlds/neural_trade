"""IndicatorGeometry: technical-analysis geometry of the learned-indicator sequence (tactical, exploratory).

Input ``ind_seq`` ``[B, L, C]`` (the LearnableIndicators output; the raw close is the LAST channel).
Output ``[B, F]`` with ``F = C * (len(slope_bars) + 3)``, layer-normalised. Per channel x, at the window's
end t (everything reads only bars inside the window, so it is causal; every op is differentiable, so the
gradient reaches the indicator periods):

* slope_k      = (x_t - x_{t-k}) / (k * s_x) for each k in ``slope_bars``; s_x = the channel's own 1-bar
                 change volatility over the window (unit-free for every channel kind);
* distance     = asinh((close_t - x_t) / s_close), s_close = the close's 1-bar change volatility;
* cross        = (tanh(d_t) - tanh(d_{t-5})) / 2 with d_j = (close_j - x_j) / s_close: a smooth "price crossed
                 the channel in the last 5 bars" signal in (-1, 1) (+ = crossed from below to above);
* squeeze      = std(x over the last 10 bars) / std(x over the window).

Volatilities use sqrt(var + eps) so a constant channel has a finite gradient.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

_EPS = 1e-8


def _std(t, axis):
    """Population std with a finite gradient at zero variance."""
    return tf.sqrt(tf.math.reduce_variance(t, axis=axis) + _EPS)


class IndicatorGeometry(layers.Layer):
    def __init__(self, slope_bars=(3, 10), cross_bars: int = 5, squeeze_bars: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.slope_bars = tuple(int(k) for k in slope_bars)
        self.cross_bars = int(cross_bars)
        self.squeeze_bars = int(squeeze_bars)
        if not self.slope_bars or min(self.slope_bars) < 1:
            raise ValueError(f"slope_bars must be positive ints, got {slope_bars}")
        self.norm = layers.LayerNormalization(name="geometry_norm")

    @property
    def n_per_channel(self) -> int:
        return len(self.slope_bars) + 3

    def feature_names(self):
        return ([f"slope_{k}" for k in self.slope_bars] + ["distance", f"cross_{self.cross_bars}",
                f"squeeze_{self.squeeze_bars}"])

    def build(self, input_shape):
        need = max(max(self.slope_bars), self.cross_bars, self.squeeze_bars) + 1
        if input_shape[1] is not None and int(input_shape[1]) < need:
            raise ValueError(f"IndicatorGeometry needs a window of at least {need} bars, got {input_shape[1]}")
        super().build(input_shape)

    def call(self, ind_seq):
        x = tf.cast(ind_seq, tf.float32)                                   # [B, L, C]
        close = x[:, :, -1:]                                               # [B, L, 1]
        s_x = _std(x[:, 1:] - x[:, :-1], axis=1)                           # [B, C]
        s_c = _std(close[:, 1:] - close[:, :-1], axis=1)                   # [B, 1]
        last = x[:, -1]                                                    # [B, C]
        feats = [(last - x[:, -1 - k]) / (float(k) * s_x) for k in self.slope_bars]
        dist = (close - x) / tf.expand_dims(s_c, 1)                        # [B, L, C]
        feats.append(tf.math.asinh(dist[:, -1]))
        sg = tf.tanh(dist)
        feats.append((sg[:, -1] - sg[:, -1 - self.cross_bars]) / 2.0)
        feats.append(_std(x[:, -self.squeeze_bars:], axis=1) / _std(x, axis=1))
        return self.norm(tf.concat(feats, axis=-1))                        # [B, C * (len(slopes) + 3)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"slope_bars": list(self.slope_bars), "cross_bars": self.cross_bars,
                    "squeeze_bars": self.squeeze_bars})
        return cfg
