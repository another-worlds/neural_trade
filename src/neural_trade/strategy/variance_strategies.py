"""Variance-driven strategies for a model with no directional skill and a small variance edge (NT-077).

Design source: docs/research/2026-09-29-strategy-architectures/README.md, sections 2.0 (which sigma),
3 (the ranked shortlist) and 5 (engine requirements). Every threshold is fitted in ``from_calibration``
on the CALIBRATION block's SignalFrame only (``fit``). ``horizon`` indexes the SignalFrame's horizons;
the default -1 is the longest one (h2 on the reference setup). ``sigma_source`` is ``"model"`` (the
model's predicted sigma) or ``"ewma"`` (the model-free EWMA twin, ``SignalFrame.sigma_ewma``): the
model adds value only where the model row beats its twin. Every strategy waits ``EWMA_WARMUP`` bars,
so model and EWMA rows decide on the same bars. sigma_hat_t = sigma_$[t, h] / close[t].

Exposure mode (``ExposureStrategy``, traded by ``backtest.run_exposure_backtest``):

* ``vol_target``: e_t = min(1, sigma* / sigma_hat_t), sigma* = the calibration median of sigma_hat;
  decided every 60 bars, rebalanced when |e_t - held| > ``band`` (research 2.1, shortlist 1).
* ``net_edge_kelly``: mu_t = mu_gauss[t, h] / close (the Gaussian read-out of calibrated P(up));
  aim = sign(mu) x f x max(|mu| - cost, 0) / sigma_ret[t, h]^2 clipped to [-1, 1]; decided every 20
  bars and traded to the band's edge (research 2.3 / 2.4, shortlist 3).

Discrete mode (today's engine):

* ``edge_over_cost``: long / short when |mu_t| > cost, held ``max_hold`` = 20 bars, no stop
  (research 2.5, the discrete twin of ``net_edge_kelly``).
* ``vol_regime_long``: long while sigma_hat < the calibration q_in quantile, out when it rises above
  the q_out quantile (research 2.2, shortlist 2).
* ``gated_ta``: an MA cross (SMA 20 / 50) or a Bollinger breakout (20-bar SMA +- 2 sd) of the close,
  entered only when sigma_hat >= the calibration q quantile; held at most 60 bars (research 2.7,
  shortlist 4). The indicators are trailing (bars <= t) and not tuned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.strategy.signals import EWMA_WARMUP, SIGMA_SOURCES, SignalFrame
from neural_trade.strategy.strategies import ExposureStrategy, FittedOnCalibration, Strategies, Strategy
from neural_trade.strategy.trades import Order

DEFAULT_COST = 0.0026        # 13 bps per side, as a round-trip return (BacktestConfig's default costs)
PRIMARIES = ("ma_cross", "bollinger")


def _check_source(source: str) -> None:
    if source not in SIGMA_SOURCES:
        raise InvalidConfigurationError(f"sigma_source must be one of {SIGMA_SOURCES}, got {source!r}")


def _check_quantile(name: str, q: float) -> None:
    if not 0.0 < float(q) < 1.0:
        raise InvalidConfigurationError(f"{name} must lie in (0, 1), got {q!r}")


def sigma_hat(s: SignalFrame, source: str, h: int) -> np.ndarray:
    """Horizon ``h``'s sigma in return units per bar: sigma_for(source, h) / close."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return s.sigma_for(source, h) / s.close


def _cal_quantile(cal: SignalFrame, source: str, h: int, q: float) -> float:
    """The q quantile of sigma_hat over the calibration block's finite bars."""
    x = sigma_hat(cal, source, h)
    x = x[np.isfinite(x)]
    if not len(x):
        raise InvalidConfigurationError(f"the calibration block has no finite {source} sigma (it needs more than "
                                        f"{EWMA_WARMUP} bars for the EWMA)")
    return float(np.quantile(x, q))


def _sigma_hat_at(s: SignalFrame, source: str, h: int, t: int) -> float:
    c = s.close[t]
    return float(s.sigma_for(source, h)[t] / c) if c > 0 else float("nan")


def _mu_at(s: SignalFrame, h: int, t: int) -> float:
    """mu_gauss in return units: sigma_ret x Phi^-1(P(up)) at bar t."""
    c = s.close[t]
    return float(s.mu_gauss[t, h] / c) if c > 0 else float("nan")


def _side(sign: float) -> str:
    return "LONG" if sign > 0 else "SHORT"


# ------------------------------------------------------------------ exposure mode
@Strategies.register(name="vol_target", tags=["exposure", "variance", "nt-077"])
@dataclass
class VolTargetStrategy(ExposureStrategy):
    """Volatility-targeted long: e_t = min(max_abs_exposure, sigma* / sigma_hat_t)."""

    name: ClassVar[str] = "vol_target"
    fitted_fields: ClassVar[Tuple[str, ...]] = ("sigma_star",)
    decide_every: int = 60
    band: float = 0.10
    sigma_source: str = "model"
    horizon: int = -1
    sigma_star: float = float("nan")     # fitted: the calibration median of sigma_hat

    def __post_init__(self):
        _check_source(self.sigma_source)

    def warmup(self) -> int:
        return EWMA_WARMUP

    def fit(self, calibration):
        self.sigma_star = _cal_quantile(calibration, self.sigma_source, self.horizon, 0.5)
        return self

    def target(self, s, t, current):
        sh = _sigma_hat_at(s, self.sigma_source, self.horizon, t)
        if not (np.isfinite(sh) and np.isfinite(self.sigma_star)):
            return float("nan")
        if sh <= 0:
            return float(self.max_abs_exposure)
        return float(min(self.max_abs_exposure, self.sigma_star / sh))


@Strategies.register(name="net_edge_kelly", tags=["exposure", "variance", "nt-077"])
@dataclass
class NetEdgeKellyStrategy(ExposureStrategy):
    """Fractional-Kelly aim on the expected move net of costs, traded to the no-trade band's edge."""

    name: ClassVar[str] = "net_edge_kelly"
    decide_every: int = 20
    band: float = 0.10
    trade_to_band_edge: bool = True
    f: float = 0.25                      # the Kelly fraction
    cost: float = DEFAULT_COST           # round-trip cost as a return
    horizon: int = -1

    def warmup(self) -> int:
        return EWMA_WARMUP

    def target(self, s, t, current):
        mu = _mu_at(s, self.horizon, t)
        sr = float(s.sigma_ret[t, self.horizon])
        if not (np.isfinite(mu) and np.isfinite(sr)):
            return float("nan")
        excess = max(abs(mu) - self.cost, 0.0)
        if excess == 0.0 or sr <= 0:
            return 0.0
        aim = np.sign(mu) * self.f * excess / sr ** 2
        return float(np.clip(aim, -1.0, 1.0))


# ------------------------------------------------------------------ discrete mode
@Strategies.register(name="edge_over_cost", tags=["variance", "nt-077"])
@dataclass
class EdgeOverCostStrategy(FittedOnCalibration, Strategy):
    """Long / short when the Gaussian expected move |mu_t| exceeds the round-trip cost; no stop."""

    name: ClassVar[str] = "edge_over_cost"
    cost: float = DEFAULT_COST
    horizon: int = -1
    size: float = 1.0
    max_hold: int = 20

    def warmup(self) -> int:
        return EWMA_WARMUP

    def decide(self, s, t):
        mu = _mu_at(s, self.horizon, t)
        if np.isfinite(mu) and abs(mu) > self.cost:
            return Order(_side(mu), self.size, reason="edge_over_cost", max_hold=self.max_hold,
                         info={"mu": mu})
        return None


@Strategies.register(name="vol_regime_long", tags=["variance", "nt-077"])
@dataclass
class VolRegimeLongStrategy(FittedOnCalibration, Strategy):
    """Long while predicted volatility is low; out when it rises above a higher quantile (hysteresis)."""

    name: ClassVar[str] = "vol_regime_long"
    fitted_fields: ClassVar[Tuple[str, ...]] = ("in_below", "out_above")
    q_out: float = 0.80
    q_in: Optional[float] = None         # None: q_out - 0.10
    sigma_source: str = "model"
    horizon: int = -1
    size: float = 1.0
    max_hold: int = 10 ** 9
    in_below: float = float("nan")       # fitted: the calibration q_in quantile of sigma_hat
    out_above: float = float("nan")      # fitted: the calibration q_out quantile of sigma_hat

    def __post_init__(self):
        _check_source(self.sigma_source)
        if self.q_in is None:
            self.q_in = round(float(self.q_out) - 0.10, 10)
        _check_quantile("q_out", self.q_out)
        _check_quantile("q_in", self.q_in)
        if self.q_in > self.q_out:
            raise InvalidConfigurationError(f"q_in ({self.q_in}) must not exceed q_out ({self.q_out})")

    def warmup(self) -> int:
        return EWMA_WARMUP

    def fit(self, calibration):
        self.in_below = _cal_quantile(calibration, self.sigma_source, self.horizon, self.q_in)
        self.out_above = _cal_quantile(calibration, self.sigma_source, self.horizon, self.q_out)
        return self

    def decide(self, s, t):
        sh = _sigma_hat_at(s, self.sigma_source, self.horizon, t)
        if np.isfinite(sh) and sh < self.in_below:
            return Order("LONG", self.size, reason="low_vol", max_hold=self.max_hold)
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        sh = _sigma_hat_at(s, self.sigma_source, self.horizon, t)
        return "VOL" if np.isfinite(sh) and sh > self.out_above else None


def _sma(close: np.ndarray, t: int, window: int) -> float:
    return float(np.mean(close[t - window + 1: t + 1]))


@Strategies.register(name="gated_ta", tags=["variance", "nt-077"])
@dataclass
class GatedTAStrategy(FittedOnCalibration, Strategy):
    """A textbook TA primary (MA cross or Bollinger breakout), entered only in high predicted volatility."""

    name: ClassVar[str] = "gated_ta"
    fitted_fields: ClassVar[Tuple[str, ...]] = ("gate",)
    primary: str = "ma_cross"            # "ma_cross" or "bollinger"
    q: float = 0.5                       # the gate: sigma_hat >= the calibration q quantile
    sigma_source: str = "model"
    horizon: int = -1
    fast: int = 20                       # ma_cross: fast and slow SMA of the close
    slow: int = 50
    bb_window: int = 20                  # bollinger: SMA +- bb_k population sd of the close
    bb_k: float = 2.0
    size: float = 1.0
    max_hold: int = 60
    gate: float = float("nan")           # fitted: the calibration q quantile of sigma_hat

    def __post_init__(self):
        _check_source(self.sigma_source)
        _check_quantile("q", self.q)
        if self.primary not in PRIMARIES:
            raise InvalidConfigurationError(f"primary must be one of {PRIMARIES}, got {self.primary!r}")
        if not 1 <= self.fast < self.slow:
            raise InvalidConfigurationError(f"need 1 <= fast < slow, got fast={self.fast}, slow={self.slow}")
        if self.bb_window < 2:
            raise InvalidConfigurationError(f"bb_window must be at least 2, got {self.bb_window}")

    def warmup(self) -> int:
        return max(EWMA_WARMUP, self.slow + 1, self.bb_window)

    def fit(self, calibration):
        self.gate = _cal_quantile(calibration, self.sigma_source, self.horizon, self.q)
        return self

    def _cross(self, s, t) -> Tuple[float, float]:
        """fast - slow at bars t - 1 and t (trailing SMAs of the close)."""
        c = s.close
        return (_sma(c, t - 1, self.fast) - _sma(c, t - 1, self.slow), _sma(c, t, self.fast) - _sma(c, t, self.slow))

    def _bands(self, s, t):
        w = s.close[t - self.bb_window + 1: t + 1]
        mid, sd = float(np.mean(w)), float(np.std(w))
        return mid, mid + self.bb_k * sd, mid - self.bb_k * sd

    def decide(self, s, t):
        sh = _sigma_hat_at(s, self.sigma_source, self.horizon, t)
        if not (np.isfinite(sh) and sh >= self.gate):
            return None
        if self.primary == "ma_cross":
            before, now = self._cross(s, t)
            sign = 1 if (before <= 0 < now) else -1 if (before >= 0 > now) else 0
        else:
            _, upper, lower = self._bands(s, t)
            c = s.close[t]
            sign = 1 if c > upper else -1 if c < lower else 0
        if sign == 0:
            return None
        return Order(_side(sign), self.size, reason=self.primary, max_hold=self.max_hold)

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        sign = 1 if side == "LONG" else -1
        if self.primary == "ma_cross":
            _, now = self._cross(s, t)
            return "REV" if sign * now < 0 else None        # fast back through slow: the opposite cross
        mid, _, _ = self._bands(s, t)
        return "MID" if sign * (s.close[t] - mid) < 0 else None   # the close back through the middle band
