"""Learned indicator periods."""
from __future__ import annotations

import numpy as np

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")


def test_indicator_evolution_and_summary():
    import pandas as pd

    from neural_trade.visualization.indicator_evolution import indicator_evolution, indicator_summary

    rng = np.random.default_rng(3)
    cols = ([f"ma_period_{i}" for i in range(3)] + [f"macd_{i}_{r}" for i in range(3) for r in ("fast", "slow", "signal")]
            + [f"rsi_period_{i}" for i in range(3)] + [f"bb_period_{i}" for i in range(3)])
    rows = [{"epoch": e, "val_loss": 5 - 0.1 * e, **{f"period/{c}": 10 + e * rng.normal() for c in cols}}
            for e in range(8)]
    fig = indicator_evolution(rows)
    assert T.empty_panels(fig) == []
    s = indicator_summary(pd.DataFrame(rows))
    assert len(s) == 18 and {"change %", "corr with val loss"} <= set(s.columns)
