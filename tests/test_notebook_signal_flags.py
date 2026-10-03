"""BacktestExplorer.signal_summary: the flags table says when a served delta of 0 (beta = 0) decides a flag."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from neural_trade.evaluation.frame import HORIZONS


def _explorer(frame, bars):
    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer

    return BacktestExplorer({"config": Config(), "test": frame, "cal": frame, "bars": bars,
                             "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))})


def _flags(viz_frame, viz_backtest, betas):
    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    fr.meta.pop("delta_raw", None)   # this helper is the no-raw path: a 0 decides the flags
    fr.delta = {h: betas[h] * np.asarray(fr.delta[h], float) for h in HORIZONS}
    ex = _explorer(fr, bars)
    return ex.signal_summary()["flags (share of bars true)"], ex.signals


def test_flags_table_notes_that_a_served_delta_of_zero_decides_the_flags(viz_frame, viz_backtest):
    """beta = 0 on every horizon: magnitude_coherent is 1.0 by ties and direction_aligned is the share of bars
    with all three P(up) <= 0.5. The table printed both as measured (final review, finding 14)."""
    flags, s = _flags(viz_frame, viz_backtest, {h: 0.0 for h in HORIZONS})
    assert flags.loc["magnitude_coherent", "share true"] == 1.0
    assert flags.loc["direction_aligned", "share true"] == pytest.approx(float(np.mean((s.p <= 0.5).all(1))))
    note = flags["note"]
    assert "true on every bar by ties" in note["magnitude_coherent"] and "h0, h1, h2 (beta = 0)" in note["magnitude_coherent"]
    assert "all three P(up) <= 0.5, not a sign agreement" in note["direction_aligned"]
    assert note["var_spike"] == ""
    # beta = 0 on h1 only: the note names it
    flags1, _ = _flags(viz_frame, viz_backtest, {"h0": 0.3, "h1": 0.0, "h2": 0.4})
    assert "served delta is 0 on h1 (beta = 0)" in flags1.loc["direction_aligned", "note"]
    assert "P(up) <= 0.5" in flags1.loc["direction_aligned", "note"]


def test_flags_table_has_no_note_while_every_served_delta_varies(viz_frame, viz_backtest):
    flags, _ = _flags(viz_frame, viz_backtest, {"h0": 0.21, "h1": 0.023, "h2": 0.25})
    assert list(flags.columns) == ["share true", "bars"]
    assert flags["share true"].between(0, 1).all()


def test_raw_heads_decide_the_flags_when_the_served_delta_is_zero(viz_frame, viz_backtest):
    """D-051: with raw heads, a served delta of 0 does not make the flags the tie or the DDD share."""
    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    n = len(fr.last_close)
    p = np.column_stack([np.asarray(fr.prob(h, True), float)[:n] for h in HORIZONS])
    sign = np.where(p > 0.5, 1.0, -1.0)
    raw = np.column_stack([sign[:, i] * scale for i, scale in enumerate((5.0, 15.0, 40.0))])
    fr.meta["delta_raw"] = {h: raw[:, i] for i, h in enumerate(HORIZONS)}
    fr.delta = {h: np.zeros(n) for h in HORIZONS}
    ex = _explorer(fr, bars)
    flags = ex.signal_summary()["flags (share of bars true)"]
    s = ex.signals
    assert s.coherence_on_raw
    assert flags.loc["magnitude_coherent", "share true"] == 1.0
    assert flags.loc["direction_aligned", "share true"] == 1.0
    ddd = float(np.mean((s.p <= 0.5).all(1)))
    assert ddd < 1.0
    note = flags["note"]
    assert "measured on the raw heads, not decided by the 0" in note["magnitude_coherent"]
    assert "measured on the raw heads, not decided by the 0" in note["direction_aligned"]
    assert "true on every bar by ties" not in note["magnitude_coherent"]
    assert "P(up) <= 0.5" not in note["direction_aligned"]
    assert note["var_spike"] == ""
