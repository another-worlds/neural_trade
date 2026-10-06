"""D-051 / NT-115: coherence flags use the raw heads; entry and take-profit stay on the served delta.

Counts below are on this hand-built frame. No training run.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy.signals import SignalFrame
from neural_trade.strategy.strategies import EnhancedMultiHorizonStrategy, LiberalStrategy


def _frame(served: np.ndarray, raw: np.ndarray | None, *, p: float = 0.8) -> PredictionFrame:
    n = len(served)
    close = np.full(n, 100_000.0)
    prob = np.full(n, p)
    var = np.full(n, 0.01)          # sigma_$ = 10 at pred_scale 100
    frame = PredictionFrame(
        y=np.zeros((n, 3)), last_close=close,
        delta={h: served[:, i] for i, h in enumerate(HORIZONS)},
        direction_prob={h: prob for h in HORIZONS},
        variance_scaled={h: var for h in HORIZONS},
        pred_scale=100.0,
    )
    if raw is not None:
        frame.meta["delta_raw"] = {h: raw[:, i] for i, h in enumerate(HORIZONS)}
    return frame


def _flag_checks(p: np.ndarray, deltas: np.ndarray):
    mag = (np.abs(deltas[:, 0]) <= np.abs(deltas[:, 1]) + 1e-6) & (np.abs(deltas[:, 1]) <= np.abs(deltas[:, 2]) + 1e-6)
    aligned = ((p > 0.5) == (deltas > 0)).all(1)
    return mag, aligned


def _counts(s: SignalFrame):
    """Entries from decide(), and INCOH exits from exit_signal on an open long held 3 bars."""
    enhanced, liberal = EnhancedMultiHorizonStrategy(), LiberalStrategy()
    from types import SimpleNamespace

    held = SimpleNamespace(info={})
    out = {"enhanced": 0, "liberal": 0, "incoh": 0}
    for t in range(len(s)):
        out["enhanced"] += enhanced.decide(s, t) is not None
        out["liberal"] += liberal.decide(s, t) is not None
        out["incoh"] += enhanced.exit_signal(s, t, "LONG", 3, float(s.close[t]), held) == "INCOH"
    return out


def test_positive_beta_flags_match_the_raw_heads():
    raw = np.array([[1.0, 3.0, 9.0], [4.0, 2.0, 8.0], [-2.0, -5.0, -1.0], [1.0, -4.0, 7.0]], float)
    served = 0.25 * raw
    s = SignalFrame.build(_frame(served, raw), 1.0)
    p = np.full((len(raw), 3), 0.8)
    mag, aligned = _flag_checks(p, raw)
    assert s.coherence_on_raw
    np.testing.assert_array_equal(s.magnitude_coherent, mag)
    np.testing.assert_array_equal(s.direction_aligned, aligned)
    np.testing.assert_allclose(s.delta, served)


def test_beta_0_on_h1_still_follows_the_raw_heads():
    raw = np.array([[2.0, 5.0, 9.0], [8.0, 1.0, 3.0], [1.0, 4.0, 2.0]], float)
    served = raw.copy()
    served[:, 1] = 0.0
    s = SignalFrame.build(_frame(served, raw), 1.0)
    p = np.full((len(raw), 3), 0.8)
    mag, aligned = _flag_checks(p, raw)
    served_mag, served_aligned = _flag_checks(p, served)
    assert s.coherence_on_raw
    np.testing.assert_array_equal(s.magnitude_coherent, mag)
    np.testing.assert_array_equal(s.direction_aligned, aligned)
    assert not np.array_equal(mag, served_mag) or not np.array_equal(aligned, served_aligned)
    # the served h1 of 0 would make direction_aligned ask P(up) <= 0.5 there; these bars do not
    assert served_aligned.sum() == 0
    assert aligned.sum() == len(raw)


def test_without_raw_heads_the_flags_stay_on_the_served_delta():
    served = np.array([[1.0, 0.0, 4.0], [2.0, 3.0, 5.0]], float)
    s = SignalFrame.build(_frame(served, None), 1.0)
    p = np.full((len(served), 3), 0.8)
    mag, aligned = _flag_checks(p, served)
    assert not s.coherence_on_raw
    np.testing.assert_array_equal(s.magnitude_coherent, mag)
    np.testing.assert_array_equal(s.direction_aligned, aligned)


def test_enhanced_does_not_enter_when_the_served_h1_delta_is_zero():
    """Raw h1 is +500. The served h1 is 0, so d1 > 0 still blocks the entry."""
    n = 4
    raw = np.column_stack([np.full(n, 10.0), np.full(n, 500.0), np.full(n, 40.0)])
    served = raw.copy()
    served[:, 1] = 0.0
    s = SignalFrame.build(_frame(served, raw), 1.0)
    enhanced = EnhancedMultiHorizonStrategy()
    assert all(enhanced.decide(s, t) is None for t in range(n))
    # liberal sizes off the served h1, floored by half a predicted sigma, not the raw 500
    order = LiberalStrategy().decide(s, 0)
    assert order is not None
    floor = 0.5 * float(s.sigma[0, 1])
    assert floor == pytest.approx(5.0)
    assert order.tp == pytest.approx(floor)
    assert order.info["tp1_offset"] == pytest.approx(floor * 0.5)
    assert order.tp != pytest.approx(500.0)


def test_take_profit_stays_on_the_served_delta_and_incoh_follows_the_raw_order():
    """Same served h1. Only the raw magnitude order changes, so take-profits stay and INCOH moves.

    N = 8. Before (raw ordered): enhanced entries 8, liberal entries 8, INCOH 0.
    After (first 4 bars break |h0| <= |h1| <= |h2|): enhanced entries 8, liberal entries 8, INCOH 4.
    """
    n = 8
    served = np.column_stack([np.full(n, 10.0), np.full(n, 20.0), np.full(n, 40.0)])
    raw_before = served.copy()
    raw_after = served.copy()
    raw_after[:4, 0] = 80.0
    raw_after[:4, 2] = 5.0
    before = SignalFrame.build(_frame(served, raw_before), 1.0)
    after = SignalFrame.build(_frame(served, raw_after), 1.0)
    enhanced, liberal = EnhancedMultiHorizonStrategy(), LiberalStrategy()
    for t in range(n):
        for strat in (enhanced, liberal):
            a, b = strat.decide(before, t), strat.decide(after, t)
            assert a is not None and b is not None
            assert a.tp == pytest.approx(b.tp)
            assert a.info["tp1_offset"] == pytest.approx(b.info["tp1_offset"])
            assert a.tp == pytest.approx(b.tp) and abs(a.tp) == pytest.approx(20.0 if strat is enhanced else 20.0)
    # enhanced tp2 multiplier is 1, so tp offset is the served |h1| of 20; liberal is the same here
    assert enhanced.decide(before, 0).tp == pytest.approx(20.0)
    assert liberal.decide(before, 0).tp == pytest.approx(20.0)
    assert _counts(before) == {"enhanced": 8, "liberal": 8, "incoh": 0}
    assert _counts(after) == {"enhanced": 8, "liberal": 8, "incoh": 4}
