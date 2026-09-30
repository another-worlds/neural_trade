"""Indicators registry (NT-046): the family contract of D-031 and the M(eps) warm-up bound
of D-037 (each entry declares inputs, learnable parameters with textbook defaults and bounds,
output channels, drawing spec and M(eps); M(eps) is at least the empirical offset-invariance
value)."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.exceptions import ComponentValidationError, DuplicateRegistrationError
from neural_trade.indicators import (
    ChannelSpec,
    IndicatorFamily,
    Indicators,
    ParamSpec,
    compute_reference,
    num_learnable_logits,
)

FAMILIES = ("ma", "macd", "rsi", "bb")
TEXTBOOK = {"ma": {"period": 20.0}, "macd": {"fast": 12.0, "slow": 26.0, "signal": 9.0},
            "rsi": {"period": 14.0}, "bb": {"period": 20.0}}
EPS = 1e-3
SHIFT = -0.5  # the maximal slowing meta shift (META_SCALE * tanh(-1) -> -0.5)


@pytest.mark.parametrize("name", FAMILIES)
def test_each_family_entry_declares_the_full_contract(name):
    fam = Indicators.get(name)
    assert isinstance(fam, IndicatorFamily) and fam.name == name
    # inputs: the close (and the series derived from it) today
    assert "close" in fam.inputs
    assert all(inp in ("close", "gains", "losses") for inp in fam.inputs)
    # learnable parameters: textbook defaults and bounds
    assert len(fam.params) == len(TEXTBOOK[name])
    for p in fam.params:
        assert isinstance(p, ParamSpec)
        assert p.default == TEXTBOOK[name][p.name]
        assert p.minimum == 2.0 and p.maximum is None  # None -> the configured ceiling
    # output channels and drawing spec
    assert len(fam.channels) >= 1
    assert all(isinstance(c, ChannelSpec) and c.draw in ("price", "panel") for c in fam.channels)
    assert fam.draw in ("price", "panel")
    # M(eps) is declared and positive at the textbook parameters
    m = fam.m_eps(TEXTBOOK[name], EPS, SHIFT)
    assert isinstance(m, int) and m > 0


def test_registry_is_strict():
    with pytest.raises(ComponentValidationError):
        Indicators.register(name="broken")(object())
    with pytest.raises(DuplicateRegistrationError):
        Indicators.register(name="ma")(Indicators.get("rsi"))


def test_default_config_declares_54_logits_and_the_old_default_18():
    from neural_trade.core.config import Config

    assert num_learnable_logits(Config()) == 54  # 14 families x 3 instances (NT-047)
    assert num_learnable_logits(Config(INDICATOR_FAMILIES={})) == 18  # the NT-046 four


def _empirical_offset_invariance(fam, periods, n=1400, k=400):
    """Smallest m such that starting the state k bars later changes no channel by more than
    EPS (relative to the channel's own scale) from bar k+m on."""
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(0.0, 0.05, n)).astype(np.float32)
    full = compute_reference(fam, x, periods, SHIFT)
    off = compute_reference(fam, x[k:], periods, SHIFT)
    worst = np.zeros(n - k)
    for cf, co in zip(full, off):
        scale = max(1.0, float(np.abs(cf).max()))
        worst = np.maximum(worst, np.abs(cf[0, k:] - co[0, :]) / scale)
    over = np.nonzero(worst > EPS)[0]
    return int(over[-1]) + 1 if len(over) else 0


@pytest.mark.parametrize("name, periods", [
    ("ma", {"period": 30.0}),                                # the slowest configured instance
    ("macd", {"fast": 5.0, "slow": 35.0, "signal": 5.0}),
    ("rsi", {"period": 21.0}),
    ("bb", {"period": 25.0}),
])
def test_m_eps_is_at_least_the_empirical_offset_invariance_value(tf, name, periods):
    fam = Indicators.get(name)
    declared = fam.m_eps(periods, EPS, SHIFT)
    empirical = _empirical_offset_invariance(fam, periods)
    assert declared >= empirical, (f"{name}: declared M(eps)={declared} < empirical "
                                   f"offset-invariance {empirical}")
