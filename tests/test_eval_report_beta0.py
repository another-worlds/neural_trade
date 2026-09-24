"""The report's price-head table when the calibration's shrink beta is 0: the served delta is 0 on every
sample, so its skill, EV and correlation are fixed by construction and are not printed as measured."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.evaluation.report import SERVED_ZERO_NA, evaluate


def _frames(betas, n=3000, seed=0):
    """(served frame, raw heads): weakly informative raw heads, served = beta x raw."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 200, (n, 3))
    raw = {h: 0.15 * y[:, i] + rng.normal(0, 150, n) for i, h in enumerate(HORIZONS)}
    prob = {h: 1 / (1 + np.exp(-raw[h] / 100)) for h in HORIZONS}
    frame = PredictionFrame(y, np.full(n, 110_000.0), {h: betas[h] * raw[h] for h in HORIZONS}, prob,
                            {h: np.ones(n) for h in HORIZONS}, 200.0, 0.0, (10, 15, 20), "test")
    return frame, raw


def _row(md, label):
    """The cells h0..h2 of the markdown row whose label is exactly ``label``."""
    (line,) = [ln for ln in md.splitlines() if ln.startswith(f"| {label} |")]
    return [c.strip() for c in line.strip("|").split("|")[1:]]


def _fmt(v):
    return f"{v:.4f}"


def test_beta_zero_prints_the_raw_heads_correlation_not_the_served_zero():
    """The row read '(the same raw and served) | 0.0000 | 0.0000 | 0.0000' while the raw heads'
    correlation was not 0 (the delta figure showed it): a constant has no correlation."""
    from scipy.stats import spearmanr

    betas = {h: 0.0 for h in HORIZONS}
    frame, raw = _frames(betas)
    md = evaluate(frame, Config(), raw_delta=raw, delta_scale=betas).to_markdown()
    assert "the same raw and served" not in md
    pearson = _row(md, "corr, Pearson, raw heads (the same for served while beta > 0)")
    spearman = _row(md, "corr, Spearman, raw heads")
    for i, h in enumerate(HORIZONS):
        y = frame.y[:, i]
        assert pearson[i] == _fmt(np.corrcoef(raw[h], y)[0, 1]) != "0.0000"
        assert spearman[i] == _fmt(spearmanr(raw[h], y).correlation)
    # the served skill and EV of a delta that is 0 are 0 by construction: n/a, the raw heads' are measured
    assert _row(md, "skill vs zero (1 - MSE / MSE of 0), served") == [SERVED_ZERO_NA] * 3
    assert _row(md, "EV, served") == [SERVED_ZERO_NA] * 3
    assert all(c.startswith("-") or c[0].isdigit() for c in _row(md, "skill vs zero, raw heads"))
    assert SERVED_ZERO_NA == "n/a (beta = 0: served delta is 0)"


def test_beta_above_zero_the_raw_heads_correlation_equals_the_served_one():
    betas = {"h0": 0.21, "h1": 0.023, "h2": 0.25}
    frame, raw = _frames(betas, seed=1)
    rep = evaluate(frame, Config(), raw_delta=raw, delta_scale=betas)
    md = rep.to_markdown()
    pearson = _row(md, "corr, Pearson, raw heads (the same for served while beta > 0)")
    for i, h in enumerate(HORIZONS):
        assert pearson[i] == _fmt(rep.model["horizons"][h]["delta"]["corr"])        # served, beta > 0
        assert rep.model["horizons"][h]["delta"]["corr"] == pytest.approx(
            rep.model["horizons"][h]["delta_raw"]["corr"], abs=1e-9)
    served_skill = _row(md, "skill vs zero (1 - MSE / MSE of 0), served")
    assert served_skill == [_fmt(rep.model["horizons"][h]["delta"]["skill_vs_zero"]) for h in HORIZONS]
    assert "n/a" not in " ".join(served_skill + _row(md, "EV, served"))


def test_without_raw_heads_a_zero_beta_horizon_reads_n_a_and_the_others_are_measured():
    """beta = 0 on h1 and no raw heads (they cannot be recovered from a zero beta): the served rows."""
    betas = {"h0": 0.3, "h1": 0.0, "h2": 0.2}
    frame, _ = _frames(betas, seed=2)
    rep = evaluate(frame, Config(), delta_scale=betas)
    assert "delta_raw" not in rep.model["horizons"]["h1"]
    md = rep.to_markdown()
    for label, key in (("corr, Pearson", "corr"), ("corr, Spearman", "corr_spearman"), ("EV, served", "ev"),
                       ("skill vs zero (1 - MSE / MSE of 0), served", "skill_vs_zero")):
        cells = _row(md, label)
        assert cells[1] == SERVED_ZERO_NA, label
        assert [cells[0], cells[2]] == [_fmt(rep.model["horizons"][h]["delta"][key]) for h in ("h0", "h2")]
    # the dollar rows are real values of the served prediction (0 predicts no move): still printed
    assert _row(md, "RMSE ($), served")[1] == _row(md, "RMSE ($), zero prediction")[1]
    assert _row(md, "mean predicted ($), served")[1] == "0.00"
