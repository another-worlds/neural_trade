"""NT-042: every figure and table that takes horizons works for N = 2 and N = 4 (and keeps N = 3).

No TensorFlow: frames, reports, signals and backtests are built from arrays. Each figure must draw one
column (or series) per horizon in that horizon's colour (theme.horizon_color) and leave no panel empty
(theme.empty_panels, D-014).
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pytest

from neural_trade.core.horizons import horizon_keys
from neural_trade.visualization import theme as T

N_CASES = (2, 4)


def _frame(n_h: int, n: int = 2500, seed: int = 0):
    from neural_trade.evaluation.frame import PredictionFrame

    rng = np.random.default_rng(seed)
    keys = horizon_keys(n_h)
    sigma = rng.uniform(60, 240, (n, n_h))
    signal = rng.normal(0, 1, (n, n_h))
    y = 0.2 * signal * sigma + rng.normal(0, 1, (n, n_h)) * sigma
    p = 1 / (1 + np.exp(-0.4 * signal))
    pred_scale = 100.0
    delta = {h: 0.3 * signal[:, i] * sigma[:, i] for i, h in enumerate(keys)}
    lo_hi = {h: (delta[h] - 1.645 * sigma[:, i], delta[h] + 1.645 * sigma[:, i]) for i, h in enumerate(keys)}
    fr = PredictionFrame(
        y=y, last_close=np.full(n, 100_000.0), delta=delta,
        direction_prob={h: p[:, i] for i, h in enumerate(keys)},
        variance_scaled={h: (sigma[:, i] / pred_scale) ** 2 for i, h in enumerate(keys)},
        pred_scale=pred_scale, horizon_steps=tuple(10 + 5 * i for i in range(n_h)),
        direction_prob_calibrated={h: 0.5 + 0.8 * (p[:, i] - 0.5) for i, h in enumerate(keys)},
        intervals=lo_hi)
    fr.meta["delta_raw"] = {h: 2.0 * v for h, v in delta.items()}
    fr.meta["delta_scale"] = {h: 0.5 for h in keys}
    return fr


def _config(n_h: int):
    from neural_trade.core.config import Config

    steps = [10 + 5 * i for i in range(n_h)]
    return Config(HORIZON_STEPS=steps, EXTENDED_TREND_PERIODS=steps, EPOCHS=6, EARLY=3)


def _colours(fig) -> set:
    """Every colour string in the figure's JSON (traces, layout, annotations)."""
    return {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}", fig.to_json())}


def _assert_horizons_drawn(fig, n_h: int):
    cols = _colours(fig)
    for h in horizon_keys(n_h):
        assert T.horizon_color(h).lower() in cols, f"{h}'s colour {T.horizon_color(h)} is not in the figure"
    assert T.empty_panels(fig) == [], f"empty panels: {T.empty_panels(fig)}"


# ------------------------------------------------------------------ theme
def test_horizon_colours_extend_in_a_fixed_order_and_avoid_the_reserved_roles():
    """h0-h2 keep slots 1-3 (owner colours, D-014); h3-h5 take HORIZON_EXTENSION in order; every
    extension colour is at least 30 CIE76 Delta E from the status, LONG / SHORT, costs (NEUTRAL) colours
    and the paper, and every horizon colour that far from every other one (acceptance 4)."""
    def lab(hex_):
        rgb = np.array([int(hex_.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
        xyz = m @ rgb / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    assert [T.horizon_color(h) for h in ("h0", "h1", "h2")] == ["#3987e5", "#d95926", "#199e70"]
    assert dict(T.HORIZON_COLORS) == {"h0": "#3987e5", "h1": "#d95926", "h2": "#199e70"}
    six = [T.horizon_color(h) for h in horizon_keys(6)]
    assert six[3:] == list(T.HORIZON_EXTENSION)
    assert T.HORIZON_COLORS["h3"] == T.HORIZON_EXTENSION[0] and T.HORIZON_COLORS.get("h4") == T.HORIZON_EXTENSION[1]
    assert T.HORIZON_COLORS.get("weighted") is None
    reserved = {"GOOD": T.GOOD, "WARNING": T.WARNING, "SERIOUS": T.SERIOUS, "CRITICAL": T.CRITICAL,
                "LONG": T.LONG_COLOR, "SHORT": T.SHORT_COLOR, "costs (NEUTRAL)": T.NEUTRAL, "PAPER": T.PAPER}
    for i, c in enumerate(six):
        for name, r in reserved.items():
            if i >= 3:     # h0-h2 are the owner's fixed slots; the extension is what this item chose
                assert np.linalg.norm(lab(c) - lab(r)) >= 30, (f"h{i}", name)
        for j, other in enumerate(six):
            if j != i:
                assert np.linalg.norm(lab(c) - lab(other)) >= 30, (f"h{i}", f"h{j}")
    assert T.horizon_label("h3", _config(4)) == "h3 (25 bars)"


# ------------------------------------------------------------------ evaluation tables and report
@pytest.mark.parametrize("n_h", N_CASES)
def test_report_tables_cover_every_horizon(n_h):
    from neural_trade.evaluation.report import evaluate
    from neural_trade.visualization import analytics_tables as AT

    fr, cfg = _frame(n_h), _config(n_h)
    keys = horizon_keys(n_h)
    rep = evaluate(fr, cfg, raw_delta=fr.meta["delta_raw"])
    assert rep.horizons == keys and list(rep.model["horizons"]) == list(keys)
    c = rep.model["coherence"]
    pair_keys = [f"mag_{a}_le_{b}" for a, b in zip(keys, keys[1:])]
    assert all(k in c and f"{k}_raw" in c for k in pair_keys)
    assert c["coherence_primary"] == c[f"delta_dir_align_h{n_h // 2}"]
    md = rep.to_markdown()
    assert f"| | {' | '.join(keys)} | all {n_h} |" in md
    assert f"full chain {' <= '.join(keys)}" in md
    for table in (AT.classification_table(fr, cfg, digits=None),
                  AT.delta_quality_table(fr, cfg, raw_delta=fr.meta["delta_raw"], digits=None),
                  AT.alignment_table(fr, cfg, raw_delta=fr.meta["delta_raw"], digits=None)):
        text = table.to_string()
        assert all(h in text for h in keys), text[:400]
    mo = AT.magnitude_ordering_table(fr, raw_delta=fr.meta["delta_raw"], digits=None)
    assert len(mo) == n_h        # one row per neighbouring pair plus the full chain


# ------------------------------------------------------------------ the analytics figures
@pytest.mark.parametrize("n_h", N_CASES)
@pytest.mark.parametrize("name", ["direction_analytics", "delta_analytics", "variance_analytics",
                                  "confidence_analytics", "coherence_analytics"])
def test_model_analytics_figures_draw_every_horizon(name, n_h):
    from neural_trade.visualization import model_analytics as MA

    fr, cfg = _frame(n_h), _config(n_h)
    fig = getattr(MA, name)(fr, cfg)
    _assert_horizons_drawn(fig, n_h)


@pytest.mark.parametrize("n_h", N_CASES)
def test_eval_report_figure_draws_every_horizon(n_h):
    from neural_trade.visualization.eval_report import eval_report_figure

    fr, cfg = _frame(n_h), _config(n_h)
    fig = eval_report_figure(fr, cfg)
    _assert_horizons_drawn(fig, n_h)
    assert {t.name for t in fig.data} >= {f"{h} reliability" for h in horizon_keys(n_h)}


# ------------------------------------------------------------------ signals, strategies, trading figures
@pytest.mark.parametrize("n_h", N_CASES)
def test_signal_frame_and_strategies_trade_on_n_horizons(n_h):
    from neural_trade.strategy import Bars, SignalFrame, backtest, build_backtest_config, build_strategy

    fr = _frame(n_h)
    rng = np.random.default_rng(2)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, len(fr))))
    fr.last_close = close
    sig = SignalFrame.build(fr, 1.0)
    assert sig.p.shape == (len(fr), n_h) and sig.primary == n_h // 2 and sig.coherence_on_raw
    assert np.all((sig.agreement >= 1.0 / n_h - 1e-12) & (sig.agreement <= 1.0))
    bars = Bars.from_close(close)
    for name in ("calibrated_quantile", "enhanced_multi_horizon", "threshold_spike", "liberal"):
        strat = build_strategy(name, None, calibration=sig)
        res = backtest(sig, bars, strat, build_backtest_config({"random_seeds": 0}))
        assert np.isfinite(res.summary["net_pnl"])


@pytest.mark.parametrize("n_h", N_CASES)
def test_trading_figures_draw_on_n_horizons(n_h):
    from neural_trade.strategy import Bars, SignalFrame, backtest, build_backtest_config, build_strategy
    from neural_trade.visualization.trading_dashboard import trade_analytics, trading_dashboard

    fr = _frame(n_h)
    rng = np.random.default_rng(2)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, len(fr))))
    fr.last_close = close
    sig = SignalFrame.build(fr, 1.0)
    bars = Bars.from_close(close)
    strat = build_strategy("calibrated_quantile", None, calibration=sig)
    res = backtest(sig, bars, strat, build_backtest_config({"random_seeds": 0}))
    assert res.summary["n_trades"] > 0
    fig = trading_dashboard(res, bars=bars, signals=sig, strategy=strat)
    assert T.empty_panels(fig) == []
    primary = f"h{n_h // 2}"
    assert T.horizon_color(primary).lower() in _colours(fig)      # the primary horizon's sigma line
    assert any(t.name == f"sigma {primary}" for t in fig.data)
    fig2 = trade_analytics(res, bars=bars, signals=sig, horizon_steps=fr.horizon_steps)
    assert f"Predicted vs realised {primary} move" in fig2.to_json()
    # random_seeds=0 leaves the two random-entry panels empty at three horizons too (not an NT-042 matter)
    assert set(T.empty_panels(fig2)) <= {"y7", "y9"}


# ------------------------------------------------------------------ training figures (logged keys)
def _history(n_h: int, n: int = 6):
    rng = np.random.default_rng(1)
    rows = []
    for e in range(n):
        r = {"epoch": e, "loss": 7 - 0.2 * e, "val_loss": 6 - 0.1 * e + 0.05 * rng.normal(), "seconds": 20.0}
        for pre in ("", "val_"):
            for k in ("point_loss", "trend_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss",
                      "t_perp_loss", "casimir_loss", "hd_loss"):
                r[pre + k] = float(rng.uniform(0.01, 1))
            for h in horizon_keys(n_h):
                for k in ("point", "extended", "dir_loss", "nll", "crps", "soft_ece"):
                    r[f"{pre}{k}_{h}"] = float(rng.uniform(0.01, 1))
        for h in horizon_keys(n_h):
            for pre in ("train_", "val_"):
                for k in ("dir_mcc", "gauss_dir_mcc", "dir_bal_acc", "dir_ece", "dir_acc", "dir_brier", "dir_f1",
                          "dir_sensitivity", "dir_specificity", "mean_dir_prob", "true_up_rate"):
                    r[f"{pre}{k}_{h}"] = float(rng.uniform(0, 0.6))
            r[f"val_pred_up_rate_{h}"], r[f"val_true_up_rate_{h}"] = 0.48, 0.5
            r[f"val_pit_ks_{h}"], r[f"pit_ks_{h}"] = 0.1, 0.05
            r[f"dir_n_{h}"], r[f"val_dir_n_{h}"] = 400.0, 300.0
        r.update(lr=1e-3 if e < 4 else 5e-4, lr_indicator=5e-3, grad_global_norm=8.0, nonfinite_grad_steps=0.0)
        rows.append(r)
    return rows


@pytest.mark.parametrize("n_h", N_CASES)
@pytest.mark.parametrize("name", ["training_dashboard", "training_direction_detail", "training_loss_terms"])
def test_training_figures_draw_every_logged_horizon(name, n_h):
    from neural_trade.visualization import training_dashboard as TD

    fig = getattr(TD, name)(_history(n_h), _config(n_h))
    cols = _colours(fig)
    if name != "training_loss_terms":
        for h in horizon_keys(n_h):
            assert T.horizon_color(h).lower() in cols, h
    assert T.empty_panels(fig) == []


def test_reference_frame_figures_unchanged_in_shape():
    """N = 3 keeps three columns: the analytics figures still have exactly three x axes per row."""
    from neural_trade.visualization import model_analytics as MA

    fr, cfg = _frame(3), _config(3)
    fig = MA.direction_analytics(fr, cfg)
    assert fig.layout.xaxis3 is not None and "xaxis10" not in fig.layout
    assert copy.deepcopy(fig) is not None
