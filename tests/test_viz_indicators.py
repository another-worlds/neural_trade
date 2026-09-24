"""Learned indicator periods: the evolution figure, its table, the applied periods, period_init.json."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.indicator_evolution import (
    applied_periods, configured_periods, indicator_applied_periods, indicator_evolution, indicator_summary,
)

H = ("h0", "h1", "h2")
NAMES = list(configured_periods(Config()))


def _rows(n=20, seed=3, drift=None, noise=0.02, vl_noise=0.05):
    """metrics.jsonl-style rows: every period drifts from its configured start; val loss falls."""
    rng = np.random.default_rng(seed)
    start = configured_periods(Config())
    drift = drift or {}
    rows = []
    for e in range(n):
        row = {"epoch": e, "val_loss": 5 - 0.05 * e + vl_noise * rng.normal()}
        for c, s0 in start.items():
            row[f"period/{c}"] = s0 * (1 + drift.get(c, -0.01) * (e + 1)) * (1 + noise * rng.normal())
        rows.append(row)
    return rows


def _panel_traces(fig, r, c):
    sp = fig.get_subplot(r, c)
    xa, ya = sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")
    return [t for t in fig.data if (t.xaxis or "x") == xa and (t.yaxis or "y") == ya and t.x is not None
            and any(v is not None for v in t.x)]


# ------------------------------------------------------------------ finding 11: the baseline is the start
def test_change_is_measured_from_the_configured_start_not_from_epoch_1():
    rows = [{"epoch": 0, "val_loss": 5.0, "period/ma_period_0": 4.0}, {"epoch": 1, "val_loss": 4.9,
                                                                       "period/ma_period_0": 3.0}]
    s = indicator_summary(rows, Config())
    assert s.loc["ma_period_0", "start"] == 5.0                # MA_SPANS[0]: where the logit is initialised
    assert s.loc["ma_period_0", "change %"] == pytest.approx(-40.0)    # old code: -25 (from epoch 1)
    assert s.loc["ma_period_0", "max"] == 5.0 and s.attrs["change_from"] == "config"
    fig = indicator_evolution(rows, Config())
    line = next(t for t in _panel_traces(fig, 1, 1) if t.type == "scatter" and t.mode == "lines+markers")
    assert line.x[0] == 0 and line.y[0] == 5.0 and line.marker.symbol[0] == "circle-open"
    assert any("Change of each base period, configured start" in (a.text or "") for a in fig.layout.annotations)


def test_without_a_start_the_change_is_from_epoch_1_and_says_so():
    rows = [{"epoch": 0, "period/ma_period_0": 4.0}, {"epoch": 1, "period/ma_period_0": 3.0}]
    s = indicator_summary(rows)
    assert "start" not in s.columns and s.loc["ma_period_0", "change %"] == pytest.approx(-25.0)
    fig = indicator_evolution(rows)
    assert any("end of epoch 1" in (a.text or "") for a in fig.layout.annotations)


def test_a_metrics_path_without_config_uses_the_runs_config_yaml(tmp_path):
    """The notebooks call the figure with the config and the table without it: both must measure
    change from the same start (the table used to show +49.6% for macd_1_slow, the figure +66.6%)."""
    rows = _rows(n=10)
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    Config(MA_SPANS=[6, 10, 30]).to_yaml(tmp_path / "config.yaml")
    bare = indicator_summary(tmp_path / "metrics.jsonl")
    assert bare.attrs["change_from"] == "config" and bare.loc["ma_period_0", "start"] == 6.0
    given = indicator_summary(tmp_path / "metrics.jsonl", Config(MA_SPANS=[6, 10, 30]))
    pd.testing.assert_series_equal(bare["change %"], given["change %"])
    assert "near bound" in bare.columns                                  # the clip bounds came with it
    fig = indicator_evolution(tmp_path / "metrics.jsonl")
    assert any("configured start to epoch 10" in (a.text or "") for a in fig.layout.annotations)
    (tmp_path / "config.yaml").write_text("not: [a, mapping", encoding="utf-8")   # unreadable: epoch-1 fallback
    assert indicator_summary(tmp_path / "metrics.jsonl").attrs["change_from"] == "epoch 1"


def test_change_labels_print_start_and_end_alike():
    rows = [{"epoch": e, "val_loss": 5.0 - e / 10, "period/ma_period_0": v, "period/rsi_period_1": w}
            for e, (v, w) in enumerate(zip([4.77, 3.5, 2.9], [14.96, 15.3, 15.72]))]
    fig = indicator_evolution(rows)                                      # no start known: from epoch 1
    text = {t.name: t.text[0] for t in _panel_traces(fig, 3, 1) if t.type == "bar"}
    assert text == {"ma_period_0": "4.8 → 2.9", "rsi_period_1": "15.0 → 15.7"}


def test_telemetry_does_not_import_the_plotting_code():
    import inspect

    from neural_trade.core import indicator_periods
    from neural_trade.telemetry import epoch_logger
    from neural_trade.visualization import indicator_evolution as ie

    assert "visualization" not in inspect.getsource(epoch_logger)
    assert ie.configured_periods is indicator_periods.configured_periods is epoch_logger.configured_periods
    assert ie.PERIOD_INIT_FILE == epoch_logger.PERIOD_INIT_FILE == "period_init.json"


def test_period_init_json_is_the_start_and_flags_a_warm_start(tmp_path):
    rows = _rows(n=3)
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    start = configured_periods(Config())
    start["ma_period_0"] = 8.0                                 # warm start: not the configured 5
    (tmp_path / "period_init.json").write_text(json.dumps({"periods": start, "matches_config": False}))
    s = indicator_summary(tmp_path / "metrics.jsonl", Config())
    last = rows[-1]["period/ma_period_0"]
    assert s.loc["ma_period_0", "start"] == 8.0
    assert s.loc["ma_period_0", "change %"] == pytest.approx(100 * (last - 8.0) / 8.0)
    fig = indicator_evolution(tmp_path / "metrics.jsonl", Config())
    assert any("warm start" in (t.name or "") for t in fig.data)


def test_epoch_logger_records_the_periods_when_training_begins(tf, tmp_path):
    from neural_trade.telemetry.epoch_logger import JsonlEpochLogger

    class _Layer:
        def __init__(self, periods, config):
            self.periods, self.config = periods, config

        def get_learned_parameters(self):
            return dict(self.periods)

    cfg = Config()
    cb = JsonlEpochLogger(tmp_path, indicator_layer=_Layer(configured_periods(cfg), cfg))
    cb.on_train_begin()
    rec = json.loads((tmp_path / "period_init.json").read_text(encoding="utf-8"))
    assert rec["matches_config"] is True and rec["periods"]["macd_1_slow"] == 35.0
    assert not (tmp_path / "metrics.jsonl").exists()          # no fake epoch row
    warm = dict(configured_periods(cfg), ma_period_0=3.3)
    cb.indicator_layer = _Layer(warm, cfg)
    cb.on_train_begin()                                       # a second fit() of the same logger: kept
    assert json.loads((tmp_path / "period_init.json").read_text())["matches_config"] is True
    cb2 = JsonlEpochLogger(tmp_path / "w", indicator_layer=_Layer(warm, cfg))
    cb2.on_train_begin()
    assert json.loads((tmp_path / "w" / "period_init.json").read_text())["matches_config"] is False
    assert cb.n_errors == 0 and cb2.n_errors == 0
    model, _ = _tiny_indicator_model(tf, cfg)                  # the real layer, freshly built from the config
    cb3 = JsonlEpochLogger(tmp_path / "real", indicator_layer=model.get_layer("learnable_indicators"))
    cb3.on_train_begin()
    rec = json.loads((tmp_path / "real" / "period_init.json").read_text())
    assert rec["matches_config"] is True and cb3.n_errors == 0
    assert rec["periods"] == pytest.approx(configured_periods(cfg), rel=1e-5)


# ------------------------------------------------------------------ finding 67: correlation beyond the trend
def test_correlation_panel_uses_epoch_to_epoch_changes_against_a_noise_band():
    n = 20
    wiggle_a = [(-1) ** e for e in range(n)]                          # changes +-2, +-2, ...
    wiggle_b = np.cumsum([0] + [1 if (e // 2) % 2 == 0 else -1 for e in range(n - 1)])   # changes 1, 1, -1, -1
    rows = [{"epoch": e, "val_loss": 5 - 0.1 * e + 0.05 * wiggle_a[e],
             "period/rsi_period_0": 9 - 0.15 * e + 0.05 * wiggle_b[e]} for e in range(n)]
    s = indicator_summary(rows, Config())
    r = s.loc["rsi_period_0"]
    assert r["corr with val loss"] > 0.9 and r["r with epoch"] < -0.9      # the shared training trend
    band = S.corr_null_r(n - 1)                                             # r units, not Fisher z
    assert abs(r["r of changes"]) < band                                     # nothing beyond it
    fig = indicator_evolution(rows, Config())
    bars = [t for t in _panel_traces(fig, 3, 2) if t.type == "bar"]
    assert len(bars) == 1 and bars[0].x[0] == pytest.approx(r["r of changes"], abs=1e-6)
    assert bars[0].name == "Δ in noise"
    levels = next(t for t in _panel_traces(fig, 3, 2) if t.type == "scatter")
    assert levels.x[0] == pytest.approx(r["corr with val loss"], abs=1e-6)
    rect = next(sh for sh in fig.layout.shapes if sh.type == "rect")
    assert rect.x0 == pytest.approx(-band) and rect.x1 == pytest.approx(band)
    assert not any(c in (T.SERIES[0], T.SERIES[1]) for t in bars for c in [t.marker.color])


def _rows_with_r_of_changes(n, r, seed=5):
    """``n`` epochs whose epoch-to-epoch changes of RSI #0 and of val loss correlate at exactly ``r``."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n - 1)
    a -= a.mean()
    a /= np.linalg.norm(a)
    b = rng.normal(size=n - 1)
    b -= b.mean() + (b @ a) * a
    b /= np.linalg.norm(b)
    y = r * a + math.sqrt(1 - r * r) * b
    vl = 5 - 0.05 * np.arange(n) + 0.05 * np.concatenate([[0.0], np.cumsum(a)])
    per = 14 - 0.1 * np.arange(n) + 0.5 * np.concatenate([[0.0], np.cumsum(y)])
    return [{"epoch": e, "val_loss": float(vl[e]), "period/rsi_period_0": float(per[e])} for e in range(n)]


@pytest.mark.parametrize("n, r, band, printed, cls", [
    (20, 0.47, 0.4542, "±0.45", "Δ beyond noise"),   # Fisher-z half-width 0.49: 0.47 was greyed out
    (20, 0.40, 0.4542, "±0.45", "Δ in noise"),
    (8, 0.85, 0.7531, "±0.75", "Δ beyond noise"),    # 7 changes: the z-scale band was ±0.98
])
def test_noise_band_is_on_the_r_scale_it_is_drawn_on(n, r, band, printed, cls):
    """Final check, finding 9: the band is the 95% no-relation half-width of r (tanh of the Fisher-z
    half-width, 0.45 for 19 changes), not the Fisher-z half-width itself (0.49), which greyed out
    bars between the two as 'in noise' and printed the wrong threshold."""
    rows = _rows_with_r_of_changes(n, r)
    s = indicator_summary(rows, Config())
    assert s.loc["rsi_period_0", "r of changes"] == pytest.approx(r, abs=1e-9)
    assert s.attrs["n_changes"] == n - 1
    fig = indicator_evolution(rows, Config())
    rect = next(sh for sh in fig.layout.shapes if sh.type == "rect")
    assert rect.x1 == pytest.approx(band, abs=1e-4) and rect.x0 == pytest.approx(-band, abs=1e-4)
    assert rect.x1 == pytest.approx(S.corr_null_r(n - 1)) == pytest.approx(math.tanh(S.corr_null(n - 1)))
    title = fig.get_subplot(3, 2).xaxis.title.text
    assert f"shaded {printed} = no relation at 95% ({n - 1} epoch-to-epoch changes)" in title
    bars = [t for t in _panel_traces(fig, 3, 2) if t.type == "bar"]
    assert [t.name for t in bars] == [cls]


def test_correlations_are_nan_below_eight_epochs():
    s = indicator_summary(_rows(n=5), Config())
    assert s[["corr with val loss", "r with epoch", "r of changes"]].isna().all().all()
    fig = indicator_evolution(_rows(n=5), Config())
    assert any("needs 8+ epochs" in (a.text or "") for a in fig.layout.annotations)


# ------------------------------------------------------------------ finding 68 / 71: readable change panel
def test_change_panel_is_a_sorted_labelled_bar_chart_sharing_rows_with_the_correlation_panel():
    fig = indicator_evolution(_rows(), Config())
    bars = [t for t in _panel_traces(fig, 3, 1) if t.type == "bar"]
    assert len(bars) == 18 and {b.name for b in bars} == set(NAMES)    # one bar per period, raw id as name
    xs = [b.x[0] for b in bars]
    assert xs == sorted(xs)                                            # sorted by change
    labels = [b.y[0] for b in bars]
    assert "MACD #1 slow" in labels and "RSI #2" in labels and "ma_period_0" not in labels
    assert all("→" in b.text[0] for b in bars)                    # "start -> last" on every bar
    ya1, ya2 = fig.get_subplot(3, 1).yaxis, fig.get_subplot(3, 2).yaxis
    assert list(ya1.categoryarray) == labels == list(ya2.categoryarray)


def test_family_panels_name_each_copy_by_its_start_and_use_a_per_panel_legend():
    fig = indicator_evolution(_rows(), Config())
    ma = [t for t in _panel_traces(fig, 1, 1) if t.showlegend is not False]
    assert [t.name for t in ma] == ["#0: 5", "#1: 10", "#2: 30"]
    assert {t.legend for t in ma} == {"legend2"} and fig.layout.legend2.title.text.startswith("<b>Moving-average")
    macd_keys = {t.name for t in fig.data if t.legend == "legend3" and t.showlegend is not False}
    assert macd_keys == {"#0: 12/26/9", "#1: 5/35/5", "#2: 8/17/9"}
    # the MACD role keys live in the figure-wide legend, so the MACD heading fits a narrow output
    roles = {t.name: t for t in fig.data if (t.name or "").startswith("MACD ") and t.legend in (None, "legend")}
    assert set(roles) == {"MACD fast", "MACD slow", "MACD signal"}
    assert [roles[f"MACD {r}"].line.dash for r in ("fast", "slow", "signal")] == ["solid", "dash", "dashdot"]


def _heading_px(fig, legend_id):
    """Width of a per-panel heading, calibrated on Edge renders: bold 12 px title at ~5.8 px a
    character, then ~36 px of key and gaps plus ~5 px a character per entry."""
    title = fig.layout[legend_id].title.text.replace("<b>", "").replace("</b>", "").strip()
    names = [t.name for t in fig.data if t.legend == legend_id and t.showlegend is not False]
    return 5.8 * len(title) + 20 + sum(36 + 5.0 * len(nm) for nm in names)


def test_headings_fit_a_1100_px_output():
    """At 1100 px the right column leaves ~450 px for a heading: a wider heading pushed the right
    margin (squashing the right column) or ran into the next heading."""
    rows = _rows(drift={"rsi_period_1": 0.0})
    rows = [dict(r, **{"period/rsi_period_1": 14 + (-1) ** e * 0.3 * (1 + e % 3)}) for e, r in enumerate(rows)]
    for r_, e in zip(rows, range(len(rows))):                 # one period tracks val loss: both bar keys show
        r_["val_loss"] = 5 - 0.05 * e + 0.3 * (-1) ** e * (1 + e % 3)
    app = _applied_frame(rows)
    fig = indicator_evolution(rows, Config(), applied=app)
    assert {t.name for t in fig.data if t.legend == "legend6"} == {"Δ in noise", "Δ beyond noise",
                                                                    "levels (shared trend)"}
    for legend_id in ("legend2", "legend3", "legend4", "legend5", "legend6"):
        assert _heading_px(fig, legend_id) < 445, legend_id
    top = [t.name for t in fig.data if t.legend in (None, "legend") and t.showlegend is not False and t.name]
    assert sum(80 + 5.0 * len(nm) for nm in top) < 2 * 1000           # two rows at most above the panels
    assert fig.layout.margin.t >= 160                                  # room for the title, 3 lines and 2 rows


# ------------------------------------------------------------------ finding 69: log axes
def test_family_panels_are_log_scaled_with_explicit_ticks():
    fig = indicator_evolution(_rows(), Config())
    for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
        ya = fig.get_subplot(r, c).yaxis
        assert ya.type == "log" and len(ya.tickvals) >= 2
        assert list(ya.ticktext) == [f"{t:g}" for t in ya.tickvals]
        lo, hi = 10 ** ya.range[0], 10 ** ya.range[1]
        assert all(lo <= t <= hi for t in ya.tickvals)
        assert "log" in ya.title.text


# ------------------------------------------------------------------ finding 70: clip bounds
def test_clip_ceiling_is_drawn_only_where_a_period_nears_it_and_flagged():
    drift = {"macd_1_slow": 0.35 / 20 * 1.0}                  # 35 -> ~58 bars: 97% of the 60-bar ceiling
    rows = _rows(drift=drift, noise=0.0)
    rows[-1]["period/macd_1_slow"] = 58.3
    s = indicator_summary(rows, Config())
    assert s.loc["macd_1_slow", "near bound"] == "ceiling"
    assert s.loc["macd_1_slow", "headroom %"] == pytest.approx(100 * (60 / 58.3 - 1))
    assert (s.drop("macd_1_slow")["near bound"] == "").all()
    fig = indicator_evolution(rows, Config())
    ceil = [t for t in fig.data if t.name == "clip ceiling"]
    macd_y = fig.get_subplot(1, 2).yaxis.plotly_name.replace("axis", "")
    assert len(ceil) == 1 and ceil[0].yaxis == macd_y and list(ceil[0].y) == [60.0, 60.0]
    assert any(t.name == "clip bound (base periods)" for t in fig.data)
    # the flag is part of the ceiling's own label, above the line and ending at the last epoch: no
    # arrow box over the other MACD lines
    note = next(a for a in fig.layout.annotations if "MACD #1 slow ends at 97% of it" in (a.text or ""))
    assert note.text.startswith("clip ceiling 60 bars = lookback") and not note.showarrow
    assert note.x == 20 and note.xanchor == "right" and note.yanchor == "bottom"
    assert note.y == pytest.approx(np.log10(60)) and note.yref == macd_y
    ya = fig.get_subplot(1, 2).yaxis
    assert 10 ** ya.range[1] > 60 * 1.05                    # room above the ceiling for the label
    bar = next(t for t in _panel_traces(fig, 3, 1) if t.name == "macd_1_slow")
    assert bar.text[0] == "35.0 → 58.3 at ceiling"


def test_markers_are_not_cut_at_the_panel_edges():
    fig = indicator_evolution(_rows(), Config())
    for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
        ya = fig.get_subplot(r, c).yaxis
        ys = np.concatenate([np.asarray(t.y, float) for t in _panel_traces(fig, r, c)])
        ys = ys[np.isfinite(ys)]
        assert 10 ** ya.range[0] < ys.min() / 1.03 and 10 ** ya.range[1] > ys.max() * 1.03


# ------------------------------------------------------------------ readability guarantees
def test_colours_by_role_no_dotted_lines_no_empty_panels_and_size_budget():
    fig = indicator_evolution(_rows(n=200), Config())
    assert T.empty_panels(fig) == []
    colours = set()
    for t in fig.data:
        for c in (getattr(t.line, "color", None) if hasattr(t, "line") else None, t.marker.color):
            colours.update(c if isinstance(c, (list, tuple)) else [c])
        if hasattr(t, "line") and t.line.dash is not None:
            assert t.line.dash != "dot", t.name                 # dotted means training everywhere
    assert not colours & {T.HORIZON_COLORS[h] for h in H}
    assert len(fig.to_json()) < 600_000
    text = fig.layout.title.text
    assert "base" in text.lower() and "tanh" in text             # says what the plotted value is


def test_registry_entry_still_takes_the_old_minimal_frame():
    from neural_trade.registries.visualizations import Visualizations

    df = pd.DataFrame({"epoch": [0, 1], "ma_period_0": [5.0, 5.2], "change_ma_period_0": [None, 4.0]})
    fig = Visualizations.build("indicator_evolution", df, Config())
    assert "ma_period_0" in {t.name for t in fig.data}
    notes = [a.text for a in fig.layout.annotations]
    assert "not in this log" in notes                              # the MACD / RSI / BB panels say so


def test_legacy_params_csv(tmp_path):
    rows = _rows(n=10)
    df = pd.DataFrame([{**{k.split("/", 1)[1]: v for k, v in r.items() if k.startswith("period/")},
                        "epoch": r["epoch"], "log_val_loss": r["val_loss"], "change_ma_period_0": 0.0}
                       for r in rows])
    df.to_csv(tmp_path / "indicator_params_history.csv", index=False)
    s = indicator_summary(tmp_path / "indicator_params_history.csv", Config())
    assert len(s) == 18 and s["corr with val loss"].notna().all()
    assert T.empty_panels(indicator_evolution(tmp_path / "indicator_params_history.csv", Config())) == []


# ------------------------------------------------------------------ finding 12: the applied periods
def _tiny_indicator_model(tf, cfg):
    from neural_trade.models.layers.learnable_indicators import LearnableIndicators

    L = tf.keras.layers
    inp = tf.keras.Input(shape=(cfg.LOOKBACK,))
    r = L.Reshape((cfg.LOOKBACK, 1))(inp)
    feat = L.Concatenate()([L.GlobalAveragePooling1D()(r), L.GlobalMaxPooling1D()(r)])
    meta = L.Dense(18, activation="tanh", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0, seed=1),
                   bias_initializer=tf.keras.initializers.Constant(0.1))(feat)
    out = LearnableIndicators(cfg, name="learnable_indicators")([inp, meta])
    return tf.keras.Model(inp, out), tf.keras.Model(inp, meta)


def test_applied_periods_match_the_layer_and_use_the_window_normaliser(tf):
    from neural_trade.data.scaling import WindowNormalizer

    cfg = Config()
    model, meta = _tiny_indicator_model(tf, cfg)
    rng = np.random.default_rng(1)
    raw = 30000 + np.cumsum(rng.normal(0, 20, (64, cfg.LOOKBACK)), axis=1).astype("float32")
    norm = WindowNormalizer("window_relative", 250.0)
    Xn = norm.transform(raw, raw[:, -1])
    app = applied_periods(model, Xn)
    layer = model.get_layer("learnable_indicators")
    adj = tf.constant(meta(Xn).numpy())
    for j, (name, var) in enumerate(zip(layer.get_learned_parameters(), layer.get_indicator_trainable_variables())):
        alpha = layer._alpha(var, adj, j).numpy()                 # what the layer applies to each window
        np.testing.assert_allclose(app[name].to_numpy(), 2.0 / (alpha + 1e-8) - 1.0, rtol=2e-4)
    assert app.attrs["base"] == pytest.approx(layer.get_learned_parameters())
    assert app.shape == (64, 18) and (app.std() > 0).all()        # the period really differs per window

    class _Predictor:                                              # a serving.Predictor: normalises raw windows
        def __init__(self):
            self.model = model
            self.bundle = type("B", (), {"normalizer": norm})()

    via_predictor = applied_periods(_Predictor(), raw, block="test")
    np.testing.assert_allclose(via_predictor.to_numpy(), app.to_numpy(), rtol=1e-5)
    assert via_predictor.attrs["block"] == "test" and via_predictor.attrs["n"] == 64


def _applied_frame(rows, n=500, seed=0):
    rng = np.random.default_rng(seed)
    base = {k.split("/", 1)[1]: v for k, v in rows[-1].items() if k.startswith("period/")}
    app = pd.DataFrame({c: b * np.exp(rng.normal(0.1, 0.15, n)) for c, b in base.items()})
    app.attrs.update(base=base, n=n, block="test", meta_scale=0.5)
    return app


def test_applied_periods_in_the_table_the_evolution_figure_and_their_own_figure():
    rows = _rows(drift={"macd_1_slow": 0.02})                     # 35 -> 49 bars: applied p95 passes 60
    app = _applied_frame(rows)
    s = indicator_summary(rows, Config(), applied=app)
    p50 = float(np.median(app["bb_period_2"]))
    assert s.loc["bb_period_2", "applied p50"] == pytest.approx(p50)
    assert s.loc["bb_period_2", "applied p50 vs base %"] == pytest.approx(
        100 * (p50 / app.attrs["base"]["bb_period_2"] - 1))
    fig = indicator_evolution(rows, Config(), applied=app)
    diamonds = [t for t in fig.data if t.name == "applied median"]
    assert len(diamonds) == 4 and all(min(t.x) > 20.4 for t in diamonds)       # in the strip right of epoch 20
    assert "served weights = epoch 20 (the last)" in fig.layout.title.text
    assert any((t.name or "").startswith("median applied period, epoch 20 weights, 500 test windows")
               for t in fig.data)
    # MACD: one column and one diamond shape per role, keyed with the role's line in the top legend
    macd_x = fig.get_subplot(1, 2).xaxis.plotly_name.replace("axis", "")
    macd = next(t for t in diamonds if t.xaxis == macd_x)
    by_role = {}
    for xv, sym, txt in zip(macd.x, macd.marker.symbol, macd.text):
        by_role.setdefault(txt.split(":")[0].split()[-1], set()).add((xv, sym))
    assert {r: len(v) for r, v in by_role.items()} == {"fast": 1, "slow": 1, "signal": 1}   # 3 copies share each
    cols = {r: next(iter(v)) for r, v in by_role.items()}
    assert cols["fast"][0] < cols["slow"][0] < cols["signal"][0]
    assert [cols[r][1] for r in ("fast", "slow", "signal")] == ["diamond", "diamond-wide", "diamond-tall"]
    keys = {t.name: t.marker.symbol for t in fig.data if (t.name or "").startswith("MACD ")}
    assert keys == {"MACD fast": "diamond", "MACD slow": "diamond-wide", "MACD signal": "diamond-tall"}
    assert any(sh.type == "rect" and sh.x0 > 20 for sh in fig.layout.shapes)       # the shaded strip
    g = indicator_applied_periods(app, Config())
    assert T.empty_panels(g) == [] and len(g.to_json()) < 600_000
    cats = list(g.layout.yaxis.categoryarray)
    assert len(cats) == 18 and cats[-1] == "MA #0" and cats[0] == "BB #2"     # MA on top, BB at the bottom
    base_trace = next(t for t in g.data if (t.name or "").startswith("base period"))
    assert list(base_trace.x) == pytest.approx([app.attrs["base"][c] for c in NAMES], rel=1e-6)
    assert any(sh.x0 == 60 for sh in g.layout.shapes)                      # the lookback line
    # the two whisker keys are drawn at their own widths (constant item sizing drew both 5 px wide)
    assert g.layout.legend.itemsizing == "trace"
    widths = {t.name: t.line.width for t in g.data if t.name in ("middle 50% of windows", "5-95% of windows")}
    assert widths == {"middle 50% of windows": 8, "5-95% of windows": 2}
    assert g.layout.title.text.count("<br>") == 2                  # title, then the subtitle on two lines


def test_served_epoch_is_found_by_matching_the_base_periods():
    rows = _rows()
    app = _applied_frame(rows[:12])                  # base periods of epoch 12 (the served checkpoint)
    fig = indicator_evolution(rows, Config(), applied=app)
    assert "served weights = epoch 12," in fig.layout.title.text
    served = [sh for sh in fig.layout.shapes if sh.type == "line" and sh.x0 == 12]
    assert len(served) == 4 and all(sh.line.dash in (None, "solid") for sh in served)   # not dashed like a bound
    key = next(t for t in fig.data if t.name == "served weights (epoch 12)")
    assert key.line.color == served[0].line.color
    diamonds = [t for t in fig.data if t.name == "applied median"]
    assert diamonds and all(min(t.x) > 20.4 for t in diamonds)       # right of the last epoch, not over the lines
    assert all("served weights (epoch 12)" in txt for t in diamonds for txt in t.text)
    other = _applied_frame(rows)
    other.attrs["base"] = {k: v * 1.3 for k, v in other.attrs["base"].items()}
    fig = indicator_evolution(rows, Config(), applied=other)
    assert "matches no logged epoch" in fig.layout.title.text
    assert not [t for t in fig.data if t.name == "applied median"]
