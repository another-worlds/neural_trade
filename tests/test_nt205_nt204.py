"""NT-205 (the calibration explorer compares the refit with the saved pipeline by temperatures and signal states,
not by settings alone) and NT-204 (the no-signal marker in predict_last / predict_frame / the CLI predict output;
the online calibrator's warm start is clamped). D-066."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_trade.calibration.online_calibrator import OnlineTemperatureCalibrator
from neural_trade.core.config import Config
from neural_trade.evaluation.frame import HORIZONS
from neural_trade.notebook.calibration_ui import CalibrationExplorer
from neural_trade.serving import predictor as pm
from tests.test_direction_signal_surfaces import _explorer, _explorer_with_saved

ALL_OK = {h: "ok" for h in HORIZONS}


def _stale_saved_explorer(saved_kind, refit_kind):
    """The saved pipeline (and the served frames) come from `saved_kind` data, the refit runs on `refit_kind`
    data: the same settings, different fitted temperatures / signal states (a run calibrated before NT-124)."""
    saved = _explorer_with_saved(saved_kind)
    other = _explorer(refit_kind)
    blocks = {**saved.blocks, "cal_raw": other.blocks["cal_raw"], "test_raw": other.blocks["test_raw"]}
    return CalibrationExplorer(blocks)


def _names(fig):
    return [t.name for t in fig.data]


# ---------------------------------------------------------------- NT-205
def test_refit_with_same_settings_but_other_signal_is_not_said_to_reproduce_the_saved_pipeline():
    ex = _stale_saved_explorer(ALL_OK, {"h0": "ok", "h1": "upper", "h2": "ok"})
    ex.refit("none", shrink_delta=False)                       # = the saved settings
    assert ex.matches_saved() and not ex.reproduces_saved()
    assert ex.pipeline.direction_signal()["h1"] == "none" and ex.saved.direction_signal()["h1"] == "ok"
    text = ex._status()
    assert "reproduces" not in text and "differs from the saved pipeline" in text


def test_saved_curve_is_drawn_when_the_refit_differs_even_with_the_saved_settings():
    ex = _stale_saved_explorer(ALL_OK, {"h0": "ok", "h1": "upper", "h2": "ok"})
    ex.refit("none", shrink_delta=False)
    rel, _cov = ex.figures("h1")
    assert "saved P(up)" in _names(rel)


def test_a_refit_that_reproduces_the_saved_pipeline_says_so_and_draws_no_saved_curve():
    ex = _explorer_with_saved(ALL_OK)
    ex.refit("none", shrink_delta=False)
    assert ex.reproduces_saved()
    assert "reproduces the served pipeline" in ex._status()
    assert "saved P(up)" not in _names(ex.figures("h1")[0])


def test_equal_signal_states_but_other_temperatures_do_not_reproduce():
    ex = _explorer_with_saved(ALL_OK)
    ex.refit("none", shrink_delta=False)
    assert ex.reproduces_saved()
    ex.saved.temperature_scaler.temperatures["h1"] *= 1.5          # same settings, same signal, another T
    assert ex.saved.direction_signal() == ex.pipeline.direction_signal() and not ex.reproduces_saved()


def test_equal_temperatures_but_other_signal_states_do_not_reproduce():
    ex = _explorer_with_saved(ALL_OK)
    ex.refit("none", shrink_delta=False)
    ex.saved.direction_signal = lambda: {"h0": "ok", "h1": "none", "h2": "ok"}   # same T, another state
    assert not ex.reproduces_saved()


def test_changed_settings_keep_the_old_saved_wording():
    ex = _explorer_with_saved(ALL_OK)
    ex.refit("sigma", shrink_delta=True, alpha=0.2)
    assert not ex.matches_saved() and not ex.reproduces_saved()
    assert "drawn dash-dotted" in ex._status() and "reproduces" not in ex._status()


def test_saved_branch_of_ece_na_names_the_saved_curve_on_a_no_signal_saved_horizon():
    """The saved pipeline's h1 is 'none', the refit's is 'ok': only the saved ECE is n/a in the subtitle."""
    ex = _stale_saved_explorer({"h0": "ok", "h1": "upper", "h2": "ok"}, ALL_OK)
    ex.refit("none", shrink_delta=False)
    assert ex.saved.direction_signal()["h1"] == "none" and ex.pipeline.direction_signal()["h1"] == "ok"
    sub = ex.figures("h1")[0].layout.title.text
    assert "saved n/a (no direction signal)" in sub
    assert "calibrated n/a" not in sub
    sub0 = ex.figures("h0")[0].layout.title.text
    assert "n/a" not in sub0


# ---------------------------------------------------------------- NT-204
def _predictor(monkeypatch, signal, n=3):
    cfg = Config()
    preds = {k: {h: np.linspace(0.4, 0.6, n) for h in HORIZONS} for k in ("delta", "direction_prob", "variance")}

    class Pipe:
        def apply(self, p, alpha=0.1, windows=None):
            return {"direction_prob": p["direction_prob"], "delta": p["delta"],
                    "intervals": {h: (np.zeros(n) - 1, np.ones(n)) for h in HORIZONS}}

        def direction_signal(self):
            return dict(signal)

    bundle = SimpleNamespace(config=cfg, pred_scale=50.0, pred_mean=0.0,
                             calibration_pipeline=Pipe() if signal is not None else None,
                             normalizer=SimpleNamespace(transform=lambda X, lc: np.asarray(X, "float32")),
                             build_model=lambda: None)
    monkeypatch.setattr(pm, "heads_to_predictions", lambda *a, **k: preds)
    monkeypatch.setattr("neural_trade.utils.seeding.set_arithmetic_rewrite", lambda c: None)
    pr = pm.Predictor(bundle, model=SimpleNamespace(predict=lambda ds, verbose=0: None))
    series = tuple(cfg.INPUT_SERIES or ["close"])
    shape = (n, cfg.LOOKBACK) if len(series) == 1 else (n, cfg.LOOKBACK, len(series))
    return pr, np.full(shape, 100.0, "float32")


def test_batch_frame_has_a_direction_signal_column_per_horizon(monkeypatch):
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}
    pr, X = _predictor(monkeypatch, sig)
    fr = pr.predict(X).to_frame()
    assert {h: fr[f"{h}_direction_signal"].iloc[0] for h in HORIZONS} == sig
    pr0, X0 = _predictor(monkeypatch, None)
    assert not any(c.endswith("_direction_signal") for c in pr0.predict(X0).to_frame().columns)


def test_predict_last_carries_direction_signal_per_horizon(monkeypatch):
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}
    pr, X = _predictor(monkeypatch, sig, n=1)
    pr.config.INPUT_SERIES = ["close"]
    out = pr.predict_last(X[0, :] if X.ndim == 2 else X[0, :, 0])
    assert {h: out[h]["direction_signal"] for h in HORIZONS} == sig
    assert isinstance(out["h1"]["p_up_calibrated"], float) and "direction_signal" not in {
        k for k in out["h1"] if k.startswith("direction_signal_")}
    json.dumps(out)
    pr0, X0 = _predictor(monkeypatch, None, n=1)
    pr0.config.INPUT_SERIES = ["close"]
    out0 = pr0.predict_last(X0[0, :] if X0.ndim == 2 else X0[0, :, 0])
    assert {out0[h]["direction_signal"] for h in HORIZONS} == {"n/a"}


def test_predict_frame_carries_the_signal_columns(monkeypatch, synthetic_bars):
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}
    pr, _ = _predictor(monkeypatch, sig)

    def predict(X, lc=None, alpha=0.1, batch_size=None, **k):
        n = len(X)
        d = {h: np.zeros(n) for h in HORIZONS}
        return pm.PredictionBatch(d, d, d, d, d, d, {h: (np.zeros(n), np.ones(n)) for h in HORIZONS},
                                  np.asarray(lc), (10, 15, 20), None, sig)

    monkeypatch.setattr(pr, "predict", predict)
    fr = pr.predict_frame(synthetic_bars)
    assert len(fr) > 1
    assert set(fr["h1_direction_signal"]) == {"none"} and set(fr["h0_direction_signal"]) == {"ok"}


def _cli_predictor(signal):
    n = 2
    cols = {f"{h}_p_up_calibrated": np.full(n, 0.5) for h in HORIZONS}
    cols |= {f"{h}_direction_signal": signal[h] for h in HORIZONS}
    import pandas as pd

    frame = pd.DataFrame(cols)
    last = {h: {"p_up_calibrated": 0.5, "direction_signal": signal[h]} for h in HORIZONS}
    return SimpleNamespace(config=SimpleNamespace(input_series=lambda: ("close",)),
                           predict_frame=lambda df, alpha=0.1, batch_size=None: frame,
                           predict_last=lambda c, alpha=0.1: last)


def _run_predict(monkeypatch, tmp_path, signal, **kw):
    from neural_trade import cli

    csv = tmp_path / "bars.csv"
    csv.write_text("close\n1\n2\n")
    monkeypatch.setattr("neural_trade.serving.predictor.Predictor.from_artifacts",
                        classmethod(lambda cls, p: _cli_predictor(signal)))
    args = argparse.Namespace(artifacts="x", csv=str(csv), last=kw.get("last", False), alpha=0.1, batch_size=None,
                              out=kw.get("out"), tail=5)
    return cli.cmd_predict(args)


def test_cli_predict_warns_for_a_no_signal_horizon_and_prints_the_column(monkeypatch, tmp_path, capsys):
    from neural_trade import cli

    warned = []
    monkeypatch.setattr(cli.logger, "warning", lambda msg, *a: warned.append(msg % a))
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}
    assert _run_predict(monkeypatch, tmp_path, sig) == 0
    printed = capsys.readouterr().out
    assert "h1_direction_signal" in printed and "none" in printed
    assert len(warned) == 1 and "no usable direction signal for h1:" in warned[0]
    warned.clear()
    _run_predict(monkeypatch, tmp_path, ALL_OK)
    assert warned == []


def test_cli_predict_last_prints_the_signal(monkeypatch, tmp_path, capsys):
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}
    assert _run_predict(monkeypatch, tmp_path, sig, last=True) == 0
    assert json.loads(capsys.readouterr().out)["h1"]["direction_signal"] == "none"


def test_online_calibrator_warm_start_is_clamped_to_its_range():
    c = OnlineTemperatureCalibrator(temperatures={"h0": 1000.0, "h1": 0.01, "h2": 2.0})
    assert c.state["h0"]["T"] == c.state["h0"]["T_ema"] == c.T_max == 10.0
    assert c.state["h1"]["T"] == c.state["h1"]["T_ema"] == c.T_min == 0.1
    assert c.state["h2"]["T"] == 2.0
    assert c.calibrate(0.9, "h0") == pytest.approx(1 / (1 + np.exp(-np.log(9) / 10.0)))
    c.update(0.9, 1, "h0")                                  # the first update no longer jumps from outside the range
    assert c.T_min <= c.state["h0"]["T"] <= c.T_max
    wide = OnlineTemperatureCalibrator(temperatures={"h0": 1000.0}, T_max=2000.0)
    assert wide.state["h0"]["T"] == 1000.0
