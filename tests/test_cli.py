"""neural-trade CLI (plan B16): argument handling, registry/env commands, and a full
train -> predict -> backtest round trip on synthetic bars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_trade.cli import _parse_sets, build_parser, main


def test_set_values_are_parsed_as_yaml():
    assert _parse_sets(["EPOCHS=5", "HORIZON_STEPS=[5, 10, 20]", "LR=1e-3", "CSV_PATH=a.csv"]) == {
        "EPOCHS": 5, "HORIZON_STEPS": [5, 10, 20], "LR": 1e-3, "CSV_PATH": "a.csv"}
    with pytest.raises(SystemExit):
        _parse_sets(["EPOCHS"])


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_env_and_registry_commands(capsys):
    assert main(["env", "--no-devices"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert "python" in json.dumps(env).lower()
    assert main(["registry", "list"]) == 0
    out = capsys.readouterr().out
    for name in ("Models", "Losses", "Metrics", "Callbacks", "Optimizers"):
        assert name.lower() in out.lower()
    assert main(["registry", "list", "optimizers"]) == 0
    assert "adamw" in capsys.readouterr().out
    assert main(["registry", "info", "Optimizers", "adam"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "adam"
    assert main(["registry", "search", "adam"]) == 0
    assert "adam" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["registry", "list", "nonsense"])


@pytest.mark.slow
def test_train_predict_backtest_round_trip(tmp_path, synthetic_bars, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    rc = main(["train", "--csv", str(csv), "--epochs", "1", "--runs-dir", str(tmp_path / "runs"), "--no-calibrate",
               "--set", "MAX_SEQUENCE_COUNT=1500", "--set", "BATCH_SIZE=32",
               "--set", "CALLBACKS=[early_stopping, model_checkpoint]"])
    assert rc == 0
    run_dir = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert (run_dir / "artifacts" / "weights.h5").exists() and (run_dir / "eval_report_test.json").exists()
    meta = json.loads((run_dir / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    assert meta["var_scale"] and meta["var_scale"] > 0

    assert main(["predict", "--artifacts", str(run_dir / "artifacts"), "--csv", str(csv), "--last"]) == 0
    last = json.loads(capsys.readouterr().out)
    assert set(last) == {"h0", "h1", "h2"} and "sigma" in last["h1"]

    out = tmp_path / "bt"
    assert main(["backtest", "--artifacts", str(run_dir / "artifacts"), "--csv", str(csv), "--strategy", "liberal",
                 "--out", str(out), "--plot", "--random-seeds", "5"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["strategy"] == "liberal" and "buy_and_hold_return" in summary
    assert (out / "backtest.json").exists() and (out / "trades.csv").exists() and (out / "backtest.html").exists()


def test_cli_backtest_sets_bar_minutes_from_the_artifacts_config(tmp_path, monkeypatch, capsys):
    """NT-040 (1), strengthened by NT-113 (3): the CLI backtest is a live path, so it must set
    BacktestConfig.bar_minutes from the artifacts' own RESAMPLE_MINUTES, not leave it at the 1.0
    default -- checked by the Sharpe identity itself (annualised == per-bar x sqrt(periods_per_year(5))
    / sqrt(periods_per_year(1))), not only by reading back the ``bar_minutes`` the CLI passed to
    ``build_backtest_config``. No training needed: the Predictor is faked (PredictionBatch built by
    hand), so this stays fast."""
    import neural_trade.serving.predictor as predictor_mod
    from neural_trade.core.config import Config
    from neural_trade.serving.predictor import PredictionBatch
    from neural_trade.strategy.performance import periods_per_year

    rng = np.random.default_rng(0)
    n = 300
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close,
                       "Volume": np.full(n, 1.0)})
    csv = tmp_path / "bars.csv"
    df.to_csv(csv, index=False)

    horizons = ("h0", "h1", "h2")
    batch = PredictionBatch(
        delta={h: rng.normal(0, 50, n) for h in horizons},
        direction_prob={h: rng.uniform(0.3, 0.7, n) for h in horizons},
        direction_prob_calibrated={h: rng.uniform(0.3, 0.7, n) for h in horizons},
        sigma={h: np.full(n, 50.0) for h in horizons},
        variance_scaled={h: np.full(n, 0.25) for h in horizons},
        gauss_up_prob={h: rng.uniform(0.3, 0.7, n) for h in horizons},
        interval={h: (close - 100, close + 100) for h in horizons},
        last_close=close, horizon_steps=(10, 15, 20))
    anchors = np.arange(n)

    def make_predictor(resample_minutes):
        class FakePredictor:
            def __init__(self):
                self.config = Config(RESAMPLE_MINUTES=resample_minutes)

                class _Bundle:
                    pred_scale, pred_mean = 100.0, 0.0
                    meta = {"var_scale": 1.0, "weighted_direction_quantiles": None}
                self.bundle = _Bundle()

            def predict_windows_frame(self, raw_df, batch_size=None):
                return batch, df, anchors
        return FakePredictor()

    captured = {}
    real_build = __import__("neural_trade.strategy.params", fromlist=["build_backtest_config"]).build_backtest_config

    def spy(params=None, **kw):
        captured["params"] = dict(params or {})
        return real_build(params, **kw)

    monkeypatch.setattr("neural_trade.strategy.build_backtest_config", spy)

    def run(resample_minutes):
        monkeypatch.setattr(predictor_mod.Predictor, "from_artifacts",
                            classmethod(lambda cls, d, rm=resample_minutes: make_predictor(rm)))
        rc = main(["backtest", "--artifacts", str(tmp_path / "artifacts"), "--csv", str(csv),
                   "--strategy", "liberal", "--random-seeds", "0"])
        assert rc == 0
        return json.loads(capsys.readouterr().out)

    five = run(5)
    assert captured["params"]["bar_minutes"] == 5.0
    one = run(1)
    assert captured["params"]["bar_minutes"] == 1.0
    assert one["n_trades"] > 0, "the Sharpe identity needs a strategy that actually trades"
    assert five["sharpe_net"] == pytest.approx(
        one["sharpe_net"] * (periods_per_year(5.0) / periods_per_year(1.0)) ** 0.5, rel=1e-9)


def test_registry_listing_includes_the_repository_plugins(capsys, monkeypatch):
    """Definition of done: plugins/examples/echo_metric.py appears in `registry list metrics`."""
    import sys

    from neural_trade.registries.metrics import Metrics

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    try:
        assert main(["registry", "list", "metrics"]) == 0
        assert "n_samples" in capsys.readouterr().out
    finally:
        if Metrics.has("n_samples"):
            Metrics.remove("n_samples")
        for name in [m for m in sys.modules if m.startswith("neural_trade_plugins")]:
            del sys.modules[name]
