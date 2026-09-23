"""neural-trade CLI (plan B16): argument handling, registry/env commands, and a full
train -> predict -> backtest round trip on synthetic bars."""
from __future__ import annotations

import json
from pathlib import Path

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
