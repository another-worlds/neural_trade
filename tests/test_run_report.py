"""neural_trade.notebook.run_report: notebook 09's summary and table functions, on a fake run
directory (no training, no CSV, no artifacts) so the tests stay fast."""
from __future__ import annotations

import json

import pytest

from neural_trade.core.config import Config
from neural_trade.notebook import run_report


def _write(run_dir, meta=None, status=None, eval_report=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    Config().to_yaml(run_dir / "config.yaml")
    if meta is not None:
        (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if status is not None:
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    if eval_report is not None:
        (run_dir / "eval_report_dev.json").write_text(json.dumps(eval_report), encoding="utf-8")


META = {
    "run_id": "20260101T000000Z-abc123-deadbeef-default__f-2__s0",
    "engine": {"scenario": "long_360d_stab", "cell_key": "default__f-2__s0", "fold": -2, "seed": 0,
              "commit": "abc123", "strategy": {"name": "calibrated_quantile", "params": {}}},
    "dataset": {"path": "Bitcoin_BTCUSDT.csv"},
    "setup": {"bar_minutes": 1, "LOOKBACK": 60, "HORIZON_STEPS": [10, 15, 20]},
    "blocks": {
        "train": {"n": 100, "first_timestamp": "2024-01-01T00:00:00", "last_timestamp": "2024-06-01T00:00:00"},
        "val": {"n": 10, "first_timestamp": "2024-06-01T00:01:00", "last_timestamp": "2024-06-10T00:00:00"},
        "cal": {"n": 10, "first_timestamp": "2024-06-11T00:00:00", "last_timestamp": "2024-06-20T00:00:00"},
        "test": {"n": 20, "first_timestamp": "2024-06-21T00:00:00", "last_timestamp": "2024-07-01T00:00:00"},
    },
}
STATUS = {"weights_epoch": 4, "weights_val_loss": 5.93, "weights_source": "best validation epoch",
          "sec_per_step": 0.31, "epochs_completed": 10}


EVAL_REPORT = {
    "model": {"horizons": {
        h: {"direction": {"auc": 0.52 + i * 0.01}, "variance": {"crpss": 0.02 + i * 0.01, "coverage90": 0.90 + i * 0.01}}
        for i, h in enumerate(run_report.HORIZONS)}},
    "baselines": {"logreg_lags": {"horizons": {h: {"direction": {"auc": 0.50}} for h in run_report.HORIZONS}}},
    "backtest": {"strategy": "calibrated_quantile",
                "summary": {"total_return": 0.1671, "sharpe_net": 7.82, "max_drawdown": 0.0466, "n_trades": 1968},
                "baselines": {"buy_and_hold": {"total_return": -0.0485}}},
}

MANIFEST = {
    "candidates": {
        "C1": {"rank": 1, "mode": "per_model", "params": {"entry_quantile": 0.9, "size": 1.0},
              "cells": [{"cell": "cell-a", "return": 0.02, "win_share": 0.51, "max_drawdown": 0.03,
                        "sharpe_net": 1.5, "trades": 100, "buy_and_hold": 0.01},
                       {"cell": "cell-b", "return": 0.05, "win_share": 0.53, "max_drawdown": 0.04,
                        "sharpe_net": 2.0, "trades": 120, "buy_and_hold": -0.02}],
              "mean_return": 0.035, "profitable_cells": "2/2", "mean_win_share": 0.52, "max_drawdown": 0.04}
    }
}


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "run"
    _write(d, meta=META, status=STATUS, eval_report=EVAL_REPORT)
    return d


@pytest.fixture()
def manifest_path(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return p


def test_load_config_reads_the_runs_own_yaml(run_dir):
    cfg = run_report.load_config(run_dir)
    assert isinstance(cfg, Config)


def test_blocks_table_has_one_row_per_split_with_dates(run_dir):
    t = run_report.blocks_table(run_dir)
    assert list(t.index) == ["train", "val", "cal", "test"]
    assert t.loc["test", "n bars"] == 20
    assert t.loc["test", "first"] == "2024-06-21T00:00:00"


def test_run_overview_reads_the_setup_and_served_epoch(run_dir):
    info = run_report.run_overview(run_dir)
    assert info["dataset"] == "Bitcoin_BTCUSDT.csv"
    assert info["horizon_steps"] == [10, 15, 20]
    assert info["weights_epoch"] == 4
    assert info["sec_per_step"] == pytest.approx(0.31)
    assert info["scenario"] == "long_360d_stab" and info["fold"] == -2


def test_run_overview_adds_candidate_context_only_with_both_args(run_dir, manifest_path):
    plain = run_report.run_overview(run_dir)
    assert "candidate" not in plain
    with_cand = run_report.run_overview(run_dir, candidate_id="C1", manifest_path=manifest_path)
    assert with_cand["candidate"]["rank"] == 1
    assert with_cand["candidate"]["mean_return"] == pytest.approx(0.035)


def test_overview_markdown_names_the_run_and_the_candidate(run_dir, manifest_path):
    text = run_report.overview_markdown(run_dir, candidate_id="C1", manifest_path=manifest_path)
    assert META["run_id"] in text
    assert "C1" in text and "rank 1" in text
    assert "epoch 4" in text


def test_overview_markdown_without_a_candidate_does_not_mention_one(run_dir):
    text = run_report.overview_markdown(run_dir)
    assert "candidate" not in text.lower()


def test_key_numbers_table_has_model_and_baseline_auc_per_horizon(run_dir):
    t = run_report.key_numbers_table(run_dir)
    assert list(t.index) == list(run_report.HORIZONS)
    assert t.loc["h0", "direction AUC (model)"] == pytest.approx(0.52)
    assert t.loc["h0", "direction AUC (logreg_lags)"] == pytest.approx(0.50)
    assert t.loc["h1", "variance CRPSS vs constant"] == pytest.approx(0.03)
    assert t.loc["h2", "coverage @ 0.90"] == pytest.approx(0.92)


def test_key_numbers_table_missing_report_is_empty_not_an_error(tmp_path):
    d = tmp_path / "bare"
    _write(d, meta=META, status=STATUS)
    t = run_report.key_numbers_table(d)
    assert list(t.index) == list(run_report.HORIZONS)
    assert t["direction AUC (model)"].isna().all()


def test_backtest_headline_reads_the_stored_report(run_dir):
    h = run_report.backtest_headline(run_dir)
    assert h["strategy"] == "calibrated_quantile"
    assert h["total_return"] == pytest.approx(0.1671)
    assert h["n_trades"] == 1968
    assert h["buy_and_hold"] == pytest.approx(-0.0485)


def test_candidate_context_and_cells_table(manifest_path):
    cand = run_report.candidate_context("C1", manifest_path)
    assert cand["rank"] == 1
    assert run_report.candidate_context("C9", manifest_path) is None
    assert run_report.candidate_context("C1", manifest_path.parent / "missing.json") is None

    t = run_report.candidate_cells_table("C1", manifest_path)
    assert list(t.index) == ["cell-a", "cell-b"]
    assert t.loc["cell-b", "sharpe_net"] == pytest.approx(2.0)
    assert t.attrs["mean_return"] == pytest.approx(0.035)


def test_candidate_cells_table_unknown_id_is_empty(manifest_path):
    t = run_report.candidate_cells_table("C9", manifest_path)
    assert t.empty
