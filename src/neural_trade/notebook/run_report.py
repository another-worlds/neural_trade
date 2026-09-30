"""Notebook 09: one run's summary, training record, fit evaluation and backtest, so a saved run can
be reviewed without retraining or touching the training CSV (dataset D-041 / RUNBOOK "Long runs").

Everything here reads only what a scored cell writes to its run directory: ``meta.json``,
``status.json``, ``config.yaml``, ``metrics.jsonl``, ``eval_report_<role>.json`` and the saved
prediction blocks (``predictions_oos.npz``, ``predictions_cal.npz``, written by
``experiments.scorer.save_predictions``). The backtest is refitted at the notebook's own strategy,
knobs and costs (:func:`load_saved_blocks` feeds :class:`neural_trade.notebook.BacktestExplorer`,
which fits on the calibration block and runs the out-of-sample block, matching
``experiments.scorer.fit_and_backtest``'s BlockSignals/var_scale recipe), so it never touches the
big CSV either.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

HORIZONS = ("h0", "h1", "h2")
ZERO_COST_PARAMS: Dict[str, Any] = {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0, "random_seeds": 20}
DEFAULT_STRATEGY = "calibrated_quantile"
DEFAULT_STRATEGY_PARAMS: Dict[str, Any] = {"entry_quantile": 0.9}


def _read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_config(run_dir):
    """The run's own :class:`Config` (``config.yaml``); no CSV is read."""
    from neural_trade.core.config import Config

    return Config.from_yaml(Path(run_dir) / "config.yaml")


def load_saved_blocks(run_dir) -> Dict[str, Any]:
    """``run_dir``'s saved blocks as :class:`neural_trade.notebook.BacktestExplorer` expects them
    (the same shape ``backtest_ui.load_run_blocks`` builds from a fresh inference pass), built
    instead from the cell's saved predictions: no CSV, no re-inference. ``predictor`` only serves its
    ``var_scale`` (fitted once at train time on the same calibration predictions
    ``experiments.scorer.BlockSignals.build`` would recompute, so the two agree)."""
    from neural_trade.experiments.scorer import load_block
    from neural_trade.serving.predictor import Predictor

    run_dir = Path(run_dir)
    predictor = Predictor.from_artifacts(run_dir / "artifacts")
    test, bars, extra = load_block(run_dir / "predictions_oos.npz")
    cal, _, _ = load_block(run_dir / "predictions_cal.npz")
    return {"run_dir": run_dir, "predictor": predictor, "config": predictor.config.copy(), "test": test, "cal": cal,
            "bars": bars, "times": extra.get("anchor_timestamp")}


def _block_row(name: str, block: Optional[dict]) -> Dict[str, Any]:
    if not block:
        return {"block": name}
    return {"block": name, "n bars": block.get("n"), "first": block.get("first_timestamp"),
            "last": block.get("last_timestamp")}


def blocks_table(run_dir) -> pd.DataFrame:
    """One row per split (train / val / cal / test) with its bar count and date range, from ``meta.json``."""
    meta = _read_json(Path(run_dir) / "meta.json") or {}
    blocks = meta.get("blocks") or {}
    rows = [_block_row(name, blocks.get(name)) for name in ("train", "val", "cal", "test")]
    return pd.DataFrame(rows).set_index("block")


def candidate_context(candidate_id: str, manifest_path) -> Optional[Dict[str, Any]]:
    """The candidate's entry in ``configs/candidates/manifest.json`` (rank, mode, params and the
    six-cell summary), or ``None`` when the id or the file is not there."""
    manifest = _read_json(Path(manifest_path))
    if not manifest:
        return None
    return (manifest.get("candidates") or {}).get(candidate_id)


def candidate_cells_table(candidate_id: str, manifest_path) -> pd.DataFrame:
    """The candidate's own cells (return, win share, drawdown, Sharpe, trades, buy-and-hold) from
    ``configs/candidates/manifest.json``, for context next to this run's own numbers (D-020: the
    candidates are already-verified evidence, not something the notebook recomputes)."""
    cand = candidate_context(candidate_id, manifest_path)
    if not cand:
        return pd.DataFrame()
    df = pd.DataFrame(cand["cells"]).set_index("cell")
    df.attrs["mean_return"] = cand.get("mean_return")
    df.attrs["profitable_cells"] = cand.get("profitable_cells")
    df.attrs["mean_win_share"] = cand.get("mean_win_share")
    df.attrs["max_drawdown"] = cand.get("max_drawdown")
    return df


def run_overview(run_dir, *, candidate_id: Optional[str] = None, manifest_path=None) -> Dict[str, Any]:
    """What the run is: setup, blocks, the served epoch and training speed, and (with
    ``candidate_id`` / ``manifest_path``) which saved candidate it is a cell of."""
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "meta.json") or {}
    status = _read_json(run_dir / "status.json") or {}
    cfg = load_config(run_dir)
    setup = meta.get("setup") or {}
    dataset = meta.get("dataset") or {}
    engine = meta.get("engine") or {}
    out = {
        "run_id": meta.get("run_id") or run_dir.name,
        "dataset": dataset.get("path") or cfg.CSV_PATH,
        "bar_minutes": setup.get("bar_minutes", cfg.RESAMPLE_MINUTES),
        "lookback_bars": setup.get("LOOKBACK", cfg.LOOKBACK),
        "horizon_steps": setup.get("HORIZON_STEPS", list(cfg.HORIZON_STEPS)),
        "scenario": engine.get("scenario"),
        "cell": engine.get("cell_key"),
        "fold": engine.get("fold"),
        "seed": engine.get("seed"),
        "commit": engine.get("commit"),
        "strategy": (engine.get("strategy") or {}).get("name"),
        "weights_epoch": status.get("weights_epoch"),
        "weights_val_loss": status.get("weights_val_loss"),
        "weights_source": status.get("weights_source"),
        "sec_per_step": status.get("sec_per_step"),
        "epochs_completed": status.get("epochs_completed"),
    }
    if candidate_id and manifest_path is not None:
        cand = candidate_context(candidate_id, manifest_path)
        out["candidate_id"] = candidate_id
        out["candidate"] = cand
    return out


def overview_markdown(run_dir, *, candidate_id: Optional[str] = None, manifest_path=None) -> str:
    """:func:`run_overview` and :func:`blocks_table` as a markdown block for ``IPython.display.Markdown``."""
    info = run_overview(run_dir, candidate_id=candidate_id, manifest_path=manifest_path)
    steps = "/".join(f"{s}m" for s in info["horizon_steps"])
    lines = [f"**{info['run_id']}**  (commit `{info['commit']}`, scenario `{info['scenario']}`, "
            f"cell `{info['cell']}`, fold {info['fold']}, seed {info['seed']})",
            "",
            f"- dataset `{info['dataset']}`, {info['bar_minutes']}-minute bars, "
            f"{info['lookback_bars']}-bar window, horizons {steps}",
            f"- served weights: epoch {info['weights_epoch']} (val loss {info['weights_val_loss']:.4f}, "
            f"{info['weights_source']})" if info.get("weights_val_loss") is not None else
            f"- served weights: epoch {info['weights_epoch']}",
            f"- {info['epochs_completed']} epochs trained, {info['sec_per_step']:.4f} sec/step"
            if info.get("sec_per_step") is not None else "- training speed not recorded",
            f"- default strategy `{info['strategy']}`"]
    cand = info.get("candidate")
    if cand:
        lines.append(f"- candidate `{info['candidate_id']}` (rank {cand.get('rank')}, {cand.get('mode')}, "
                     f"params {cand.get('params')}): mean return {cand.get('mean_return'):+.2%} over "
                     f"{cand.get('profitable_cells')} profitable cells, mean win share "
                     f"{cand.get('mean_win_share'):.1%}, max drawdown {cand.get('max_drawdown'):.2%}")
    return "\n".join(lines)


def key_numbers_table(run_dir, *, role: str = "dev", baseline: str = "logreg_lags") -> pd.DataFrame:
    """Per horizon: direction AUC (model vs ``baseline``), variance CRPSS against constant variance,
    and 0.90-target conformal coverage, from ``eval_report_<role>.json`` (D-020: dev-block numbers,
    already scored; nothing here is recomputed)."""
    report = _read_json(Path(run_dir) / f"eval_report_{role}.json") or {}
    model = report.get("model", {}).get("horizons", {})
    base = (report.get("baselines", {}).get(baseline) or {}).get("horizons", {})
    rows = {}
    for h in HORIZONS:
        m, b = model.get(h, {}), base.get(h, {})
        rows[h] = {
            "direction AUC (model)": (m.get("direction") or {}).get("auc"),
            f"direction AUC ({baseline})": (b.get("direction") or {}).get("auc"),
            "variance CRPSS vs constant": (m.get("variance") or {}).get("crpss"),
            "coverage @ 0.90": (m.get("variance") or {}).get("coverage90"),
        }
    return pd.DataFrame(rows).T


def backtest_headline(run_dir, *, role: str = "dev") -> Dict[str, Optional[float]]:
    """The run's own scored backtest headline (return, Sharpe, drawdown, trades, buy-and-hold) from
    ``eval_report_<role>.json``'s ``backtest`` block, for a quick summary line; the notebook's own
    backtest cell (zero cost, D-044) is the number that counts (this run's stored report may predate
    the zero-cost default)."""
    report = _read_json(Path(run_dir) / f"eval_report_{role}.json") or {}
    bt = report.get("backtest") or {}
    summary = bt.get("summary") or {}
    bh = (bt.get("baselines") or {}).get("buy_and_hold") or {}
    return {"strategy": bt.get("strategy"), "total_return": summary.get("total_return"),
            "sharpe_net": summary.get("sharpe_net"), "max_drawdown": summary.get("max_drawdown"),
            "n_trades": summary.get("n_trades"), "buy_and_hold": bh.get("total_return")}


__all__ = ["DEFAULT_STRATEGY", "DEFAULT_STRATEGY_PARAMS", "ZERO_COST_PARAMS", "backtest_headline",
           "blocks_table", "candidate_cells_table", "candidate_context", "key_numbers_table", "load_config",
           "load_saved_blocks", "overview_markdown", "run_overview"]
