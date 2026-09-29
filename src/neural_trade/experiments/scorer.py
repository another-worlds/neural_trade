"""The experiment engine's one scorer (NT-026): every run's out-of-sample block, scored the same way.

For a trained run of fold f (a TrainResult of train_and_evaluate on FOLD_INDEX f):

* **The report.** The fold's out-of-sample block (the test block of its purged split, D-005) is
  scored by :func:`neural_trade.evaluation.report.evaluate`, with the baselines fit on the fold's
  train block and the confidence threshold from its calibration block. The report is labelled by
  the fold's role: ``dev`` (an earlier fold: rows rank on these) or ``test`` (the latest fold:
  shown, never used to rank or choose; D-020). It is written as ``eval_report_<role>.json`` / ``.md``.
* **The backtest.** The scenario's strategy (default ``Strategies.default``, calibrated_quantile,
  D-009) trades the same block through :func:`neural_trade.strategy.backtest.backtest`: decisions at
  a bar's close, fills at the next bar's open, stops on high/low, the cost profile of
  ``BacktestConfig`` (10 + 1 + 2 = 13 bps per side unless the scenario sets it) and the annualisation
  of the run's bar size. Its knobs are fitted on the fold's CALIBRATION block only: the confidence
  scale (``var_scale_from(cal)``) and, for a strategy with ``from_calibration``, its entry lines.
  The result carries buy-and-hold, always-flat and the size-matched random null
  (``random_same_frequency``: the strategy's trade rate, holding time and mean position size).

``scores`` (what the run store indexes) are the report's flat metrics (``EvalReport.flat()``: per
horizon direction / delta / variance, coherence, ``backtest/<summary>``), the backtest baselines
(``backtest/<baseline>/<key>``), the fitted baselines' ranked metrics (``baseline/<name>/<h>/...``)
and the training facts (``train/...``). The leaderboard's ranking number is ``backtest/sharpe_net``
(net Sharpe after costs) on dev rows.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

ROLES = ("dev", "test")


class ScoringError(RuntimeError):
    """A run cannot be scored honestly (for example: no calibration-block predictions to fit on)."""


@dataclass
class Scored:
    role: str
    report: Any                     # evaluation.report.EvalReport
    backtest: Any                   # strategy.backtest.BacktestResult
    strategy: Any                   # the fitted Strategy instance
    scores: Dict[str, Optional[float]]
    paths: Dict[str, Path] = field(default_factory=dict)


def _numbers(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[prefix + k] = float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            v = float(v)
            out[prefix + k] = None if math.isnan(v) else v
    return out


def _strategy_params(strategy) -> Dict[str, Any]:
    if dataclasses.is_dataclass(strategy):
        return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in dataclasses.asdict(strategy).items()}
    return {}


def training_facts(result) -> Dict[str, Optional[float]]:
    """Deterministic facts of the training run (timings stay out of the scores)."""
    hist = getattr(getattr(result, "history", None), "history", None) or {}
    facts = {"train/epochs_run": len(hist.get("loss", []) or []),
             "train/weights_epoch": getattr(result, "weights_epoch", None),
             "train/weights_val_loss": getattr(result, "weights_val_loss", None)}
    return {k: v for k, v in _numbers({k: v for k, v in facts.items() if v is not None}).items()}


def leaderboard_scores(report, backtest_result, result=None) -> Dict[str, Optional[float]]:
    """The flat scores of one scored run (see the module docstring for the key families)."""
    from neural_trade.evaluation.baselines import RELEVANT
    from neural_trade.evaluation.report import HIGHER_IS_BETTER, LOWER_IS_BETTER

    out = _numbers(report.flat())
    for name, summary in (backtest_result.baselines or {}).items():
        out.update(_numbers(summary or {}, f"backtest/{name}/"))
    ranked = HIGHER_IS_BETTER | LOWER_IS_BETTER
    for name, scored in (report.baselines or {}).items():
        for h, row in (scored.get("horizons") or {}).items():
            for group in RELEVANT.get(name, ()):
                vals = {k: v for k, v in (row.get(group) or {}).items() if k in ranked}
                out.update(_numbers(vals, f"baseline/{name}/{h}/{group}/"))
    if result is not None:
        out.update(training_facts(result))
    return dict(sorted(out.items()))


def _write_new(path: Path, text: str) -> Path:
    """Write a file that must not exist yet (the engine never overwrites a run's files)."""
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def _pct(v) -> str:
    return "n/a" if v is None or not math.isfinite(float(v)) else f"{100.0 * float(v):+.2f}%"


def _num(v, fmt="{:+.2f}") -> str:
    return "n/a" if v is None or not math.isfinite(float(v)) else fmt.format(float(v))


def engine_markdown(report, bt: Dict[str, Any]) -> str:
    """The section the engine adds to the report's markdown: role, fold, the fitted knobs and the
    backtest against buy-and-hold, always-flat and the size-matched random null."""
    meta = report.meta or {}
    role = meta.get("role", report.split)
    ranks = "rows rank on it" if role == "dev" else "shown, never used to rank or choose (D-020)"
    s = bt.get("summary") or {}
    base = bt.get("baselines") or {}
    bh, flat, rnd = base.get("buy_and_hold") or {}, base.get("always_flat") or {}, base.get("random_same_freq") or {}
    cfg = bt.get("config") or {}
    params = bt.get("params") or {}
    knobs = ", ".join(f"{k} {params[k]:.4f}" for k in ("long_above", "short_below", "median")
                      if isinstance(params.get(k), (int, float)))
    blocks = meta.get("blocks") or {}
    oos = blocks.get("test") or {}
    L = ["", "## Experiment engine: out-of-sample block and backtest", "",
         f"Role: **{role}** ({ranks}). Fold {meta.get('fold')} (TimeSeriesSplit fold {meta.get('fold_id')}, "
         f"{meta.get('n_usable_folds')} usable folds); this report scores the fold's out-of-sample block"
         + (f": {oos.get('n')} sequences, {oos.get('first_timestamp')} .. {oos.get('last_timestamp')}." if oos else "."),
         "",
         f"Strategy `{bt.get('strategy')}`, knobs fitted on the calibration block only: var_scale "
         f"{_num(bt.get('var_scale'), '{:.4g}')}" + (f", {knobs}" if knobs else "") + ". Costs per side: fee "
         f"{cfg.get('fee_bps')} bps + half-spread {cfg.get('half_spread_bps')} bps + slippage {cfg.get('slippage_bps')} "
         f"bps; fills at the next bar's open ({cfg.get('fill')}); stops on {cfg.get('tp_sl_on')}; Sharpe annualised "
         f"over {cfg.get('minutes_per_year')} minutes per year at {cfg.get('bar_minutes')}-minute bars.", "",
         "| | net return | net Sharpe | max drawdown | trades |", "|---|---|---|---|---|",
         f"| strategy | {_pct(s.get('total_return'))} | {_num(s.get('sharpe_net'))} | {_pct(s.get('max_drawdown'))} | "
         f"{s.get('n_trades', 'n/a')} |",
         f"| buy and hold | {_pct(bh.get('total_return'))} | {_num(bh.get('sharpe_net'))} | "
         f"{_pct(bh.get('max_drawdown'))} | {bh.get('n_trades', 'n/a')} |",
         f"| always flat | {_pct(flat.get('total_return'))} | {_num(flat.get('sharpe_net'))} | "
         f"{_pct(flat.get('max_drawdown'))} | {flat.get('n_trades', 'n/a')} |"]
    if rnd.get("n_seeds"):
        L += [f"| random null, mean of {rnd['n_seeds']} seeds (p05 .. p95 of the net return: "
              f"{_pct(rnd.get('random_p05_total_return'))} .. {_pct(rnd.get('random_p95_total_return'))}) | "
              f"{_pct(rnd.get('random_mean_total_return'))} | {_num(rnd.get('random_mean_sharpe_net'))} | | |", "",
              f"The random null enters at the strategy's rate ({_num(rnd.get('trade_rate'), '{:.4f}')} per flat bar), "
              f"holds {rnd.get('hold_bars')} bars and sizes each entry at the strategy's mean position size "
              f"({_num(rnd.get('size_frac'), '{:.3f}')}). The strategy beats "
              f"{_num(rnd.get('percentile_total_return'), '{:.0f}')}% of its seeds on net return, "
              f"{_num(rnd.get('percentile_sharpe_net'), '{:.0f}')}% on net Sharpe and "
              f"{_num(rnd.get('percentile_gross_return'), '{:.0f}')}% on gross return."]
    else:
        L += ["", "No random null: the strategy made no trade (or the scenario set random_seeds to 0)."]
    return "\n".join(L) + "\n"


def score_result(result, *, role: str, strategy: Optional[str] = None,
                 strategy_params: Optional[Mapping[str, Any]] = None,
                 backtest_params: Optional[Mapping[str, Any]] = None, run_id: Optional[str] = None,
                 out_dir=None, meta: Optional[Mapping[str, Any]] = None, arrays=None) -> Scored:
    """Score a TrainResult's out-of-sample block (see the module docstring).

    ``role``: "dev" or "test" (the fold's role in the scenario). ``meta`` is added to the report's
    meta (fold, blocks, scenario, cell). ``arrays``: ``data.processor.split_arrays(result.config)``
    when the caller has it. With ``out_dir`` the report is written there (files that must not exist).
    """
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import evaluate
    from neural_trade.strategy import (Bars, SignalFrame, Strategies, backtest, build_backtest_config,
                                       build_strategy, var_scale_from)

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    cfg = result.config
    arrays = arrays if arrays is not None else split_arrays(cfg)
    test_block, train = arrays["test"], arrays["train"]
    frame = PredictionFrame.from_result(result, "test", X_raw=test_block["X"])
    if len(frame) != len(test_block["y"]) or not np.allclose(frame.last_close, test_block["last_close"], rtol=1e-6):
        raise ScoringError(f"the run's test predictions ({len(frame)}) do not line up with FOLD_INDEX "
                           f"{cfg.FOLD_INDEX}'s rebuilt out-of-sample block ({len(test_block['y'])})")
    if getattr(result, "predictions_cal", None) is None:
        raise ScoringError("the run has no calibration-block predictions: the strategy's knobs are fitted on the "
                           "calibration block only (train with fit_calibration=True)")
    cal = PredictionFrame.from_result(result, "cal")
    baselines = BaselineSet.fit(train["X"], train["y"], train["last_close"], float(cfg.DIR_DEADBAND_BPS))

    var_scale = var_scale_from(cal)
    cal_signals = SignalFrame.build(cal, var_scale)
    signals = SignalFrame.build(frame, var_scale)
    name = strategy or Strategies.default
    strat = build_strategy(name, strategy_params, calibration=cal_signals)
    bcfg = build_backtest_config({**dict(backtest_params or {}), "bar_minutes": float(cfg.RESAMPLE_MINUTES)})
    bars = Bars.from_frame(arrays["df"], test_block["anchor_bar"])
    if len(bars) != len(frame) or not np.allclose(bars.close, frame.last_close, rtol=1e-6):
        raise ScoringError(f"the out-of-sample bars ({len(bars)}) do not line up with its predictions ({len(frame)})")
    res = backtest(signals, bars, strat, bcfg)
    bt = res.to_dict()
    bt.update(params=_strategy_params(strat), fitted_on="cal", var_scale=float(var_scale), n_bars=len(bars),
              calibrated_probabilities=frame.direction_prob_calibrated is not None)

    frame.split = role
    report = evaluate(frame, cfg, baselines=baselines, cal_frame=cal, run_id=run_id, backtest=bt)
    report.meta.update({"role": role, "ranks": role == "dev", "block": "out-of-sample (the fold's test block)",
                        **dict(meta or {})})
    scores = leaderboard_scores(report, res, result)
    scored = Scored(role, report, res, strat, scores)
    if out_dir is not None:
        out = Path(out_dir)
        scored.paths["json"] = _write_new(out / f"eval_report_{role}.json", report.to_json())
        scored.paths["md"] = _write_new(out / f"eval_report_{role}.md", report.to_markdown() + engine_markdown(report, bt))
    return scored


def scores_json(scores: Mapping[str, Optional[float]]) -> str:
    return json.dumps(dict(scores), indent=2, sort_keys=True)


__all__ = ["ROLES", "Scored", "ScoringError", "engine_markdown", "leaderboard_scores", "score_result",
           "training_facts"]
