"""Evaluate a finished training run: baselines fit on train, an EvalReport on test (or cal).

    report = evaluate_result(result, run_id=ctx.run_id)          # baselines fit on train

Training several (fold, seed) runs and evaluating each is now the experiment engine's job
(NT-026, ``neural_trade.experiments``); this module no longer launches training itself (NT-027
removed the dead ``walk_forward`` helper that used to do that - see docs/DECISIONS.md D-023).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from neural_trade.evaluation.baselines import BaselineSet
from neural_trade.evaluation.frame import PredictionFrame
from neural_trade.evaluation.report import EvalReport, evaluate


def evaluate_result(result, *, run_id: Optional[str] = None, with_baselines: bool = True,
                    backtest: Optional[dict] = None, out_dir=None, figure: bool = False) -> EvalReport:
    """EvalReport for a TrainResult's test split (confidence threshold from its cal split)."""
    from neural_trade.data.processor import split_arrays

    cfg = result.config
    arrays = split_arrays(cfg) if with_baselines else None
    frame = PredictionFrame.from_result(result, "test", X_raw=arrays["test"]["X"] if arrays else None)
    cal = PredictionFrame.from_result(result, "cal") if result.predictions_cal is not None else None
    baselines = None
    if arrays is not None:
        tr = arrays["train"]
        baselines = BaselineSet.fit(tr["X"], tr["y"], tr["last_close"], cfg.DIR_DEADBAND_BPS)
    report = evaluate(frame, cfg, baselines=baselines, cal_frame=cal, run_id=run_id, backtest=backtest)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.to_json(out / f"eval_report_{frame.split}.json")
        report.to_markdown(out / f"eval_report_{frame.split}.md")
        if figure:
            from neural_trade.registries.visualizations import Visualizations

            Visualizations.build("eval_report", frame, cfg).write_html(str(out / f"eval_report_{frame.split}.html"))
    return report
