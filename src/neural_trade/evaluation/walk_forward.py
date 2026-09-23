"""Evaluate a finished training run, and walk-forward (several folds x seeds) evaluation.

    report = evaluate_result(result, run_id=ctx.run_id)          # baselines fit on train
    reports = walk_forward(Config(EPOCHS=20), folds=(-2, -1), seeds=(0, 1))

Folds double as the ablation harness's "periods" (fold -2 = P1, fold -1 = P2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

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


def walk_forward(config, folds: Iterable[int] = (-3, -2, -1), seeds: Iterable[int] = (0,), *, root="runs",
                 epochs: Optional[int] = None, calibrate: bool = True, tags=()) -> List[EvalReport]:
    """Train and evaluate once per (fold, seed); each run gets its own RunContext directory."""
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.training.trainer import train_and_evaluate

    reports = []
    for fold in folds:
        for seed in seeds:
            ctx = RunContext.create(config.copy(FOLD_INDEX=int(fold)), root=root, seed=int(seed),
                                    tags=["walk_forward", f"fold{fold}", f"seed{seed}", *tags],
                                    name=f"wf-f{fold}-s{seed}")
            result = train_and_evaluate(config=ctx.config, run_context=ctx, force=True, epochs=epochs,
                                        calibrate=calibrate)
            reports.append(evaluate_result(result, run_id=ctx.run_id, out_dir=ctx.run_dir))
    return reports
