"""Compare runs by their evaluation reports (plan C3).

    from neural_trade.experiments.compare import compare_runs
    compare_runs("runs/*", metrics=["h1/direction/auc", "h1/variance/crpss", "backtest/sharpe_net"])

One row per run directory, one column per metric (keys of EvalReport.flat()), plus the run's
tags and seed. A run without ``eval_report_<split>.json`` is refused, not silently skipped:
every number in a comparison must come from a scored run.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd


def _flat(report: dict) -> dict:
    out = {}
    for h, row in report.get("model", {}).get("horizons", {}).items():
        for group in ("direction", "gauss_direction", "delta", "delta_raw", "variance"):
            for k, v in (row.get(group) or {}).items():
                out[f"{h}/{group}/{k}"] = v
    for k, v in report.get("model", {}).get("coherence", {}).items():
        out[f"coherence/{k}"] = v
    for k, v in ((report.get("backtest") or {}).get("summary") or {}).items():
        out[f"backtest/{k}"] = v
    return out


def _dirs(runs: Union[str, Path, Iterable[Union[str, Path]]]) -> List[Path]:
    if isinstance(runs, (str, Path)):
        runs = [runs]
    out = []
    for r in runs:
        matches = glob.glob(str(r))
        out += [Path(m) for m in sorted(matches)] if matches else [Path(r)]
    # A run directory has meta.json or config.yaml; container folders (ablations/, gates/ ...) are not runs.
    return [d for d in out if d.is_dir() and ((d / "meta.json").exists() or (d / "config.yaml").exists())]


def compare_runs(runs, split: str = "test", metrics: Optional[Sequence[str]] = None,
                 skip_unscored: bool = False) -> pd.DataFrame:
    """``runs``: run directories, globs, or a mix. Raises FileNotFoundError for a run without an
    evaluation report unless ``skip_unscored`` (then it is listed in ``df.attrs['unscored']``)."""
    rows, unscored = [], []
    for d in _dirs(runs):
        path = d / f"eval_report_{split}.json"
        if not path.exists():
            if skip_unscored:
                unscored.append(d.name)
                continue
            raise FileNotFoundError(f"{d} has no {path.name}: score it (evaluation.report.evaluate) before comparing")
        report = json.loads(path.read_text(encoding="utf-8"))
        flat = _flat(report)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        row = {"run_id": report.get("run_id") or d.name, "seed": meta.get("seed"),
               "tags": ",".join(meta.get("tags", []))}
        row.update({m: flat.get(m) for m in metrics} if metrics else flat)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run_id") if rows else pd.DataFrame()
    df.attrs["unscored"] = unscored
    return df
