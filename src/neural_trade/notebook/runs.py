"""Finding a trained run to load in the notebooks.

Notebooks 02-05 load the newest notebook or CLI run. Experiment-engine runs (the cells of a
scenario or sweep, under the run store's ``scenarios/`` subtree, their meta.json carrying an
``engine`` section) are left out of that default, so a quick sweep never replaces the run the
notebooks show; pass ``include_engine_runs=True``, or the run directory itself, to load one.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union


def _is_engine_run(run_dir: Path, root: Path) -> bool:
    from neural_trade.experiments.store import ENGINE_SUBTREE, is_engine_run_dir

    try:
        parts = run_dir.relative_to(root).parts
    except ValueError:
        parts = ()
    return (bool(parts) and parts[0] == ENGINE_SUBTREE) or is_engine_run_dir(run_dir)


def servable_runs(runs_dir: Union[str, Path] = "runs", *, include_engine_runs: bool = False) -> List[Path]:
    """Run directories under ``runs_dir`` (searched recursively) that hold a serving bundle
    (``artifacts/weights.h5``), newest first; engine runs only with ``include_engine_runs``."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    found = {p.parent.parent for p in root.rglob("artifacts/weights.h5")}
    if not include_engine_runs:
        found = {d for d in found if not _is_engine_run(d, root)}
    return sorted(found, key=os.path.getmtime, reverse=True)


def pick_run(run_dir: Optional[Union[str, Path]] = None, runs_dir: Union[str, Path] = "runs", *,
             include_engine_runs: bool = False) -> Path:
    """``run_dir`` if given, else the newest run under ``runs_dir`` that can be loaded (not an engine
    run unless ``include_engine_runs``); a clear error if none."""
    if run_dir:
        path = Path(run_dir)
        if not (path / "artifacts" / "weights.h5").exists():
            raise FileNotFoundError(f"{path} has no serving bundle (artifacts/weights.h5). Runs that do: "
                                    f"{[str(p) for p in servable_runs(runs_dir)[:5]] or 'none'}")
        return path
    runs = servable_runs(runs_dir, include_engine_runs=include_engine_runs)
    if not runs:
        raise FileNotFoundError(
            f"No trained run with a serving bundle under {Path(runs_dir).resolve()}. Train one first: run "
            "notebook 01_train_and_monitor (or `neural-trade train`), or set RUN_DIR to a run directory that "
            "contains artifacts/.")
    return runs[0]
