"""Finding a trained run to load in the notebooks.

Notebooks 02-07 load the newest notebook or CLI run: a run directory under the runs root that is
not part of a study. Left out of that default, so a sweep or a pre-registered study never replaces
the run the notebooks show (D-013: notebooks show the shipped defaults):

- experiment-engine runs (the cells of a scenario or sweep, under the run store's ``scenarios/``
  subtree, their meta.json carrying an ``engine`` section);
- study runs made with ``neural-trade train`` under ``experiments/<study>/runs/`` and the
  ablation cells under ``ablations/``.

Pass ``include_study_runs=True`` (``include_engine_runs=True`` includes both kinds), or the run
directory itself (``RUN_DIR``), to load one of them.
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


STUDY_SUBTREES = ("experiments", "ablations")


def _is_study_run(run_dir: Path, root: Path) -> bool:
    try:
        parts = run_dir.relative_to(root).parts
    except ValueError:
        return False
    return len(parts) > 1 and parts[0] in STUDY_SUBTREES


def servable_runs(runs_dir: Union[str, Path] = "runs", *, include_engine_runs: bool = False,
                  include_study_runs: bool = False) -> List[Path]:
    """Run directories under ``runs_dir`` (searched recursively) that hold a serving bundle
    (``artifacts/weights.h5``), newest first; engine runs only with ``include_engine_runs``, study
    runs (``experiments/``, ``ablations/``) only with ``include_study_runs`` or ``include_engine_runs``."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    found = {p.parent.parent for p in root.rglob("artifacts/weights.h5")}
    if not include_engine_runs:
        found = {d for d in found if not _is_engine_run(d, root)}
    if not (include_engine_runs or include_study_runs):
        found = {d for d in found if not _is_study_run(d, root)}
    return sorted(found, key=os.path.getmtime, reverse=True)


def pick_run(run_dir: Optional[Union[str, Path]] = None, runs_dir: Union[str, Path] = "runs", *,
             include_engine_runs: bool = False, include_study_runs: bool = False) -> Path:
    """``run_dir`` if given, else the newest run under ``runs_dir`` that can be loaded (not an engine
    or study run unless the flags say so); a clear error if none."""
    if run_dir:
        path = Path(run_dir)
        if not (path / "artifacts" / "weights.h5").exists():
            raise FileNotFoundError(f"{path} has no serving bundle (artifacts/weights.h5). Runs that do: "
                                    f"{[str(p) for p in servable_runs(runs_dir)[:5]] or 'none'}")
        return path
    runs = servable_runs(runs_dir, include_engine_runs=include_engine_runs,
                         include_study_runs=include_study_runs)
    if not runs:
        excluded = servable_runs(runs_dir, include_engine_runs=True)
        if excluded:
            raise FileNotFoundError(
                f"No notebook or CLI run with a serving bundle under {Path(runs_dir).resolve()}; only "
                f"engine or study runs exist ({[str(p) for p in excluded[:5]]}). Train one first (notebook "
                "01_train_and_monitor or `neural-trade train`), or set RUN_DIR to one of those run "
                "directories to load it.")
        raise FileNotFoundError(
            f"No trained run with a serving bundle under {Path(runs_dir).resolve()}. Train one first: run "
            "notebook 01_train_and_monitor (or `neural-trade train`), or set RUN_DIR to a run directory that "
            "contains artifacts/.")
    return runs[0]
