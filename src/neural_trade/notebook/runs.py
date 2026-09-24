"""Finding a trained run to load in the notebooks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union


def servable_runs(runs_dir: Union[str, Path] = "runs") -> List[Path]:
    """Run directories under ``runs_dir`` (searched recursively) that hold a serving bundle
    (``artifacts/weights.h5``), newest first."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    found = {p.parent.parent for p in root.rglob("artifacts/weights.h5")}
    return sorted(found, key=os.path.getmtime, reverse=True)


def pick_run(run_dir: Optional[Union[str, Path]] = None, runs_dir: Union[str, Path] = "runs") -> Path:
    """``run_dir`` if given, else the newest run under ``runs_dir`` that can be loaded; a clear error if none."""
    if run_dir:
        path = Path(run_dir)
        if not (path / "artifacts" / "weights.h5").exists():
            raise FileNotFoundError(f"{path} has no serving bundle (artifacts/weights.h5). Runs that do: "
                                    f"{[str(p) for p in servable_runs(runs_dir)[:5]] or 'none'}")
        return path
    runs = servable_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(
            f"No trained run with a serving bundle under {Path(runs_dir).resolve()}. Train one first: run "
            "notebook 01_train_and_monitor (or `neural-trade train`), or set RUN_DIR to a run directory that "
            "contains artifacts/.")
    return runs[0]
