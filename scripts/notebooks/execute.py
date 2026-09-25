"""Execute notebooks IN PLACE with their shipped parameters; the outputs are saved into the files.

    python scripts/notebooks/execute.py                  # all, in run order 00 -> 05
    python scripts/notebooks/execute.py 02 04            # some: full name or number prefix (run in file order)
    python scripts/notebooks/execute.py --no-store-widget-state 02

Prints one line per notebook, "[nb] <name>: OK (<secs>s)" or "[nb] <name>: ERROR (<secs>s) <last lines>",
and exits 1 if any notebook failed. A failed notebook is still saved, so the traceback shows in the cell
that raised. The script then stops, because later notebooks read what earlier ones write.
--keep-going runs the rest anyway.

The kernel is "python3" from the environment that runs this script (conda env nt). It starts in
notebooks/, so the notebooks' relative defaults (../configs, ../runs, the bundled CSV) resolve inside
this checkout. PYTHONPATH puts this checkout's src/ first, so in a git worktree the notebooks run the
worktree's code, not the editable install's.

Order matters. 01 trains a NEW run on the real defaults (about 5 minutes on the GPU, much longer on
the CPU). 02-05 read the newest run under ../runs. Run one training at a time: the GPU is launch-bound,
so two runs at once are slower than one after the other.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "notebooks"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def resolve(names, nb_dir: Path = NB_DIR) -> list[Path]:
    """Notebook files, in file (= run) order, from full names, number prefixes or paths; all when empty."""
    every = sorted(nb_dir.glob("*.ipynb"))
    if not names:
        return every
    picked = set()
    for arg in names:
        if Path(arg).is_file():
            picked.add(Path(arg).resolve())
            continue
        stem = Path(arg).stem
        hits = [p for p in every if p.stem == stem] or [p for p in every if p.stem.startswith(stem)]
        if len(hits) != 1:
            raise SystemExit(f"unknown or ambiguous notebook {arg!r}; one of: {', '.join(p.stem for p in every)}")
        picked.add(hits[0].resolve())
    return sorted(picked, key=lambda p: p.name)


def _summary(exc: BaseException) -> str:
    lines = [ln for ln in ANSI.sub("", str(exc)).splitlines() if ln.strip()]
    return f"{type(exc).__name__}: " + " | ".join(lines[-2:])[:500]


def execute(path: Path, *, timeout: int = 7200, store_widget_state: bool = True, kernel: str = "python3") -> str:
    """Run every cell of ``path`` in a fresh kernel started in notebooks/ and save the outputs into the file.
    Returns "OK" or an error summary."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(path), as_version=4)
    status = "OK"
    try:
        NotebookClient(nb, timeout=timeout, kernel_name=kernel, allow_errors=False,
                       store_widget_state=store_widget_state,
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
    except Exception as exc:  # CellExecutionError, CellTimeoutError, DeadKernelError ...
        status = "ERROR " + _summary(exc)
    if not store_widget_state:
        nb.metadata.pop("widgets", None)   # no stale state from an earlier execution
    with open(path, "w", encoding="utf-8", newline="\n") as fh:   # LF, like the repository
        nbformat.write(nb, fh)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="notebooks (full name, number prefix or path); default: all")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per cell (default 7200)")
    ap.add_argument("--store-widget-state", action=argparse.BooleanOptionalAction, default=True,
                    help="save the ipywidgets state into the notebook so saved widgets display without a kernel "
                         "(default: store)")
    ap.add_argument("--keep-going", action="store_true", help="execute the remaining notebooks after a failure")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if sys.platform == "win32":   # zmq needs a selector loop; the default Proactor loop warns (Jupyter does the same)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    paths = resolve(args.names)
    src = str(REPO / "src")
    os.environ["PYTHONPATH"] = os.pathsep.join([src] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                                                        if p and p != src])
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        print("[nb] note: CUDA_VISIBLE_DEVICES=-1 hides the GPU; a notebook that trains will train on the CPU",
              flush=True)
    print(f"[nb] kernel python3 from {sys.executable}; cwd {NB_DIR}; code from {src}", flush=True)

    failed = []
    for i, path in enumerate(paths):
        t0 = time.time()
        status = execute(path, timeout=args.timeout, store_widget_state=args.store_widget_state)
        ok = status == "OK"
        print(f"[nb] {path.stem}: {'OK' if ok else 'ERROR'} ({time.time() - t0:.0f}s)"
              + ("" if ok else " " + status[len("ERROR "):]), flush=True)
        if not ok:
            failed.append(path.stem)
            if not args.keep_going and i + 1 < len(paths):
                print("[nb] stopped; not executed: " + ", ".join(p.stem for p in paths[i + 1:])
                      + " (--keep-going runs them anyway)", flush=True)
                break
    if failed:
        print(f"[nb] {len(failed)} failed: {', '.join(failed)}. The traceback is saved in the failing cell. "
              "Fix the code (or build.py), rebuild, execute again", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
