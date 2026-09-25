# Notebook tooling

The six notebooks in `notebooks/` are **generated** by `build.py`, **executed in place** on their real
defaults, and **committed with their outputs** (the outputs of the last real run), so they can be read
without a kernel. The scripts here are the only way to change them.

| Script | What it does |
| --- | --- |
| `build.py` | Generates the notebooks from the cell lists in it. **Edit notebooks here, never by hand.** `--out DIR` writes elsewhere; `--check` exits 1 when `notebooks/` differs from it. |
| `execute.py` | Executes notebooks in place: kernel `python3`, cwd `notebooks/`, this checkout's `src/` first on `PYTHONPATH`, 7200 s per cell. Prints `[nb] name: OK/ERROR (secs)` and exits 1 on an error. `--no-store-widget-state` leaves the widget state out (the default stores it). |
| `check.py` | Reads the saved notebooks and lists, per notebook: size, figures, errors, stderr, empty panels and unexecuted cells. It exits 1 if any of these is found. |
| `render.py` | Renders every saved figure, plus the training-health HTML, to PNG with headless Edge, and prints a manifest. **Windows + Microsoft Edge only.** |

Run everything from the repository root with the `nt` environment's Python:

```bash
PY=C:/Users/Step/miniforge3/envs/nt/python      # or: conda activate nt, then python
export PYTHONIOENCODING=utf-8                    # the console code page is cp1251
```

Notebook names are full names (`04_diagnostics`) or number prefixes (`04`). With no names, a script
works on all six.

## The workflow

1. **Edit `scripts/notebooks/build.py`**, in the cell lists (`data`, `train`, `backtest`, `signals_nb`,
   `diag`, `compare`). Keep the cells thin: put logic in `neural_trade` (tested there) and call it from
   a cell. No `def`, `class` or `lambda` in a cell (`tests/test_notebooks_thin.py`). The first code cell
   of each notebook has the tag `parameters`.
2. **Build** only the notebooks you changed. A build writes the notebook without outputs.
   ```bash
   $PY scripts/notebooks/build.py 04
   ```
3. **Execute on the real defaults**, not on small overrides:
   ```bash
   $PY scripts/notebooks/execute.py 04        # one notebook
   $PY scripts/notebooks/execute.py           # all six, 00 -> 05
   ```
   `01_train_and_monitor` trains a **new run** on the GPU (about 5 minutes). 02-05 read the newest
   run under `runs/`, so execute 01 first when the model, the training or the evaluation changed.
   Rules for 01:
   - Do not set `CUDA_VISIBLE_DEVICES=-1`, or it trains on the CPU and takes much longer.
   - Run one training at a time, and not while another training uses the GPU: the GPU is launch-bound,
     so parallel runs are slower.

   A failed notebook is saved with the traceback in the failing cell, and the script stops.
4. **Check**. It must print `all clean` and exit 0:
   ```bash
   $PY scripts/notebooks/check.py
   ```
5. **Render, then LOOK at every figure**:
   ```bash
   $PY scripts/notebooks/render.py            # or: render.py 04 --out D:/nb_png
   ```
   Open every PNG in the printed manifest. An agent reads each path with its image-reading tool. Compare
   each figure with the markdown above its cell. Look for:
   - empty, clipped or overlapping panels, and legends that cover data;
   - wrong units or scales;
   - `n/a` where a number belongs;
   - numbers that disagree with the tables;
   - a verdict the figure does not support.

   `check.py` finds only broken figures. A figure that is drawn but wrong needs someone to look at it.
6. **Test** (fast), then the full suite:
   ```bash
   $PY -m pytest tests/test_notebook_tooling.py tests/test_notebooks_thin.py -q
   ```
7. **Commit `build.py` together with the executed notebooks (with outputs).** In the message, say which
   run the outputs come from (run id, served epoch) and what the check found (figures, errors, stderr,
   empty panels).

## Rules

- **Never commit a hand-edited notebook.** `tests/test_notebook_tooling.py` compares each committed
  notebook's cells (type, source, tags) with what `build.py` generates. Outputs are not compared. A
  change made in Jupyter fails that test and is lost at the next build. Move the change into `build.py`.
  To check for drift without the tests, run `build.py --check`.
- **Keep the outputs.** Notebooks are committed with their outputs (D-013). There is no nbstripout filter
  or hook in this repo, and none may be added: `nbstripout --install` would strip the outputs at `git add`.
  To confirm, `git config --get filter.nbstripout.clean` must print nothing.
- **Interactive controls work only with a running kernel.** In the saved file, each widget cell is
  followed by a static copy of its first result, and a reader without a kernel sees that copy.
- **Disk.** C: is nearly full. When C: has less than 5 GB free, `render.py` writes to
  `D:/neural_trade_renders`; otherwise it writes to `<system temp>/neural_trade_renders`. Delete that
  folder when you are done. Each new render replaces the PNGs of the previous one.
- **Worktrees.** `execute.py` and `check.py` use the checkout's own `src/`, not the editable install's.
  The kernel starts in the checkout's `notebooks/`, so `../runs` is that checkout's runs folder, and it
  is empty in a fresh worktree until 01 trains.
- **Never execute in tests.** The fast tests only read files. `test_all_notebooks_execute` in
  `tests/test_notebooks_thin.py` executes the notebooks on synthetic bars into a temp folder (marked
  `slow`). It never writes to `notebooks/`.
