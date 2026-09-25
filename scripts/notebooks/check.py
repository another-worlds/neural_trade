"""Check the saved outputs of the notebooks: errors, stderr, empty figure panels, unexecuted cells.

    python scripts/notebooks/check.py                  # every notebook in notebooks/
    python scripts/notebooks/check.py 01 04            # full name, number prefix, or a path to an .ipynb

Per notebook it prints the file size, then every saved plotly figure (cell, title, trace count, empty
panels, and any 'not logged' / 'no trades' notes the figure writes on itself), every error output,
every stderr stream, and every code cell without an execution count (built but never executed).
Exit 1 if any notebook has an error, a stderr stream, an empty panel or an unexecuted code cell;
exit 0 when all are clean. It only reads the files: nothing is executed.

A clean check does not replace looking at the figures (render.py): it cannot see a panel that is
drawn but wrong.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "notebooks"
if str(REPO / "src") not in sys.path:   # this checkout's neural_trade (a worktree's own), not the installed one
    sys.path.insert(0, str(REPO / "src"))

PLOTLY = "application/vnd.plotly.v1+json"
NOTE_MARKERS = ("not logged", "nothing to", "needs the")


def resolve(names, nb_dir: Path = NB_DIR) -> list[Path]:
    """Notebook files from full names, number prefixes or paths; every notebook in nb_dir when empty."""
    every = sorted(nb_dir.glob("*.ipynb"))
    if not names:
        return every
    out = []
    for arg in names:
        if Path(arg).is_file():
            out.append(Path(arg))
            continue
        stem = Path(arg).stem
        hits = [p for p in every if p.stem == stem] or [p for p in every if p.stem.startswith(stem)]
        if len(hits) != 1:
            raise SystemExit(f"unknown or ambiguous notebook {arg!r}; one of: {', '.join(p.stem for p in every)}")
        out.append(hits[0])
    return out


def _text(value) -> str:
    return value if isinstance(value, str) else "".join(value)


@dataclass
class Figure:
    cell: int
    title: str
    traces: int
    empty: list
    notes: list


@dataclass
class Report:
    path: Path
    size: int
    figures: list = field(default_factory=list)
    errors: list = field(default_factory=list)      # (cell, "Ename: evalue")
    stderr: list = field(default_factory=list)      # (cell, text)
    unexecuted: list = field(default_factory=list)  # cell indices

    @property
    def empty(self) -> list:
        return [f for f in self.figures if f.empty]

    @property
    def ok(self) -> bool:
        return not (self.errors or self.stderr or self.empty or self.unexecuted)

    def lines(self) -> list[str]:
        out = [f"==== {self.path.name}  {self.size / 1e6:.1f} MB"]
        for f in self.figures:
            flag = "  <-- EMPTY PANEL" if f.empty else ""
            notes = f" notes={f.notes}" if f.notes else ""
            out.append(f"  [{f.cell}] fig {f.title[:70]!r} traces={f.traces} empty={f.empty}{notes}{flag}")
        out += [f"  [{c}] ERROR {text[:300]}" for c, text in self.errors]
        out += [f"  [{c}] STDERR {text[:300]!r}" for c, text in self.stderr]
        out += [f"  [{c}] NOT EXECUTED (no execution count)" for c in self.unexecuted]
        out.append(f"  figures={len(self.figures)} errors={len(self.errors)} stderr={len(self.stderr)} "
                   f"empty_panels={len(self.empty)} unexecuted={len(self.unexecuted)}  {'OK' if self.ok else 'FAIL'}")
        return out


def check_notebook(path) -> Report:
    """Read one saved notebook and collect what check.py reports."""
    import plotly.graph_objects as go

    from neural_trade.visualization.theme import empty_panels

    path = Path(path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    rep = Report(path=path, size=path.stat().st_size)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if cell.get("execution_count") is None and _text(cell.get("source", "")).strip():
            rep.unexecuted.append(i)
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "error":
                rep.errors.append((i, f"{out.get('ename')}: {out.get('evalue', '')}"))
            elif kind == "stream" and out.get("name") == "stderr":
                rep.stderr.append((i, _text(out.get("text", ""))))
            spec = out.get("data", {}).get(PLOTLY)
            if spec is None:
                continue
            try:
                fig = go.Figure(spec)
            except Exception as exc:  # a figure plotly cannot load is as broken as an error output
                rep.errors.append((i, f"saved figure does not load: {type(exc).__name__}: {exc}"))
                continue
            title = re.sub(r"<[^>]+>", " ", fig.layout.title.text or "").split("  ")[0].strip()
            notes = [a.text for a in fig.layout.annotations
                     if a.text and (any(m in a.text for m in NOTE_MARKERS) or a.text == "no trades")]
            rep.figures.append(Figure(i, title, len(fig.data), empty_panels(fig), notes))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="notebooks (full name, number prefix or path); default: all")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    reports = [check_notebook(p) for p in resolve(args.names)]
    for rep in reports:
        print("\n".join(rep.lines()))
    bad = [r.path.name for r in reports if not r.ok]
    n_figs = sum(len(r.figures) for r in reports)
    print(f"{len(reports)} notebook(s), {n_figs} figures: " + (f"FAIL in {', '.join(bad)}" if bad else "all clean"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
