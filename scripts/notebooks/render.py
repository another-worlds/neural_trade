"""Render every figure saved in the notebooks to a PNG, so each one can be LOOKED at without Jupyter.

    python scripts/notebooks/render.py                      # every notebook in notebooks/
    python scripts/notebooks/render.py 01 04 --out D:/nb_png

WINDOWS + MICROSOFT EDGE ONLY. Each figure is written to a standalone HTML file, and headless Edge
(msedge.exe --screenshot) takes the screenshot. Covered: every plotly figure in a cell output, plus the
training-health tiles (an HTML output) of 01 and 04. An HTML output has no known height, so it is
screenshotted in a 4000 px window and cropped to its content (Pillow). PNGs are named <nb>_c<cell>_<k>.png
(k counts the figures in the notebook) and <nb>_c<cell>_health.png. The script
prints a manifest (PNG, notebook, cell, title), and manifest.json next to the PNGs adds each cell's source.

Default output folder: <drive D:>/neural_trade_renders when C: has less than 5 GB free, else
<system temp>/neural_trade_renders. PNGs from an earlier render in that folder are deleted first, so it
holds only the current render.

Headless Edge needs the following, and this script does all of it. Each call gets a fresh --user-data-dir,
otherwise a running Edge takes the call over and writes no screenshot. It uses --headless=new. The PNG is
checked after each call, and the call is retried (3 tries). The HTML (about 4 MB, plotly.js inlined)
and the profile folder are deleted after the call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "notebooks"
PLOTLY = "application/vnd.plotly.v1+json"
EDGE_CANDIDATES = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
PAPER = "#121211"   # neural_trade.visualization.theme.PAPER: the page behind an HTML output
TAG = re.compile(r"^\d\d_c\d+_(\d+|health)\.png$")


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


def default_out_dir() -> Path:
    """D:/neural_trade_renders when C: is short of space (< 5 GB free) and D: exists; else the system temp dir."""
    try:
        c_free = shutil.disk_usage("C:\\").free
    except OSError:
        c_free = None
    if c_free is not None and c_free < 5e9 and Path("D:\\").exists():
        return Path("D:\\") / "neural_trade_renders"
    return Path(tempfile.gettempdir()) / "neural_trade_renders"


def find_edge() -> str | None:
    found = shutil.which("msedge") or shutil.which("msedge.exe")
    return found or next((p for p in EDGE_CANDIDATES if Path(p).is_file()), None)


def crop_background(png: Path, margin: int = 16) -> None:
    """Cut the empty page below the content: an HTML output has no known height, so it is screenshotted in a
    tall window. Needs Pillow (installed with matplotlib); without it the PNG keeps the empty part."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    with Image.open(png) as img:
        rgb = img.convert("RGB")
    box = ImageChops.difference(rgb, Image.new("RGB", rgb.size, PAPER)).getbbox()
    if box and box[3] + margin < rgb.height:
        rgb.crop((0, 0, rgb.width, box[3] + margin)).save(png)


def shot(edge: str, content, png: Path, *, width: int = 1500, height: int | None = None, tries: int = 3) -> bool:
    """Screenshot a plotly figure (or an HTML string) to ``png`` with headless Edge. True when the PNG exists."""
    html_path = png.with_suffix(".html")
    is_html = isinstance(content, str)
    if is_html:
        html_path.write_text(f"<html><body style='margin:0;background:{PAPER}'>{content}</body></html>",
                             encoding="utf-8")
        height = height or 4000   # cropped to the content afterwards
    else:
        content.write_html(str(html_path), include_plotlyjs=True, full_html=True,
                           default_width=f"{width}px", config={"displayModeBar": False})
        height = height or int(content.layout.height or 600) + 20
    try:
        for _ in range(tries):
            png.unlink(missing_ok=True)
            profile = Path(tempfile.mkdtemp(prefix="edge_profile_", dir=png.parent))
            try:
                subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={profile}",
                                f"--window-size={width + 20},{height}", "--virtual-time-budget=8000",
                                f"--screenshot={png}", html_path.as_uri()],
                               capture_output=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
            finally:
                shutil.rmtree(profile, ignore_errors=True)
            if png.is_file() and png.stat().st_size > 0:
                if is_html:
                    crop_background(png)
                return True
            time.sleep(1)
        return False
    finally:
        html_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="notebooks (full name, number prefix or path); default: all")
    ap.add_argument("--out", default=None, help="output folder (default: see above)")
    ap.add_argument("--width", type=int, default=1500, help="page width in px (default 1500)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    edge = find_edge() if os.name == "nt" else None
    if edge is None:
        print("render.py needs Windows and Microsoft Edge (msedge.exe); none found. "
              "Look at the figures in Jupyter instead.")
        return 2
    import plotly.graph_objects as go

    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():   # only this script's own files
        if TAG.match(old.name) or old.name == "manifest.json":
            old.unlink()

    manifest, failed = [], []
    for path in resolve(args.names):
        nb = json.loads(path.read_text(encoding="utf-8"))
        prefix, k = path.stem[:2], 0
        for i, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", ""))[:400]
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                html = "".join(data.get("text/html", ""))
                if PLOTLY in data:
                    fig = go.Figure(data[PLOTLY])
                    title = re.sub(r"<[^>]+>", " ", fig.layout.title.text or "").split("  ")[0].strip()
                    png = out_dir / f"{prefix}_c{i}_{k}.png"
                    k += 1
                    ok = shot(edge, fig, png, width=int(fig.layout.width or args.width))
                elif "Training health" in html:
                    title = "Training health tiles and the served epoch's numbers (HTML)"
                    png = out_dir / f"{prefix}_c{i}_health.png"
                    ok = shot(edge, html, png, width=args.width)
                else:
                    continue
                (manifest if ok else failed).append({"png": str(png), "notebook": path.stem, "cell": i,
                                                     "title": title[:200], "source": source})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for m in manifest:
        print(f"{m['png']}  | {m['notebook']} cell {m['cell']} | {m['title'][:90]}")
    for m in failed:
        print(f"FAILED {m['png']}  | {m['notebook']} cell {m['cell']} | {m['title'][:90]}")
    print(f"{len(manifest)} rendered to {out_dir}" + (f", {len(failed)} FAILED" if failed else "")
          + ". Look at every one (open the PNG) before committing.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
