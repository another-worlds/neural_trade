"""Select (and optionally run) the test files that touch the code changed since a base ref (D-060).

    python scripts/test_changed.py                 # list the tests for changes against origin/remediation/plan
    python scripts/test_changed.py --run           # ... and run them (CPU, -n 4, -m "not slow")
    python scripts/test_changed.py --base HEAD~1   # another base

A test file is selected when it is itself changed, is named ``test_<module>.py`` for a changed module, or
mentions the module's dotted path or imports its name from its package. Changes to files that nearly
everything uses (core/config.py, conftest.py, pyproject.toml, requirements) select the whole fast suite
instead: the script says so and exits 0 without running anything unless --run is given.
This is a development-loop aid. It does not replace the fast suite before a push or CI.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "neural_trade"
WIDE = {"src/neural_trade/core/config.py", "tests/conftest.py", "pyproject.toml", "requirements-ci.txt",
        "src/neural_trade/__init__.py"}


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def changed_files(base: str) -> list[str]:
    files = set(_git("diff", "--name-only", base)) | set(_git("ls-files", "-o", "--exclude-standard"))
    return sorted(f for f in files if (ROOT / f).exists())


def tests_for(module_path: str, test_texts: dict[str, str]) -> set[str]:
    rel = Path(module_path).relative_to("src/neural_trade").with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return set()
    dotted = ".".join(["neural_trade", *parts])
    name = parts[-1]
    pkg = ".".join(["neural_trade", *parts[:-1]])
    imp = re.compile(rf"from\s+{re.escape(pkg)}\s+import\s+[^\n]*\b{re.escape(name)}\b")
    keyed = re.compile(rf"[\"']{re.escape(name)}[\"']")  # a registry key such as Models.build("gru_small")
    hits = set()
    for tf, text in test_texts.items():
        if Path(tf).name == f"test_{name}.py" or dotted in text or imp.search(text) or keyed.search(text):
            hits.add(tf)
    if not hits and pkg != "neural_trade":  # nothing names the module: fall back to its package
        hits = {tf for tf, text in test_texts.items() if re.search(rf"{re.escape(pkg)}", text)}
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/remediation/plan")
    ap.add_argument("--run", action="store_true", help="run the selected tests (CPU, -n 4, not slow)")
    ap.add_argument("-n", default="4", help="xdist workers for --run")
    args = ap.parse_args()

    changed = changed_files(args.base)
    if not changed:
        print("nothing changed against", args.base)
        return 0
    wide = [f for f in changed if f in WIDE]
    test_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").rglob("test_*.py"))
    texts = {tf: (ROOT / tf).read_text(encoding="utf-8", errors="replace") for tf in test_files}

    selected: set[str] = {f for f in changed if f in texts}
    for f in changed:
        if f.startswith("src/neural_trade/") and f.endswith(".py"):
            selected |= tests_for(f, texts)
    other = [f for f in changed if not f.startswith(("src/", "tests/")) and not f.endswith((".md", ".json", ".npz"))
             and not f.startswith(("runs/", "docs/", "notebooks/", "presentations/"))]

    print(f"changed files: {len(changed)} (against {args.base})")
    if wide:
        print("WIDE change (" + ", ".join(wide) + "): run the whole fast suite: pytest -m 'not slow' -n 8")
        return 0 if not args.run else _run([], args.n)
    if other:
        print("note: changes outside src/tests that may matter (scripts, configs): " + ", ".join(other[:6]))
    for tf in sorted(selected):
        print(" ", tf)
    print(f"{len(selected)} test file(s) selected")
    if args.run and selected:
        return _run(sorted(selected), args.n)
    return 0


def _run(files: list[str], n: str) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow", "-n", n, "-x",
           "--durations=8", *files]
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
