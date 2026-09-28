"""Check that every run id cited in the docs, the reports and the saved notebooks links to a tracked run.

    python scripts/check_run_evidence.py                    # exit 0: every cited run is tracked
    python scripts/check_run_evidence.py --list-untracked   # the light files to stage, one path per line
    python scripts/check_run_evidence.py --repo PATH        # another checkout (default: this script's)

A run id is ``YYYYMMDDTHHMMSSZ-<sha7>[-dirty]-<hash8>`` (experiments/run_context.py). A run directory
is a directory under runs/, at any depth, whose name is a run id, alone or followed by ``-<suffix>``
(runs/<id>/, runs/ablations/<name>/runs/<id>-<cell>/, runs/experiments/<name>/runs/<id>-<name>/).
Citations are searched in docs/**/*.md, README.md, runs/**/REPORT.md, runs/**/report.md,
runs/**/summary.md and the saved notebooks (notebooks/*.ipynb: every string in them, so the cell
outputs and the widget state). A citation that carries a suffix (``<id>-all_on__s0__P1``) resolves by
its id.

A cited id passes when it has a run directory, in git or on disk, and in each of its directories git
tracks config.yaml and meta.json and no light file is untracked. Light means not ignored: .gitignore
ignores the heavy artefacts (weights, scalers, npz, parquet, tb/) and the experiment log folders, so
they never count as missing (docs/RUNBOOK.md "Run directories in git"). Only git's view of the files
is used (``git ls-files``), so the check works in a CI checkout that has only the tracked files and in
any worktree.

Exit 0 with one summary line; exit 1 naming each failing run id, one file that cites it and the
missing paths; exit 2 when the path is not in a git work tree. ``--list-untracked`` prints the
untracked light files of the cited runs (repo-relative, forward slashes) and exits 0: stage them by
path (``git add --pathspec-from-file=<the list>``), never with ``git add runs``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

REPO = Path(__file__).resolve().parent.parent
# The sha has 7 digits unless git needs more to be unambiguous (visualization/comparison.py reads 7-40).
RUN_ID = r"\d{8}T\d{6}Z-[0-9a-f]{7,40}(?:-dirty)?-[0-9a-f]{8}"
CITATION = re.compile(rf"(?<![0-9A-Za-z])({RUN_ID})(?![0-9A-Za-z])")
RUN_DIR = re.compile(rf"({RUN_ID})(?:-.*)?")            # fullmatch against a directory name
REPORT_NAMES = frozenset({"REPORT.md", "report.md", "summary.md"})
REQUIRED = ("config.yaml", "meta.json")                # what every cited run must have in git


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


@dataclass
class CitedRun:
    run_id: str
    cited_in: list                                  # repo-relative files that cite it, sorted
    dirs: list = field(default_factory=list)        # its run directories, repo-relative
    missing: list = field(default_factory=list)     # (path, why): why is untracked, ignored or absent

    @property
    def ok(self) -> bool:
        return bool(self.dirs) and not self.missing


@dataclass
class Evidence:
    runs: list                                      # CitedRun per cited id, sorted by id
    untracked: list                                 # the untracked light files of the cited runs, sorted

    @property
    def failing(self) -> list:
        return [r for r in self.runs if not r.ok]

    def summary(self) -> str:
        files = len({f for r in self.runs for f in r.cited_in})
        head = f"check_run_evidence: {_count(len(self.runs), 'cited run id')} in {_count(files, 'file')}"
        bad = self.failing
        if not bad:
            return f"{head}, all tracked"
        no_dir = sum(not r.dirs for r in bad)
        paths = sum(len(r.missing) for r in bad)
        return (f"{head}: {len(bad)} fail ({no_dir} without a run directory, {len(bad) - no_dir} with "
                f"{_count(paths, 'missing path')}); stage a run's light files with --list-untracked")

    def report(self) -> list:
        """The failing runs, each with one citing file and its missing paths, then the summary line."""
        lines = []
        for run in self.failing:
            more = len(run.cited_in) - 1
            lines.append(f"FAIL {run.run_id} (cited in {run.cited_in[0]}"
                         + (f" and {more} more file{'s' if more > 1 else ''}" if more else "") + ")")
            if not run.dirs:
                lines.append("  no run directory under runs/, in git or on disk")
            lines += [f"  {why:9s} {path}" for path, why in run.missing]
        return lines + [self.summary()]


def toplevel(path) -> Optional[Path]:
    """The work tree root that contains ``path``, or None (not a git work tree, or no git)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(path), capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out) if out else None


def _git_paths(top: Path, *args: str) -> set:
    out = subprocess.run(["git", *args, "-z", "--", "runs"], cwd=str(top), capture_output=True, check=True).stdout
    return {p for p in out.decode("utf-8", "replace").split("\0") if p}


def _strings(node) -> Iterator[str]:
    """Every string in a parsed notebook, one at a time (never joined: a join could glue two ids)."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def cited_ids(path: Path) -> set:
    """The run ids a file cites; a notebook is read as JSON, so escapes in its strings do not hide an id."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".ipynb":
        try:
            return {m.group(1) for s in _strings(json.loads(text)) for m in CITATION.finditer(s)}
        except ValueError:
            pass   # not valid JSON: fall back to the raw text
    return {m.group(1) for m in CITATION.finditer(text)}


def _walk_runs(top: Path) -> tuple:
    """(report files, run directories) under runs/ on disk, repo-relative."""
    reports, dirs = [], set()
    for dirpath, dirnames, filenames in os.walk(top / "runs"):
        base = Path(dirpath).relative_to(top).as_posix()
        dirs.update(f"{base}/{d}" for d in dirnames if RUN_DIR.fullmatch(d))
        reports += [f"{base}/{n}" for n in filenames if n in REPORT_NAMES]
    return reports, dirs


def citing_files(top: Path, reports: Iterable[str] = ()) -> list:
    """The files whose citations must resolve, repo-relative and sorted (``reports``: runs/**/*.md hits)."""
    files = list(reports)
    for dirpath, _, filenames in os.walk(top / "docs"):
        base = Path(dirpath).relative_to(top).as_posix()
        files += [f"{base}/{n}" for n in filenames if n.endswith(".md")]
    if (top / "README.md").is_file():
        files.append("README.md")
    if (top / "notebooks").is_dir():
        files += [f"notebooks/{p.name}" for p in (top / "notebooks").iterdir()
                  if p.suffix == ".ipynb" and p.is_file()]
    return sorted(set(files))


def _run_dirs_in(paths: Iterable[str]) -> set:
    """The run directories that contain any of the repo-relative ``paths``."""
    out = set()
    for path in paths:
        parts = path.split("/")
        for i in range(1, len(parts) - 1):
            if RUN_DIR.fullmatch(parts[i]):
                out.add("/".join(parts[: i + 1]))
    return out


def collect(top: Path) -> Evidence:
    """Resolve every cited run id of the work tree ``top`` and check what git tracks of it."""
    top = Path(top)
    reports, disk_dirs = _walk_runs(top)
    cited: dict = {}
    for rel in citing_files(top, reports):
        for run_id in cited_ids(top / rel):
            cited.setdefault(run_id, []).append(rel)
    tracked = _git_paths(top, "ls-files")
    untracked = _git_paths(top, "ls-files", "--others", "--exclude-standard")   # not ignored, not tracked
    dirs_by_id: dict = {}
    for d in disk_dirs | _run_dirs_in(tracked | untracked):
        dirs_by_id.setdefault(RUN_DIR.fullmatch(d.rsplit("/", 1)[-1]).group(1), set()).add(d)
    runs, to_stage = [], set()
    for run_id in sorted(cited):
        run = CitedRun(run_id, cited[run_id], sorted(dirs_by_id.get(run_id, ())))
        for d in run.dirs:
            missing = {p: "untracked" for p in untracked if p.startswith(d + "/")}
            to_stage.update(missing)
            for name in REQUIRED:
                p = f"{d}/{name}"
                if p not in tracked and p not in missing:
                    missing[p] = "ignored" if (top / p).exists() else "absent"
            run.missing += sorted(missing.items())
        runs.append(run)
    return Evidence(runs, sorted(to_stage))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=REPO,
                    help="a directory of the checkout to check (default: this script's)")
    ap.add_argument("--list-untracked", action="store_true",
                    help="print the untracked light files of the cited runs, one path per line, and exit 0")
    args = ap.parse_args(argv)
    top = toplevel(args.repo)
    if top is None:
        print(f"check_run_evidence: {args.repo} is not in a git work tree (or git is not installed)", file=sys.stderr)
        return 2
    evidence = collect(top)
    if args.list_untracked:
        for path in evidence.untracked:
            print(path)
        return 0
    print("\n".join(evidence.report()))
    return 1 if evidence.failing else 0


if __name__ == "__main__":
    sys.exit(main())
