"""Run evidence (NT-010): .gitignore keeps a run's light files and ignores its heavy ones, and
scripts/check_run_evidence.py fails while a cited run is not tracked. The tests build small git
repositories in tmp_path; the last one checks this repository, so CI enforces the policy."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN = "20250101T000000Z-abc1234-dirty-0badc0de"      # a notebook / CLI run: runs/<id>/
CELL = "20250101T010203Z-abc1234-1234abcd"           # an ablation cell, cited with its suffix
EXP = "20250101T020304Z-def5678-deadbeef"            # an experiment run, cited in a notebook's widget state
GONE = "20250101T030405Z-abc1234-feedf00d"           # cited, but no directory anywhere
CELL_DIR = f"runs/ablations/abl/runs/{CELL}-all_on__s0__P1"
EXP_DIR = f"runs/experiments/exp/runs/{EXP}-bce_f-2"

LIGHT = ("config.yaml", "meta.json", "status.json", "env.json", "metrics.jsonl", "eval_report_test.json",
         "eval_report_test.md", "training_log.csv", "indicator_params_history.csv", "period_init.json",
         "artifacts/meta.json", "artifacts/config.yaml", "artifacts/calibration/online_calibrator.json",
         "artifacts/calibration/pipeline_meta.json", "artifacts/calibration/temperature_params.json",
         "trades.csv", "notes.txt")                  # the last two: any other file of a run is light too
HEAVY = ("weights.h5", "artifacts/weights.h5", "scaler.joblib", "artifacts/calibration/conformal_h0.joblib",
         "model.pkl", "predictions_test.npz", "frame.parquet", "tb/events.out.tfevents.1.host",
         "tb/train/scalars.csv")


def _load():
    spec = importlib.util.spec_from_file_location("check_run_evidence", REPO / "scripts" / "check_run_evidence.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cre():
    return _load()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def _write(root: Path, rel: str, text: str = "x\n") -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel


@pytest.fixture
def git_env(tmp_path, monkeypatch):
    """Git isolated from this machine's config (no global or system file, no excludes file), and never
    searching above tmp_path for a repository."""
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    for key, value in {"GIT_CONFIG_GLOBAL": str(empty), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_COUNT": "1",
                       "GIT_CONFIG_KEY_0": "core.excludesFile", "GIT_CONFIG_VALUE_0": str(empty),
                       "GIT_CEILING_DIRECTORIES": str(tmp_path), "GIT_AUTHOR_NAME": "test",
                       "GIT_AUTHOR_EMAIL": "test@example.com", "GIT_COMMITTER_NAME": "test",
                       "GIT_COMMITTER_EMAIL": "test@example.com"}.items():
        monkeypatch.setenv(key, value)
    return tmp_path


@pytest.fixture
def repo(git_env):
    """A git repository holding only this repository's .gitignore, committed."""
    root = git_env / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    shutil.copyfile(REPO / ".gitignore", root / ".gitignore")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "gitignore")
    return root


def _run(root: Path, run_dir: str, light=("config.yaml", "meta.json", "status.json", "artifacts/meta.json")) -> list:
    """A run directory with the given light files and every heavy kind; returns the light paths."""
    for name in HEAVY:
        _write(root, f"{run_dir}/{name}")
    return [_write(root, f"{run_dir}/{name}") for name in light]


# --------------------------------------------------------------------------- .gitignore


def test_gitignore_leaves_exactly_the_light_files_of_run_directories_untracked(repo):
    """Each heavy kind is ignored in a notebook run, a nested ablation cell and an experiment run; so are
    the ablation's cells/ and logs/ and the experiment's logs/; every other file shows as untracked."""
    light, ignored = [], []
    for run_dir in (f"runs/{RUN}", CELL_DIR, EXP_DIR):
        light += [_write(repo, f"{run_dir}/{name}") for name in LIGHT]
        ignored += [_write(repo, f"{run_dir}/{name}") for name in HEAVY]
    light += [_write(repo, rel) for rel in ("runs/ablations/abl/report.md", "runs/ablations/abl/results.csv",
                                            "runs/ablations/abl/summary.csv", "runs/experiments/exp/REPORT.md",
                                            "runs/experiments/exp/bce_f-2/result.json")]
    ignored += [_write(repo, rel) for rel in ("runs/ablations/abl/cells/all_on__s0__P1.json",
                                              "runs/ablations/abl/logs/all_on__s0__P1.log",
                                              "runs/experiments/exp/logs/bce_f-2.log",
                                              "runs/ablations/abl/calibration_scaler.joblib")]
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert sorted(status) == sorted(f"?? {rel}" for rel in light)
    assert sorted(_git(repo, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()) == sorted(ignored)


# --------------------------------------------------------------------------- scripts/check_run_evidence.py


def test_a_status_citing_a_run_with_untracked_light_files_fails_and_names_them(cre, repo, capsys):
    _write(repo, "docs/STATUS.md", f"- Latest run `{RUN}`: test AUC 0.486.\n")
    light = _run(repo, f"runs/{RUN}")
    assert cre.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL {RUN} (cited in docs/STATUS.md)\n" in out
    for rel in light:
        assert f"  untracked {rel}\n" in out
    assert not any(name in out for name in HEAVY)          # heavy files never count as missing
    assert out.splitlines()[-1] == ("check_run_evidence: 1 cited run id in 1 file: 1 fail (0 without a run "
                                    "directory, 1 with 4 missing paths); stage a run's light files with "
                                    "--list-untracked")


def test_a_status_citing_a_tracked_run_passes_with_one_summary_line(cre, repo, capsys):
    _write(repo, "docs/STATUS.md", f"- Latest run `{RUN}`: test AUC 0.486.\n")
    light = _run(repo, f"runs/{RUN}")
    _git(repo, "add", "docs/STATUS.md", *light)
    _git(repo, "commit", "-q", "-m", "run")
    assert cre.main(["--repo", str(repo)]) == 0
    assert capsys.readouterr().out.splitlines() == ["check_run_evidence: 1 cited run id in 1 file, all tracked"]


def test_citations_in_notebook_outputs_and_run_reports_are_found_and_the_list_stages_their_runs(cre, repo, capsys):
    """A stream output with a Windows path, the widget state, and an ablation report that cites a cell
    with its suffix; --list-untracked prints exactly their untracked light files, and staging that
    list makes the check pass."""
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": 1, "id": "cell-00", "metadata": {},
                   "source": ["run = train()"],
                   "outputs": [{"output_type": "stream", "name": "stdout", "text": [f"run: ..\\runs\\{RUN}\n"]}]}],
        "metadata": {"widgets": {"application/vnd.jupyter.widget-state+json": {"state": {"w0": {"state": {
            "outputs": [{"output_type": "stream", "text": f"logged to ..\\runs\\experiments\\{EXP}-bce_f-2\n"}]}}}}}},
        "nbformat": 4, "nbformat_minor": 5}
    _write(repo, "notebooks/01_train_and_monitor.ipynb", json.dumps(notebook, indent=1))
    _write(repo, "runs/ablations/abl/report.md", f"| all_on | s0 | P1 | {CELL}-all_on__s0__P1 |\n")
    light = sorted(_run(repo, f"runs/{RUN}") + _run(repo, CELL_DIR) + _run(repo, EXP_DIR, ("config.yaml", "meta.json")))
    _write(repo, "runs/ablations/abl/cells/all_on__s0__P1.json")
    _write(repo, "runs/ablations/abl/logs/all_on__s0__P1.log")

    assert cre.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL {RUN} (cited in notebooks/01_train_and_monitor.ipynb)" in out
    assert f"FAIL {EXP} (cited in notebooks/01_train_and_monitor.ipynb)" in out
    assert f"FAIL {CELL} (cited in runs/ablations/abl/report.md)" in out

    assert cre.main(["--repo", str(repo), "--list-untracked"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed == light                                  # sorted, forward slashes, no heavy file, no log

    (repo.parent / "light.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
    _git(repo, "add", f"--pathspec-from-file={repo.parent / 'light.txt'}")
    assert cre.main(["--repo", str(repo)]) == 0
    assert capsys.readouterr().out.splitlines() == ["check_run_evidence: 3 cited run ids in 2 files, all tracked"]


def test_a_cited_id_without_a_run_directory_fails(cre, repo, capsys):
    """Also pins the scope: a summary.md under runs/ is read; a result.json and a .txt are not."""
    _write(repo, "README.md", f"Numbers from run {GONE}.\n")
    _write(repo, "docs/research/study/README.md", f"See `runs/{GONE}/eval_report_test.md`.\n")
    _write(repo, "runs/experiments/exp/summary.md", f"| bce_f-2 | {GONE}-bce_f-2 |\n")
    _write(repo, "docs/notes.md", f"{GONE}0 and x{GONE} are not run ids.\n")
    _write(repo, "runs/experiments/exp/bce_f-2/result.json", json.dumps({"run_id": EXP}))
    _write(repo, "docs/notes.txt", f"{EXP}\n")
    assert cre.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == [f"FAIL {GONE} (cited in README.md and 2 more files)",
                   "  no run directory under runs/, in git or on disk",
                   "check_run_evidence: 1 cited run id in 3 files: 1 fail (1 without a run directory, 0 with "
                   "0 missing paths); stage a run's light files with --list-untracked"]


def test_config_and_meta_must_be_tracked_even_when_nothing_light_is_untracked(cre, repo, capsys):
    """A run directory with only a tracked status.json, and one whose config.yaml is on disk but
    ignored, both fail on the files a cited run must have in git."""
    _write(repo, "runs/experiments/exp/REPORT.md", f"{RUN} and {EXP}-bce_f-2\n")
    _run(repo, f"runs/{RUN}", ("status.json",))
    _git(repo, "add", f"runs/{RUN}/status.json")
    _run(repo, EXP_DIR, ("config.yaml", "meta.json"))
    _git(repo, "add", f"{EXP_DIR}/meta.json")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text(f"{EXP_DIR}/config.yaml\n", encoding="utf-8")
    assert cre.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert f"  absent    runs/{RUN}/config.yaml\n  absent    runs/{RUN}/meta.json\n" in out
    assert f"  ignored   {EXP_DIR}/config.yaml\n" in out
    assert f"{EXP_DIR}/meta.json" not in out


def test_heavy_files_never_count_as_missing(cre, repo, capsys):
    _write(repo, "docs/STATUS.md", f"{RUN}\n")
    light = _run(repo, f"runs/{RUN}", LIGHT)
    _git(repo, "add", "docs/STATUS.md", *light)
    assert all((repo / f"runs/{RUN}" / name).is_file() for name in HEAVY)
    assert cre.main(["--repo", str(repo), "--list-untracked"]) == 0
    assert capsys.readouterr().out == ""
    assert cre.main(["--repo", str(repo)]) == 0


def test_a_directory_outside_any_git_work_tree_exits_2(cre, git_env, capsys):
    plain = git_env / "plain"
    plain.mkdir()
    assert cre.main(["--repo", str(plain)]) == 2
    assert "not in a git work tree" in capsys.readouterr().err


def test_every_run_cited_in_this_repository_is_tracked(cre):
    """The policy on this checkout (CI runs it): every run id cited in docs/, README.md, the run reports
    and the saved notebooks resolves to a run directory whose light files are tracked. A failure lists
    each failing run id, one file that cites it and the missing paths; stage them as docs/RUNBOOK.md
    "Run directories in git" says."""
    top = cre.toplevel(REPO)
    assert top is not None, f"{REPO} is not a git work tree"
    evidence = cre.collect(top)
    detail = evidence.report()[:-1]
    n_failing = len(evidence.failing)
    assert n_failing == 0, "\n".join([evidence.summary(), *detail[:40]]
                                     + ([f"... {len(detail) - 40} more lines"] if len(detail) > 40 else []))
