"""NT-044: the landing page, the guides and ARCHITECTURE.md say what the code does.

What is checked (the docs are for the owner and reviewers, so a stale claim is a defect):

* every relative link in README.md, docs/ARCHITECTURE.md and docs/guide/*.md resolves, and so does every
  repository path written in backticks under configs/, scripts/, src/, docs/, notebooks/, plugins/;
* docs/ARCHITECTURE.md lists every subpackage of src/neural_trade;
* docs/guide/reading-figures.md names every Visualizations key and every figure function that
  scripts/notebooks/build.py calls;
* every `neural-trade ...` command in README.md and the guides is accepted by the CLI's argument parser;
* the worked example of docs/guide/experiments.md runs on a tiny CPU scenario with the same name and keys as
  configs/scenarios/reference.yaml (the commands that need no training in the fast suite, the whole path
  with real training in the slow test; `compare` is only parsed, see the guide);
* the statements of docs/guide/own-data.md about bar size, wall-clock lengths and the stability harness.
"""
from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "neural_trade"
GUIDE_DIR = REPO / "docs" / "guide"
DOCS = [REPO / "README.md", REPO / "docs" / "ARCHITECTURE.md", *sorted(GUIDE_DIR.glob("*.md"))]
NEW_GUIDES = ["concepts.md", "reading-figures.md", "experiments.md", "own-data.md"]
PATH_ROOTS = ("configs/", "scripts/", "src/", "docs/", "notebooks/", "plugins/", "tests/")

LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```", re.S | re.M)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _commands(path: Path, info: str | None = None):
    """The `neural-trade ...` commands of a document's fenced code blocks (continued lines joined, trailing
    comments dropped), as argv lists. ``info`` keeps only blocks whose info string contains it."""
    out = []
    for m in FENCE.finditer(_text(path)):
        if info is not None and info not in m.group("info"):
            continue
        body = m.group("body").replace("\\\n", " ")
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("neural-trade "):
                continue
            line = re.sub(r"\s+#\s.*$", "", line).replace("<run id>", "RUN_ID")
            out.append(shlex.split(line)[1:])
    return out


# ------------------------------------------------------------------------------- the files exist
def test_the_guides_and_architecture_exist_and_the_readme_links_to_them():
    for name in NEW_GUIDES:
        assert (GUIDE_DIR / name).is_file(), name
    assert (REPO / "docs" / "ARCHITECTURE.md").is_file()
    readme = _text(REPO / "README.md")
    for target in [*(f"docs/guide/{n}" for n in NEW_GUIDES), "docs/ARCHITECTURE.md", "docs/STATUS.md"]:
        assert f"]({target})" in readme, f"README.md does not link to {target}"


# ------------------------------------------------------------------------------- links and paths
@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_relative_link_resolves(doc):
    bad = []
    for target in LINK.findall(_text(doc)):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        if not (doc.parent / path).resolve().exists():
            bad.append(target)
    assert not bad, f"{doc.name}: links that do not resolve: {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_repository_path_written_in_backticks_exists(doc):
    bad = []
    for token in BACKTICK.findall(_text(doc)):
        token = token.strip().rstrip(",.;:")
        if not token.startswith(PATH_ROOTS) or " " in token or any(c in token for c in "<>*{}|"):
            continue
        if "..." in token:
            continue
        if not (REPO / token).exists():
            bad.append(token)
    assert not bad, f"{doc.name}: paths that do not exist: {bad}"


# ------------------------------------------------------------------------------- ARCHITECTURE.md
def test_architecture_lists_every_subpackage():
    subpackages = sorted(p.name for p in SRC.iterdir()
                         if p.is_dir() and not p.name.startswith(("__", ".")) and (p / "__init__.py").exists())
    assert len(subpackages) >= 15
    text = _text(REPO / "docs" / "ARCHITECTURE.md")
    rows = {m.group(1) for m in re.finditer(r"^\|\s*`([a-z_]+)/`\s*\|", text, re.M)}
    missing = [p for p in subpackages if p not in rows]
    extra = sorted(rows - set(subpackages))
    assert not missing, f"ARCHITECTURE.md does not list: {missing}"
    assert not extra, f"ARCHITECTURE.md lists subpackages that do not exist: {extra}"


def test_architecture_states_the_layering_rules_the_layering_test_enforces():
    text = _text(REPO / "docs" / "ARCHITECTURE.md")
    layering = _text(REPO / "tests" / "test_layering.py")
    assert "test_no_subpackage_imports_another_that_imports_it_back" in layering
    assert "test_evaluation_does_not_import_training_or_experiments" in layering
    assert "test_data_does_not_import_visualization" in layering
    assert "tests/test_layering.py" in text
    for must in ("no two subpackages import each other", "`evaluation/` does not import `training/` or `experiments/`",
                 "`data/` does not import `visualization/`"):
        assert must in text, must


def test_architecture_lists_every_registry():
    from neural_trade.registries import all_registries

    text = _text(REPO / "docs" / "ARCHITECTURE.md")
    names = set(all_registries()) | {"Strategies"}
    missing = [n for n in sorted(names) if f"`{n}`" not in text]
    assert not missing, f"ARCHITECTURE.md does not name the registries {missing}"


# ------------------------------------------------------------------------------- reading-figures.md
FIGURE_CALL = re.compile(r"\b(\w*_figure|indicator_applied_periods|dashboard|trade_analytics|compare_strategies|"
                         r"figures|board|comparison|training_health_html|show_progress|show_results)\(")
BUILD_KEY = re.compile(r'Visualizations\.build\(\s*"(\w+)"')


def test_reading_figures_names_every_visualization_and_every_notebook_figure_function():
    from neural_trade.registries.visualizations import Visualizations

    guide = _text(GUIDE_DIR / "reading-figures.md")
    build = _text(REPO / "scripts" / "notebooks" / "build.py")
    keys = set(Visualizations.registry)
    assert len(keys) >= 27
    called = set(FIGURE_CALL.findall(build)) | set(BUILD_KEY.findall(build))
    assert {"curves_figure", "dashboard", "board", "split_overview_figure"} <= called   # the regex still sees them
    missing = sorted(n for n in keys | called if f"`{n}`" not in guide and f"`{n}(" not in guide
                     and not re.search(rf"`[^`\n]*\b{re.escape(n)}\b[^`\n]*`", guide))
    assert not missing, f"docs/guide/reading-figures.md does not name: {missing}"


# ------------------------------------------------------------------------------- commands
ALL_COMMAND_DOCS = [REPO / "README.md", *sorted(GUIDE_DIR.glob("*.md"))]


def test_every_documented_neural_trade_command_is_accepted_by_the_cli_parser(capsys):
    from neural_trade.cli import build_parser

    parser = build_parser()
    seen = 0
    for doc in ALL_COMMAND_DOCS:
        for argv in _commands(doc):
            try:
                parser.parse_args(argv)
            except SystemExit as exc:                                            # pragma: no cover - failure path
                raise AssertionError(f"{doc.name}: the CLI refuses `neural-trade {' '.join(argv)}`: "
                                     f"{capsys.readouterr().err[-300:]}") from exc
            seen += 1
    assert seen >= 15


def test_the_readme_quick_start_commands_cover_the_entry_points():
    verbs = {argv[0] for argv in _commands(REPO / "README.md")}
    assert {"train", "predict", "backtest", "scenario", "leaderboard", "sweep", "registry", "env"} <= verbs


def test_the_cheap_readme_commands_run(capsys):
    from neural_trade.cli import main

    assert main(["registry", "list"]) == 0
    assert "Visualizations" in capsys.readouterr().out
    assert main(["env", "--no-devices"]) == 0


# ------------------------------------------------------------------------------- the worked example
@pytest.fixture(scope="module")
def tiny_reference(tmp_path_factory, synthetic_bars):
    """configs/scenarios/reference.yaml shrunk to a CPU test: the same name and keys, the synthetic bars, two folds
    (one dev, one test), one seed, one epoch."""
    root = tmp_path_factory.mktemp("worked_example")
    csv = root / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    spec = yaml.safe_load(_text(REPO / "configs" / "scenarios" / "reference.yaml"))
    assert spec["name"] == "reference_default"
    spec = copy.deepcopy(spec)
    spec["base_config"] = str(REPO / "configs" / "default.yaml")
    spec["overrides"] = {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32}
    spec["folds"], spec["seeds"] = [-2, -1], [0]
    spec["backtest"] = {"random_seeds": 5}
    spec["run"] = {"calibrate": False, "save_artifacts": False}
    path = root / "reference.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _worked_example():
    cmds = _commands(GUIDE_DIR / "experiments.md", info="worked-example")
    assert [c[0] for c in cmds] == ["scenario", "scenario", "leaderboard", "sweep", "sweep", "compare"]
    return cmds


def _tiny(argv, spec, store):
    """The documented command with the reference scenario replaced by the tiny one and a temporary store."""
    out = [str(spec) if a == "configs/scenarios/reference.yaml" else a for a in argv]
    if out[0] in ("scenario", "sweep", "leaderboard"):
        out += ["--store", str(store)]
    return out


def test_the_compare_example_spec_parses_and_its_command_is_accepted():
    from neural_trade.cli import build_parser
    from neural_trade.experiments.comparator import CompareSpec

    cmd = next(c for c in _worked_example() if c[0] == "compare")
    spec = CompareSpec.from_yaml(REPO / cmd[1])
    assert spec.min_folds >= 5 and len(spec.judgment_folds) >= 5
    build_parser().parse_args(cmd)


def test_the_worked_example_without_training_runs_on_a_tiny_scenario(tiny_reference, tmp_path, monkeypatch, capsys):
    from neural_trade.cli import main

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")            # a CPU run: the sweep skips its GPU-free check
    store = tmp_path / "runs"
    plan, _run, board, quick, optuna, _compare = _worked_example()
    assert main(_tiny(plan, tiny_reference, store)) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {"done": 0, "failed": 0, "pending": 2}
    assert main(_tiny(board, tiny_reference, store)) == 0
    assert "Leaderboard" in capsys.readouterr().out                 # an empty board, not an error
    assert main(_tiny(quick, tiny_reference, store)) == 0
    assert "quick sweep reference_default-quick" in capsys.readouterr().out
    assert main(_tiny(optuna, tiny_reference, store)) == 0
    assert "optuna sweep reference_default-optuna: GPU budget" in capsys.readouterr().out


@pytest.mark.slow
def test_the_worked_example_trains_ranks_and_sizes_sweeps_on_a_tiny_cpu_scenario(tiny_reference, tmp_path,
                                                                                monkeypatch, capsys):
    from neural_trade.cli import main

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    store = tmp_path / "runs"
    plan, run, board, quick, optuna, _compare = _worked_example()
    assert main(_tiny(plan, tiny_reference, store)) == 0
    capsys.readouterr()
    assert main(_tiny(run, tiny_reference, store)) == 0
    ran = json.loads(capsys.readouterr().out)
    assert len(ran["ran"]) == 2 and not ran["failed"]
    assert main(_tiny(board, tiny_reference, store)) == 0
    out = capsys.readouterr().out
    assert "reference_default" in out and "not used for ranking" in out
    # with a finished cell of the same setup the sweep sizes itself from the index as well as from --sec-per-step
    assert main(_tiny(quick, tiny_reference, store)) == 0
    assert "estimated" in capsys.readouterr().out
    assert main(_tiny(optuna, tiny_reference, store)) == 0
    assert "GPU budget" in capsys.readouterr().out


# ------------------------------------------------------------------------------- own-data.md
@pytest.fixture(scope="module")
def five_minute_csv(tmp_path_factory, synthetic_bars):
    df = synthetic_bars.set_index("datetime").resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    path = tmp_path_factory.mktemp("five_minute") / "bars_5min.csv"
    df.to_csv(path, index=False)
    return path


def test_own_data_wall_clock_lengths_become_bars_and_a_fraction_of_a_bar_is_refused():
    from neural_trade.core.config import Config
    from neural_trade.core.dataset_spec import DatasetSpec
    from neural_trade.core.exceptions import InvalidConfigurationError

    cfg = Config().override(RESAMPLE_MINUTES=5, WINDOW_MINUTES=60, HORIZON_MINUTES=[10, 15, 20],
                            EXTENDED_TREND_MINUTES=[10, 15, 20])
    spec = DatasetSpec.from_config(cfg)
    assert (spec.bar_minutes, spec.window_bars, spec.horizon_bars) == (5.0, 12, [2, 3, 4])
    assert spec.cost_profile == {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0}
    with pytest.raises(InvalidConfigurationError, match="whole number of 5-minute bars"):
        Config().override(RESAMPLE_MINUTES=5, WINDOW_MINUTES=62)
    with pytest.raises(InvalidConfigurationError, match="three"):
        Config().override(HORIZON_STEPS=[5, 10, 15, 20], EXTENDED_TREND_PERIODS=[5, 10, 15, 20])


def test_own_data_a_bar_size_that_does_not_match_the_file_is_refused(five_minute_csv):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays

    ok = split_arrays(Config().override(CSV_PATH=str(five_minute_csv), RESAMPLE_MINUTES=5, MAX_SEQUENCE_COUNT=0))
    assert ok["train"]["X"].shape[1] == 60
    with pytest.raises(ValueError, match="declared bar size is 1 minutes .* median bar spacing is 5 minutes"):
        split_arrays(Config().override(CSV_PATH=str(five_minute_csv), RESAMPLE_MINUTES=1, MAX_SEQUENCE_COUNT=0))


def test_own_data_the_csv_columns_and_epoch_timestamps_are_read_as_documented(tmp_path, synthetic_bars):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays

    df = synthetic_bars.copy()
    df["timestamp"] = (df["datetime"].astype("int64") // 10**6).astype("int64")      # epoch milliseconds
    path = tmp_path / "epoch_ms.csv"
    df.drop(columns=["datetime"]).to_csv(path, index=False)
    arrays = split_arrays(Config().override(CSV_PATH=str(path), MAX_SEQUENCE_COUNT=0))
    assert arrays["train"]["X"].shape[1] == 60

    capitalised = df.drop(columns=["datetime"]).rename(columns=str.capitalize)    # Open, High, Low, Close, Volume
    capitalised = capitalised.rename(columns={"Timestamp": "timestamp"})
    capitalised.to_csv(tmp_path / "capitalised.csv", index=False)
    assert split_arrays(Config().override(CSV_PATH=str(tmp_path / "capitalised.csv"),
                                          MAX_SEQUENCE_COUNT=0))["train"]["X"].shape[1] == 60
    shouting = capitalised.rename(columns=str.upper).rename(columns={"TIMESTAMP": "timestamp"})
    shouting.to_csv(tmp_path / "shouting.csv", index=False)                       # CLOSE: not a name the loader knows
    with pytest.raises((ValueError, KeyError)):
        split_arrays(Config().override(CSV_PATH=str(tmp_path / "shouting.csv"), MAX_SEQUENCE_COUNT=0))


def test_own_data_the_stability_harness_refuses_a_file_that_is_not_one_minute_bars(five_minute_csv, tmp_path, capsys):
    from neural_trade.cli import main

    rc = main(["stability", "--dry-run", "--csv", str(five_minute_csv), "--store", str(tmp_path / "runs"),
               "--cases", "control"])
    captured = capsys.readouterr()
    assert rc != 0 and not (tmp_path / "runs").exists()
    assert "declared bar size is 1 minutes" in captured.out + captured.err


def test_own_data_names_the_thresholds_and_exit_codes_the_harness_has():
    text = _text(GUIDE_DIR / "own-data.md")
    harness = _text(SRC / "experiments" / "stability.py")
    assert (REPO / "configs" / "stability_thresholds.yaml").is_file()
    assert (REPO / "configs" / "stability_thresholds_v2.yaml").is_file()
    assert "stability_failing_regions.json" in text
    assert "--thresholds" in _text(SRC / "cli.py") and "v2" in text
    assert "NT-051" in harness and "NT-051" in text


def test_the_readme_points_to_status_and_the_source_of_its_one_dated_snapshot():
    # the live state is STATUS; the README quotes one dated snapshot and names its source (NT-044 acceptance 5)
    readme = _text(REPO / "README.md")
    assert "docs/STATUS.md" in readme and "runs/experiments/capacity_v1/REPORT.md" in readme
    assert "(2026-10-06)" in readme
    assert not re.search(r"\d+ (?:tests? )?passed", readme)
