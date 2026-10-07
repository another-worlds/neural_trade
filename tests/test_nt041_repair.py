# ruff: noqa: F811
"""NT-041 repair round 1: the board's cost profile, an index made before the setup columns, the hole policy that must
not move a block boundary (and the resume rule for cells made before it), and the label details."""
from __future__ import annotations

import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.experiments.dataset import data_layout
from neural_trade.experiments.leaderboard import GuardRailSpec, leaderboard_for_scenario, winner
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.store import RUN_COLUMNS, RunIndex, RunStore
from tests.test_data_plan import DAY, _file, _timed  # noqa: F401
from tests.test_experiment_engine import FakeTrainer, bars_csv, spec  # noqa: F401

RELAXED = GuardRailSpec(min_trades=None, require_beat_buy_and_hold=False, require_beat_random_null=False)


def _run(tmp_path, csv, name="tiny", **changes):
    sc = Scenario.from_dict(spec(csv, name=name, **changes))
    store = RunStore(tmp_path / f"runs_{name}")
    Runner(sc, store.root, trainer=FakeTrainer()).run()
    return sc, store


def _rail(row, name):
    return next(g for g in row.guard_rails if g.name == name)


# ---------------------------------------------------------------- P1-A: the board ranks at the effective profile
@pytest.mark.parametrize("where", ["config", "backtest"])
def test_the_board_ranks_at_the_scenarios_effective_costs_so_its_rows_are_comparable(tmp_path, bars_csv, where):
    over = {"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32}
    changes = {}
    if where == "config":
        over["FEE_BPS"] = 8.0
        changes["overrides"] = over
    else:
        changes["backtest"] = {"random_seeds": 5, "fee_bps": 8.0}
    sc, store = _run(tmp_path, bars_csv, name=f"costs_{where}", **changes)
    board = leaderboard_for_scenario(store, sc.name, guard_rails=RELAXED, spec=sc)
    assert board and all(_rail(r, "cost_profile").passed for r in board), \
        [_rail(r, "cost_profile").detail for r in board]
    assert board[0].board_cost.fee_bps == 8.0 and board[0].cost_profile.fee_bps == 8.0
    assert winner(board) is not None
    # the meta records the EFFECTIVE costs the cells were scored at (not 0 when they came from backtest:)
    for row in store.index.rows():
        meta = json.loads((store.root / row["run_dir"] / "meta.json").read_text(encoding="utf-8"))
        assert meta["setup"]["cost_profile"]["fee_bps"] == 8.0


def test_a_cell_at_another_profile_is_not_comparable(tmp_path, bars_csv):
    over = {"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32, "FEE_BPS": 8.0}
    sc, store = _run(tmp_path, bars_csv, name="mixed", overrides=over,
                     variants={"default": {}, "cheap": {"FEE_BPS": 2.0}}, folds=[-1], seeds=[0])
    board = {r.configuration: r for r in leaderboard_for_scenario(store, "mixed", guard_rails=RELAXED, spec=sc)}
    assert _rail(board["default"], "cost_profile").passed
    assert not _rail(board["cheap"], "cost_profile").passed and "not comparable" in _rail(board["cheap"], "cost_profile").detail


def test_the_default_board_stays_at_zero_costs(tmp_path, bars_csv):
    sc, store = _run(tmp_path, bars_csv, name="zero")
    board = leaderboard_for_scenario(store, "zero", guard_rails=RELAXED, spec=sc)
    assert all(_rail(r, "cost_profile").passed and r.board_cost.per_side == 0 for r in board)


# ---------------------------------------------------------------- P1-B: an index made before the setup columns
NEW = ("symbol", "window_minutes", "horizon_minutes")


def _old_index(path, rows=2):
    old = [c for c in RUN_COLUMNS if c[0] not in NEW]
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE runs ({', '.join(f'{c} {t}' for c, t in old)})")
    con.execute("CREATE TABLE scores (run_id TEXT NOT NULL, name TEXT NOT NULL, value REAL, PRIMARY KEY (run_id, name))")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    for i in range(rows):
        vals = {c: None for c, _ in old}
        vals.update(run_id=f"r{i}", run_dir=f"d{i}", scenario="s", cell_key=f"k{i}", status="done", fold=-1, role="dev",
                    seed=i)
        con.execute(f"INSERT INTO runs ({', '.join(vals)}) VALUES ({', '.join('?' for _ in vals)})", list(vals.values()))
    con.commit()
    con.close()
    return path


def test_every_open_of_an_old_index_adds_the_new_columns(tmp_path):
    path = _old_index(tmp_path / "index.sqlite")
    rows = RunIndex(path).rows()                       # a read: it used to raise 'no such column: symbol'
    assert len(rows) == 2 and all(r["symbol"] is None and r["window_minutes"] is None for r in rows)
    con = sqlite3.connect(path)
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    con.close()
    assert set(NEW) <= have
    assert len(RunIndex(path).rows()) == 2             # idempotent


def test_a_read_only_old_index_is_read_with_n_a_for_the_new_columns(tmp_path):
    import os
    import stat

    path = _old_index(tmp_path / "ro.sqlite")
    os.chmod(path, stat.S_IREAD)
    try:
        rows = RunIndex(path).rows()
        assert len(rows) == 2 and rows[0]["symbol"] is None
        assert RunIndex(path).dump()["runs"]
    finally:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def test_the_leaderboard_and_the_panel_read_an_old_index_without_syncing(tmp_path):
    from neural_trade.notebook import panel_data

    root = tmp_path / "runs"
    root.mkdir()
    _old_index(root / "index.sqlite")
    store = RunStore(root)
    assert leaderboard_for_scenario(store, "s", sync=False) is not None
    assert panel_data.read_board_rows(store, ["s"]) is not None


def test_a_copy_of_the_real_old_index_opens(tmp_path):
    real = pytest.importorskip("pathlib").Path("D:/nt/neural_trade/runs/index.sqlite")
    if not real.is_file():
        pytest.skip("no real index on this machine")
    con = sqlite3.connect(real)
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    con.close()
    if "symbol" in cols:
        pytest.skip("the real index already has the new columns")
    dst = tmp_path / "index.sqlite"
    shutil.copy(real, dst)
    rows = RunIndex(dst).rows()
    assert rows and all(r["symbol"] is None for r in rows)


# ---------------------------------------------------------------- P1-C: a hole removes windows, never moves a block
HOLES = ((30 * DAY + 5, 180), (80 * DAY + 100, 700))


def _boundaries(layout):
    return [{n: (b["first_timestamp"], b["last_timestamp"]) for n, b in f["blocks"].items()} for f in layout.folds]


def test_a_hole_does_not_move_any_block_boundary(tmp_path):
    path = _file(tmp_path / "holey.csv", days=100, holes=HOLES)
    kw = dict(CSV_PATH=str(path), MAX_SEQUENCE_COUNT=60_000, N_FOLDS=4)
    dropped = data_layout(Config(GAP_POLICY="drop", **kw))
    today = data_layout(Config(GAP_POLICY="ignore", **kw))          # the grid exactly as before the gap policy
    assert _boundaries(dropped) == _boundaries(today)
    assert [f["gap"] for f in dropped.folds] == [f["gap"] for f in today.folds]
    n_dropped = 0
    for fd, ft in zip(dropped.folds, today.folds):
        for name in ("train", "val", "cal", "test"):
            b, t = fd["blocks"][name], ft["blocks"][name]
            assert b["n"] + b["n_dropped"] == t["n"] and (b["start"], b["stop"]) == (t["start"], t["stop"])
            n_dropped += b["n_dropped"]
        assert fd["windows_dropped"] == {k: v["n_dropped"] for k, v in fd["blocks"].items()}
    assert n_dropped > 0
    assert dropped.fingerprint["gaps"]["n_windows_dropped"] > 0


def test_the_trainer_blocks_equal_the_layouts_blocks_with_a_hole(tmp_path):
    from neural_trade.data.processor import split_arrays

    path = _file(tmp_path / "holey.csv", days=100, holes=((98 * DAY - 3000, 300),))
    cfg = Config(CSV_PATH=str(path), MAX_SEQUENCE_COUNT=60_000, N_FOLDS=4)
    layout = data_layout(cfg)
    arrays = split_arrays(cfg)
    ts = arrays["df"]["timestamp"]
    for name in ("train", "val", "cal", "test"):
        b = layout.fold(-1)["blocks"][name]
        assert len(arrays[name]["X"]) == b["n"]
        first = ts.iloc[int(arrays[name]["anchor_bar"][0])].isoformat()
        assert first[:16] >= b["first_timestamp"][:16]


def test_cells_made_before_the_gap_policy_are_not_done_when_a_hole_is_in_their_blocks(tmp_path):
    # the hole sits in the newest fold's blocks only (fold -1); fold -2 reads no hole
    path = _file(tmp_path / "holey.csv", days=100, holes=((97 * DAY, 90),))
    changes = dict(overrides={"CSV_PATH": str(path), "MAX_SEQUENCE_COUNT": 60_000, "EPOCHS": 1, "BATCH_SIZE": 32,
                              "N_FOLDS": 4, "VAL_FRACTION": 0.05, "CAL_FRACTION": 0.05}, seeds=[0], folds=[-2, -1])
    sc = Scenario.from_dict(spec(path, name="resume", **changes))
    root = tmp_path / "runs"
    first = Runner(sc, root, trainer=FakeTrainer()).run()
    assert {r["status"] for r in first.ran} == {"done"}
    states = {pc.cell.fold: pc.state for pc in Runner(sc, root, trainer=FakeTrainer()).plan()}
    assert states == {-2: "done", -1: "done"}                       # made with the policy: matched
    # the scorer backtested the block that holds the hole (it used to refuse: anchors not consecutive)
    from neural_trade.experiments.scorer import load_block
    oos = [d for d in (root / "scenarios" / "resume").glob("*f-1__s0*")]
    assert oos and load_block(oos[0] / "predictions_oos.npz")[1].breaks is not None
    # age the cells: remove the record the policy writes, as a run made before it lacks it
    for meta_path in (root / "scenarios" / "resume").glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["dataset"].pop("gaps", None)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    trainer = FakeTrainer()
    planned = Runner(sc, root, trainer=trainer).plan()
    states = {pc.cell.fold: pc.state for pc in planned}
    dropped = {pc.cell.fold: sum(pc.fold["windows_dropped"].values()) for pc in planned}
    assert dropped[-1] > 0 and dropped[-2] == 0
    assert states == {-2: "done", -1: "pending"}                    # only the fold that reads the hole is re-run
    Runner(sc, root, trainer=trainer).run()
    assert trainer.calls == [pc.key for pc in planned if pc.cell.fold == -1]


# ---------------------------------------------------------------- P2
def test_the_trade_panel_titles_follow_the_spec_not_the_import():
    from neural_trade.visualization import labels as L
    from neural_trade.visualization.trade_analytics import _panels

    assert any("USDT" in t for t, _ in _panels())
    with L.using(Config(SYMBOL="ETH/EUR", QUOTE_CURRENCY="EUR")):
        titles = [t for t, _ in _panels()]
    assert any("EUR" in t for t in titles) and not any("USDT" in t for t in titles)


def test_the_dashboard_margin_leaves_room_for_the_quote_code():
    import inspect

    from neural_trade.visualization import trading_dashboard as TD

    assert "len(L.quote())" in inspect.getsource(TD.trading_dashboard_figure)


def test_the_baseline_caption_names_the_block_the_report_used():
    from neural_trade.visualization.analytics_tables import baseline_table

    rep = {"beats_baseline": {"zero_delta": {"delta/rmse": {"h0": True}}},
           "baseline_margins": {"zero_delta": {"delta/rmse": {"h0": {"model": 1.0, "baseline": 2.0, "margin": 1.0,
                                                                        "dm_z": 3.0}}}},
           "meta": {"noise_tests": {"dm_z": "x"}, "boot_block": 960}}
    assert "960-bar blocks" in baseline_table(rep).attrs["caption"]


# ---------------------------------------------------------------- the backtest does not trade across a hole
def test_bars_mark_a_jump_of_the_anchors_as_a_break_and_the_engine_closes_before_it():
    from neural_trade.strategy import Bars

    df = pd.DataFrame({"Open": np.arange(100.0, 160.0), "High": np.arange(101.0, 161.0),
                       "Low": np.arange(99.0, 159.0), "Close": np.arange(100.5, 160.5)})
    anchors = np.r_[np.arange(0, 20), np.arange(35, 55)]            # bars 20..34 were a hole's dropped windows
    bars = Bars.from_frame(df, anchors)
    assert bars.breaks is not None and bars.breaks.sum() == 1 and bool(bars.breaks[19])
    assert Bars.from_frame(df, np.arange(10)).breaks is None
    assert bars.slice(10).breaks is None and bars.slice(30).breaks.sum() == 1
    with pytest.raises(ValueError, match="increase"):
        Bars.from_frame(df, np.array([3, 2, 1]))
