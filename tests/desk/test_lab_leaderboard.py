"""DESK-5.5: Lab leaderboard JSON (D-020: test columns are shown, flagged and never rank)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from neural_trade.desk.lab import leaderboard as lb
from neural_trade.experiments.leaderboard import LeaderboardRow, RoleAggregate, build_leaderboard
from neural_trade.experiments.store import RunStore

SCEN = "lb_scn"


def _write_run(root, *, configuration, fold, role, seed, sharpe, scenario=SCEN):
    run_id = f"{scenario}-{configuration}-f{fold}-s{seed}"
    d = Path(root) / "scenarios" / scenario / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id,
            "engine": {"scenario": scenario, "cell_key": f"{configuration}__f{fold}__s{seed}",
                       "configuration": configuration, "variant": configuration, "params": {}, "fold": fold,
                       "fold_id": abs(fold), "role": role, "seed": seed, "config_hash": "x", "settings_hash": "y",
                       "spec_hash": "z", "commit": "deadbee", "strategy": {"name": "calibrated_quantile", "params": {}},
                       "backtest": {}, "run": {"calibrate": True, "save_artifacts": False}},
            "dataset": {"path": "bars.csv", "sha256": "f" * 20, "size_bytes": 1, "n_bars": 100,
                        "first_timestamp": "2026-01-01T00:00:00+00:00", "last_timestamp": "2026-01-02T00:00:00+00:00"},
            "setup": {"bar_minutes": 1, "LOOKBACK": 60, "HORIZON_STEPS": [10, 15, 20]}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    scores = {"backtest/sharpe_net": sharpe, "backtest/total_return": 0.02, "backtest/max_drawdown": 0.05,
              "backtest/n_trades": 10.0, "backtest/buy_and_hold/total_return": 0.01,
              "backtest/random_same_freq/percentile_total_return": 80.0}
    result = {"status": "done", "error": None, "scores": scores, "finished_utc": "2026-01-01T00:00:00+00:00",
              "report": f"eval_report_{role}.json"}
    (d / result["report"]).write_text(json.dumps({"backtest": {"config": {
        "fill": "next_open", "fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0}}}), encoding="utf-8")
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _make_store(root, scenario=SCEN):
    """alpha: dev 0.5 / test 9; beta: dev 3 / test -9. The test column would reverse the dev order."""
    for conf, dev, test in (("alpha", 0.5, 9.0), ("beta", 3.0, -9.0)):
        for fold in (-3, -2):
            _write_run(root, configuration=conf, fold=fold, role="dev", seed=0, sharpe=dev, scenario=scenario)
        _write_run(root, configuration=conf, fold=-1, role="test", seed=0, sharpe=test, scenario=scenario)
    store = RunStore(root)
    store.sync(scenario)
    return store


def _dir_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_every_row_and_aggregate_field_is_in_the_json(tmp_path):
    _make_store(tmp_path)
    payload = lb.leaderboard(tmp_path, [SCEN])
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert set(row) == {f.name for f in dataclasses.fields(LeaderboardRow)}
        for split in ("dev", "test"):
            assert {f.name for f in dataclasses.fields(RoleAggregate)} <= set(row[split])
    json.dumps(payload, allow_nan=False)                       # strictly valid JSON


def test_fold_values_become_lists_and_round_trip(tmp_path):
    _make_store(tmp_path)
    row = lb.leaderboard(tmp_path, [SCEN])["rows"][0]
    fv = row["dev"]["fold_values"]["sharpe_net"]
    assert fv == [[-3, 3.0], [-2, 3.0]] and isinstance(fv, list)
    assert json.loads(json.dumps(row)) == row


def test_nan_inf_and_numpy_scalars_are_serialised():
    agg = RoleAggregate(n_folds=np.int64(1), n_rows=1, folds=(np.int64(-2),),
                        values={"sharpe_net": np.float64("nan"), "x": float("inf"), "y": np.float32(1.5)},
                        spread={"sharpe_net": None}, fold_values={"sharpe_net": {np.int64(-2): float("-inf")}})
    d = lb._aggregate(agg, "dev")
    assert d["values"] == {"sharpe_net": None, "x": None, "y": 1.5}
    assert d["fold_values"] == {"sharpe_net": [[-2, None]]} and d["folds"] == [-2]
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def test_test_columns_are_flagged_and_never_order_the_rows(tmp_path):
    _make_store(tmp_path)
    payload = lb.leaderboard(tmp_path, [SCEN])
    rows = payload["rows"]
    assert [r["configuration"] for r in rows] == ["beta", "alpha"]          # the dev order
    assert [r["rank"] for r in rows] == [1, 2]
    # the test column alone would give the opposite order
    assert rows[0]["test"]["values"]["sharpe_net"] < rows[1]["test"]["values"]["sharpe_net"]
    for r in rows:
        assert r["dev"]["rank_eligible"] is True and r["dev"]["split"] == "dev"
        assert r["test"]["rank_eligible"] is False and r["test"]["split"] == "test"
    assert payload["ranked_on"] == "dev" and payload["test_columns_rank"] is False
    assert "not used for ranking" in payload["test_label"]


def test_read_only_store_is_unchanged(tmp_path):
    _make_store(tmp_path)
    before = _dir_hash(tmp_path)
    lb.leaderboard(tmp_path, [SCEN])
    lb.etag(tmp_path, [SCEN])
    assert _dir_hash(tmp_path) == before


def test_etag_is_stable_per_store_and_query_and_changes_with_a_new_run(tmp_path):
    store = _make_store(tmp_path)
    a = lb.etag(tmp_path, [SCEN])
    assert a == lb.etag(tmp_path, [SCEN]) == lb.leaderboard(tmp_path, [SCEN])["etag"]
    assert lb.etag(tmp_path, [SCEN], specs_dir="other") != a
    assert lb.etag(tmp_path, [SCEN, "x"]) != a
    for fold in (-3, -2):
        _write_run(tmp_path, configuration="gamma", fold=fold, role="dev", seed=0, sharpe=1.0)
    store.sync(SCEN)
    assert lb.etag(tmp_path, [SCEN]) != a


def test_missing_store_or_empty_query_gives_an_empty_board(tmp_path):
    for payload in (lb.leaderboard(tmp_path / "nope", [SCEN]), lb.leaderboard(tmp_path, []),
                    lb.leaderboard(tmp_path, ["unknown"])):
        assert payload["rows"] == [] and payload["n_cells"] == 0 and payload["etag"]
    assert not (tmp_path / "nope").exists()                                # nothing was created


def test_key_order_is_deterministic(tmp_path):
    _make_store(tmp_path)
    a = lb.leaderboard(tmp_path, [SCEN])
    b = lb.leaderboard(tmp_path, [SCEN])
    assert json.dumps(a) == json.dumps(b)
    assert list(a["rows"][0]) == [f.name for f in dataclasses.fields(LeaderboardRow)]


def test_row_to_dict_does_not_mutate_the_row():
    rows = [{"scenario": "s", "configuration": "c", "fold": -2, "role": "dev", "seed": 0, "status": "done",
             "sharpe_net": 1.0, "total_return": 0.1, "max_drawdown": 0.0, "n_trades": 5.0,
             "buy_and_hold_return": 0.0, "random_percentile_return": 90.0}]
    row = build_leaderboard(rows)[0]
    snap = copy.deepcopy(row)
    out = lb.row_to_dict(row)
    assert row == snap
    assert math.isfinite(out["dev"]["values"]["sharpe_net"])
