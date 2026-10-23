"""Lab datasets (DESK-5.1): the export feeds the engine, the list never fingerprints a big file synchronously, the cache
key is (path, size, mtime), export names cannot escape the datasets directory."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from neural_trade.desk.data.store import MarketStore
from neural_trade.desk.lab import datasets as lab
from neural_trade.desk.lab.datasets import DatasetError, compute_fingerprint, export_dataset, list_datasets
from neural_trade.desk.store.db import open_db
from neural_trade.experiments.runner import Runner

SYM = "BTC/USDT"


@pytest.fixture
def market(tmp_path, synthetic_bars):
    with MarketStore(tmp_path / "market.sqlite") as m:
        m.write_bars(SYM, 1, synthetic_bars, "t")
        yield m


@pytest.fixture(autouse=True)
def _fresh_memory_cache():
    lab._MEMORY.clear()
    yield
    lab._MEMORY.clear()


def _export(market, home, name="tiny", **kw):
    return export_dataset(market, name, SYM, 1, None, None, home=home, **kw)


def _bars_csv(path: Path, synthetic_bars, n=300):
    synthetic_bars.head(n).to_csv(path, index=False)
    return path


def test_an_export_passes_the_engine_plan_on_a_copied_scenario(tmp_path, market):
    facts = _export(market, tmp_path / "home")
    scenario = {"schema_version": 1, "name": "tiny_desk", "description": "desk export",
                "overrides": {"CSV_PATH": facts["path"], "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32},
                "variants": {"default": {}}, "folds": [-2, -1], "seeds": [0],
                "strategy": {"name": "calibrated_quantile", "params": {}}, "backtest": {"random_seeds": 5},
                "run": {"calibrate": False, "save_artifacts": False}}
    spec_path = tmp_path / "tiny.yaml"
    spec_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    planned = Runner.from_spec(spec_path, store=tmp_path / "runs").plan()
    assert len(planned) == 2 and {pc.state for pc in planned} == {"pending"}
    # and the Lab list sees the export with the engine's own fingerprint
    [row] = [r for r in list_datasets(root=tmp_path / "empty", home=tmp_path / "home") if r["name"] == "tiny.csv"]
    assert row["source"] == "export" and row["fingerprint"]["sha256"] == facts["sha256"]
    assert row["spec"]["rows"] == facts["rows"] == 3000 and row["spec"]["bar_minutes"] == 1.0


def test_the_list_skips_csvs_that_are_not_ohlcv(tmp_path, synthetic_bars):
    _bars_csv(tmp_path / "bars.csv", synthetic_bars)
    (tmp_path / "params.csv").write_text("epoch,period\n1,14\n", encoding="utf-8")
    names = [r["name"] for r in list_datasets(root=tmp_path, home=tmp_path / "h")]
    assert names == ["bars.csv"]


def test_the_default_roots_include_the_bundled_file():
    assert any(p.name == "binance_btcusdt_1min_ccxt.csv" for p in lab.dataset_dirs()[0].glob("*.csv"))


def test_a_file_over_the_limit_is_pending_and_never_fingerprinted_synchronously(tmp_path, synthetic_bars, monkeypatch):
    csv = _bars_csv(tmp_path / "bars.csv", synthetic_bars)
    calls = []
    monkeypatch.setattr(lab, "compute_fingerprint", lambda *a, **k: calls.append(a) or {})
    [row] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=csv.stat().st_size - 1)
    assert row["fingerprint_pending"] is True and row["fingerprint"] is None and calls == []
    assert row["spec"]["rows"] is None and row["spec"]["bar_minutes"] == 1.0


def test_compute_fingerprint_fills_the_cache_and_the_next_list_returns_it(tmp_path, synthetic_bars):
    csv = _bars_csv(tmp_path / "bars.csv", synthetic_bars)
    small = csv.stat().st_size - 1
    [pending] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=small)
    assert pending["fingerprint_pending"]
    fp = compute_fingerprint(csv)
    assert fp["n_bars"] == 300 and "error" not in fp
    [row] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=small)
    assert row["fingerprint_pending"] is False and row["fingerprint"]["sha256"] == fp["sha256"]
    assert row["spec"]["rows"] == 300


def test_a_touched_file_invalidates_the_cache(tmp_path, synthetic_bars):
    csv = _bars_csv(tmp_path / "bars.csv", synthetic_bars)
    small = csv.stat().st_size - 1
    compute_fingerprint(csv)
    [row] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=small)
    assert row["fingerprint_pending"] is False
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))      # same size, new mtime
    [row] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=small)
    assert row["fingerprint_pending"] is True
    _bars_csv(csv, synthetic_bars, n=299)                                    # new size
    [row] = list_datasets(root=tmp_path, home=tmp_path / "h", sync_limit_bytes=csv.stat().st_size - 1)
    assert row["fingerprint_pending"] is True


def test_the_cache_lives_in_the_desk_settings_table(tmp_path, synthetic_bars):
    csv = _bars_csv(tmp_path / "bars.csv", synthetic_bars)
    with open_db(tmp_path / "desk.sqlite") as db:
        [row] = list_datasets(db.settings, root=tmp_path, home=tmp_path / "h")
        assert not row["fingerprint_pending"]
        assert any(k.startswith("lab.fingerprint:") for k in db.settings.all())
        lab._MEMORY.clear()
        small = csv.stat().st_size - 1
        [again] = list_datasets(db.settings, root=tmp_path, home=tmp_path / "h", sync_limit_bytes=small)
        assert again["fingerprint"] == row["fingerprint"]


@pytest.mark.parametrize("name", ["../evil", "..", "a/b", "a\\b", "/abs/path", "C:/x", "C:\\x", "", ".hidden", "x" * 200,
                                  "a/../../b", "..\\..\\x"])
def test_export_names_that_could_escape_are_refused(tmp_path, market, name):
    home = tmp_path / "home"
    with pytest.raises(DatasetError):
        _export(market, home, name)
    assert not any(tmp_path.rglob("evil*")) and not (home / "datasets").exists()


def test_export_keeps_an_existing_file_unless_asked(tmp_path, market):
    home = tmp_path / "home"
    first = _export(market, home, "x.csv")
    assert Path(first["path"]) == (home / "datasets" / "x.csv").resolve()
    before = Path(first["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        _export(market, home, "x")
    assert Path(first["path"]).read_bytes() == before
    assert _export(market, home, "x", overwrite=True)["rows"] == 3000
