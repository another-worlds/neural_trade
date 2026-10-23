"""DESK-0.8: the repo CSV importer (labelled source, provenance, idempotence, no partial write)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neural_trade.desk.data.importers import default_src, import_csv
from neural_trade.desk.data.store import MarketStore

REPO = Path(__file__).resolve().parents[2]
BUNDLED = REPO / "binance_btcusdt_1min_ccxt.csv"


@pytest.fixture
def store(tmp_path):
    with MarketStore(tmp_path / "market.sqlite") as s:
        yield s


def _csv(path, n=10, **over):
    ts = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"), "open": 100.0, "high": 101.0,
                       "low": 99.0, "close": 100.5, "volume": 2.0})
    for k, v in over.items():
        df[k] = v
    df.to_csv(path, index=False)
    return path


@pytest.mark.data
def test_bundled_csv_imports_43500_bars_under_the_repo_label(store):
    res = import_csv(store, BUNDLED)
    assert res["n_bars"] == 43500 and res["src"] == "repo:binance_btcusdt_1min_ccxt.csv"
    rng = store.ranges("BTC/USDT", 1)
    assert [(r["src"], r["n_bars"]) for r in rng] == [("repo:binance_btcusdt_1min_ccxt.csv", 43500)]
    assert len(store.read_bars("BTC/USDT", 1)) == 43500


def test_default_provenance_names_file_and_span_and_survives_later_writes(store, tmp_path):
    f = _csv(tmp_path / "a.csv")
    res = import_csv(store, f)
    text = store.provenance("BTC/USDT", 1)["repo:a.csv"]
    assert text == res["provenance"] and "a.csv" in text and "10 bars" in text
    assert "2025-01-01T00:00:00+00:00" in text and "2025-01-01T00:09:00+00:00" in text
    # a later bar write (rebuilds sources) leaves the provenance alone
    store.write_bars("BTC/USDT", 1, pd.read_csv(f).assign(timestamp=lambda d: d["timestamp"].str.replace("2025-01-01", "2025-02-01")), "other")
    assert store.provenance("BTC/USDT", 1)["repo:a.csv"] == text


def test_explicit_provenance_and_src_are_carried(store, tmp_path):
    import_csv(store, _csv(tmp_path / "b.csv"), src="long", provenance="Bitstamp BTC/USD, shifted timestamps")
    assert store.provenance("BTC/USDT", 1) == {"long": "Bitstamp BTC/USD, shifted timestamps"}
    assert store.ranges("BTC/USDT", 1)[0]["src"] == "long"


def test_second_import_is_idempotent(store, tmp_path):
    f = _csv(tmp_path / "c.csv")
    import_csv(store, f)
    before = store.read_bars("BTC/USDT", 1)
    import_csv(store, f)
    assert store.read_bars("BTC/USDT", 1).equals(before)
    assert [r["n_bars"] for r in store.ranges("BTC/USDT", 1)] == [10]


@pytest.mark.parametrize("make, match", [
    (lambda p: p.write_text("a,b\n1,2\n"), "timestamp"),
    (lambda p: p.write_text("timestamp,open,high,low,close\n2025-01-01,1,1,1,1\n"), "missing columns"),
    (lambda p: _csv(p, close=-1.0), "close"),
    (lambda p: p.write_text("timestamp,open,high,low,close,volume\n"), "empty"),
])
def test_bad_csv_gives_a_clear_error_and_writes_nothing(store, tmp_path, make, match):
    f = tmp_path / "bad.csv"
    make(f)
    with pytest.raises(ValueError, match=match):
        import_csv(store, f)
    assert store.ranges("BTC/USDT", 1) == [] and store.provenance("BTC/USDT", 1) == {}


def test_missing_file_and_wrong_bar_size_are_refused(store, tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        import_csv(store, tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="bar size"):
        import_csv(store, _csv(tmp_path / "d.csv"), tf=5)
    assert store.ranges("BTC/USDT", 5) == []


def test_default_src():
    assert default_src("x/y/foo.csv") == "repo:foo.csv"
