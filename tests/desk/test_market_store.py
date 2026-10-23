"""Market store (DESK-0.7): idempotent writes, resample equal to the neural_trade preprocessors, CSV export that the
training loaders accept, gaps equal to ``data.gaps.gap_flags``, concurrent writers."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.data.gaps import gap_flags
from neural_trade.data.loaders import validate_ohlcv_frame
from neural_trade.data.preprocessors import resample_bars, standardize_ohlcv
from neural_trade.data.preprocessors_registry import run_preprocessors
from neural_trade.desk.data import store as storemod
from neural_trade.desk.data.store import MarketStore, open_market
from neural_trade.experiments.dataset import dataset_fingerprint

SRC = Path(storemod.__file__).resolve().parents[4]  # .../src of this checkout
SYM = "BTC/USDT"


@pytest.fixture
def ms(tmp_path):
    with MarketStore(tmp_path / "market.sqlite") as s:
        yield s


def _ts_ms(df: pd.DataFrame) -> np.ndarray:
    return (pd.DatetimeIndex(df["timestamp"]).tz_convert("UTC").asi8 // 1_000_000).astype(np.int64)


def test_timestamps_are_utc_integer_milliseconds(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars, "t")
    first = int(pd.Timestamp("2025-10-11 02:30:00", tz="UTC").timestamp() * 1000)
    got = ms.read_bars(SYM, 1)
    assert got["ts"].dtype == np.int64 and int(got["ts"].iloc[0]) == first
    assert ms.conn.execute("SELECT typeof(ts), typeof(tf) FROM bars LIMIT 1").fetchone() == ("integer", "integer")
    assert ms.last_ts(SYM, 1) == first + (len(synthetic_bars) - 1) * 60_000
    assert ms.last_ts("ETH/USDT", 1) is None


def test_schema_is_the_a6_tables(ms):
    cols = lambda t: [r[1] for r in ms.conn.execute(f"PRAGMA table_info({t})")]  # noqa: E731
    assert cols("bars") == ["symbol", "tf", "ts", "o", "h", "l", "c", "v", "src"]
    assert cols("sources") == ["symbol", "tf", "src", "first_ts", "last_ts", "n_bars", "updated"]
    assert ms.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_five_minute_resample_equals_resample_bars(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars, "t")
    cfg = Config()
    cfg.RESAMPLE_MINUTES = 5
    want = resample_bars(standardize_ohlcv(synthetic_bars.copy(), cfg), cfg)
    got = ms.read_bars(SYM, 5)
    assert len(got) == len(want) == 600
    np.testing.assert_array_equal(got["ts"].to_numpy(), _ts_ms(want))
    for mine, theirs in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("v", "Volume")):
        np.testing.assert_array_equal(got[mine].to_numpy(), want[theirs].to_numpy())
    assert ms.resample(SYM, 5).equals(got)


def test_resample_range_limit_and_cache_invalidation(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars.iloc[:1000], "t")
    a = ms.read_bars(SYM, 15)
    assert ms.read_bars(SYM, 15) is not a and ms.read_bars(SYM, 15).equals(a)
    t0 = int(a["ts"].iloc[3])
    sub = ms.read_bars(SYM, 15, start=t0, end=t0 + 3 * 15 * 60_000)
    assert sub["ts"].tolist() == [t0, t0 + 900_000, t0 + 1_800_000]
    assert ms.read_bars(SYM, 15, limit=2)["ts"].tolist() == a["ts"].tail(2).tolist()
    ms.write_bars(SYM, 1, synthetic_bars.iloc[1000:1500], "t")   # same connection writes: cache rebuilt
    assert len(ms.read_bars(SYM, 15)) > len(a)
    with MarketStore(ms.path) as other:                           # another connection writes: the stamp changes
        before = len(ms.read_bars(SYM, 15))
        other.write_bars(SYM, 1, synthetic_bars.iloc[1500:2000], "t")
    assert len(ms.read_bars(SYM, 15)) > before


def test_duplicate_writes_are_idempotent(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars, "a")
    snap = ms.conn.execute("SELECT * FROM bars ORDER BY ts").fetchall()
    ms.write_bars(SYM, 1, synthetic_bars, "a")
    ms.write_bars(SYM, 1, synthetic_bars.iloc[100:200], "a")
    assert ms.conn.execute("SELECT * FROM bars ORDER BY ts").fetchall() == snap
    (row,) = ms.ranges(SYM, 1)
    assert row["n_bars"] == len(synthetic_bars) and row["src"] == "a"


def test_rewrite_replaces_values_and_duplicates_in_one_frame_keep_the_last(ms, synthetic_bars):
    part = synthetic_bars.iloc[:10].copy()
    ms.write_bars(SYM, 1, part, "a")
    changed = part.copy()
    changed["close"] = changed["close"] + 1.0
    messy = pd.concat([part, changed])   # file order: originals first, then the changed rows (last wins)
    ms.write_bars(SYM, 1, messy, "b")
    got = ms.read_bars(SYM, 1)
    assert len(got) == 10
    np.testing.assert_array_equal(got["c"].to_numpy(), changed["close"].to_numpy())
    assert [r["src"] for r in ms.ranges(SYM, 1)] == ["b"] and ms.ranges(SYM, 1)[0]["n_bars"] == 10


def test_unsorted_input_and_short_column_names(ms, synthetic_bars):
    df = synthetic_bars.iloc[:20]
    short = pd.DataFrame({"ts": _ts_ms(standardize_ohlcv(df.copy(), None)), "o": df["open"], "h": df["high"],
                          "l": df["low"], "c": df["close"], "v": df["volume"]}).iloc[::-1]
    ms.write_bars(SYM, 1, short, "s")
    got = ms.read_bars(SYM, 1)
    assert got["ts"].is_monotonic_increasing and len(got) == 20
    np.testing.assert_array_equal(got["c"].to_numpy(), df["close"].to_numpy())


def test_ranges_per_source(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars.iloc[:100], "x")
    ms.write_bars(SYM, 1, synthetic_bars.iloc[200:300], "y")
    r = ms.ranges(SYM, 1)
    assert [(d["src"], d["n_bars"]) for d in r] == [("x", 100), ("y", 100)]
    assert r[0]["last_ts"] < r[1]["first_ts"]
    assert ms.ranges(SYM, 5) == []


def test_read_bars_range_and_limit(ms, synthetic_bars):
    ms.write_bars(SYM, 1, synthetic_bars, "t")
    t = ms.read_bars(SYM, 1)["ts"]
    lo, hi = int(t.iloc[10]), int(t.iloc[20])
    got = ms.read_bars(SYM, 1, start=lo, end=hi)
    assert got["ts"].tolist() == t.iloc[10:20].tolist()                        # start inclusive, end exclusive
    assert ms.read_bars(SYM, 1, limit=3)["ts"].tolist() == t.tail(3).tolist()  # newest, ascending
    assert ms.read_bars(SYM, 1, start="2025-10-11 02:40", end="2025-10-11 02:45")["ts"].tolist() == t.iloc[10:15].tolist()
    assert ms.read_bars("none", 1).empty


def test_export_csv_passes_the_training_loaders_and_fingerprint_matches(ms, synthetic_bars, tmp_path):
    ms.write_bars(SYM, 1, synthetic_bars, "t")
    path = tmp_path / "datasets" / "x.csv"
    facts = ms.export_csv(SYM, 1, None, None, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "timestamp,open,high,low,close,volume"
    cfg = Config()
    df = validate_ohlcv_frame(run_preprocessors(pd.read_csv(path), cfg), bar_minutes=1)
    assert len(df) == len(synthetic_bars)
    # pandas' fast CSV parser (the loaders' too) is not bit-exact on repr floats: compare to 1e-12 relative
    np.testing.assert_allclose(df["Close"].to_numpy(), synthetic_bars["close"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(df["Volume"].to_numpy(), synthetic_bars["volume"].to_numpy(), rtol=1e-12)
    fp = dataset_fingerprint(Config(CSV_PATH=str(path)))
    assert "error" not in fp, fp
    for key in ("sha256", "size_bytes", "n_rows", "n_bars", "first_timestamp", "last_timestamp", "loader"):
        assert facts[key] == fp[key], key
    assert (facts["rows"], facts["first"], facts["last"]) == (fp["n_bars"], fp["first_timestamp"], fp["last_timestamp"])


def test_export_csv_range_and_empty(ms, synthetic_bars, tmp_path):
    ms.write_bars(SYM, 1, synthetic_bars, "t")
    facts = ms.export_csv(SYM, 5, "2025-10-11 03:00", "2025-10-11 04:00", tmp_path / "five.csv")
    assert facts["rows"] == 12 and facts["first"] == "2025-10-11T03:00:00+00:00"
    with pytest.raises(ValueError, match="nothing to export"):
        ms.export_csv(SYM, 1, "2030-01-01", None, tmp_path / "none.csv")


def test_gaps_match_gap_flags(ms, synthetic_bars):
    holes = synthetic_bars.drop(index=list(range(100, 103)) + [500] + list(range(900, 920))).reset_index(drop=True)
    ms.write_bars(SYM, 1, holes, "t")
    got = ms.gaps(SYM, 1)
    frame = standardize_ohlcv(holes.copy(), None)
    flags = gap_flags(frame, 1.0)
    ts = _ts_ms(frame)
    assert [g["end_ts"] for g in got] == ts[flags].tolist()
    assert [g["start_ts"] for g in got] == ts[np.flatnonzero(flags) - 1].tolist()
    assert [g["missing"] for g in got] == [3, 1, 20]
    # tolerance and range follow gap_flags too; a 5-minute resample sees only the holes wider than 1.5 bins
    assert len(ms.gaps(SYM, 1, tolerance=2.5)) == 2
    five = ms.read_bars(SYM, 5)
    f5 = gap_flags(pd.DataFrame({"timestamp": pd.to_datetime(five["ts"], unit="ms", utc=True)}), 5.0)
    assert [g["end_ts"] for g in ms.gaps(SYM, 5)] == five["ts"].to_numpy()[f5].tolist()
    assert ms.gaps(SYM, 1, start=ts[0], end=ts[50]) == []
    ms.write_bars(SYM, 1, synthetic_bars.iloc[100:103], "fill")   # a hole can be filled later
    assert [g["missing"] for g in ms.gaps(SYM, 1)] == [1, 20]


def test_empty_and_invalid_writes(ms, synthetic_bars):
    assert ms.write_bars(SYM, 1, synthetic_bars.iloc[:0], "t") == 0
    with pytest.raises(ValueError):
        ms.write_bars(SYM, 1, synthetic_bars.drop(columns=["datetime"]), "t")
    with pytest.raises(ValueError):
        ms.write_bars(SYM, 0, synthetic_bars, "t")
    assert ms.read_bars(SYM, 1).empty


def test_failed_open_closes_the_connection(tmp_path, monkeypatch):
    opened = []
    real = storemod.sqlite3.connect

    def spy(*a, **k):
        c = real(*a, **k)
        opened.append(c)
        return c

    monkeypatch.setattr(storemod.sqlite3, "connect", spy)
    monkeypatch.setattr(MarketStore, "_migrate", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        MarketStore(tmp_path / "m.sqlite")
    with pytest.raises(Exception, match="closed"):
        opened[0].execute("SELECT 1")


def test_newer_schema_is_refused(tmp_path):
    p = tmp_path / "m.sqlite"
    MarketStore(p).close()
    import sqlite3

    c = sqlite3.connect(p)
    c.execute("PRAGMA user_version=99")
    c.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        MarketStore(p)


def test_open_market_uses_the_home(desk_home):
    with open_market(desk_home) as s:
        assert s.path == desk_home / "market.sqlite"


def test_importing_the_store_loads_no_heavy_module():
    code = ("import sys, neural_trade.desk.data.store\n"
            "print('LOADED:' + ','.join(m for m in ('tensorflow', 'fastapi', 'nicegui', 'ccxt') if m in sys.modules))\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip().splitlines()[-1] == "LOADED:", out


_WORKER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    import numpy as np, pandas as pd
    from neural_trade.desk.data.store import MarketStore
    path, go, idx, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    while not go.exists():           # all workers open the fresh file in the same instant
        time.sleep(0.001)
    s = MarketStore(path)
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for k in range(5):
        lo = (idx * 3 + k * 40) % 400   # overlapping windows: workers write the same minutes
        ts = base + pd.to_timedelta(np.arange(lo, lo + 100), unit="min")
        df = pd.DataFrame({"datetime": ts, "open": 1.0 + idx, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0})
        s.write_bars("BTC/USDT", 1, df, f"w{idx}")
    s.close()
""")


@pytest.mark.parametrize("round_", range(3))
def test_concurrent_processes_race_on_a_fresh_file(tmp_path, round_):
    path, go, n = tmp_path / f"race{round_}" / "market.sqlite", tmp_path / f"go{round_}", 4
    script = tmp_path / "worker.py"
    script.write_text(_WORKER, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "CUDA_VISIBLE_DEVICES": "-1"}
    procs = [subprocess.Popen([sys.executable, str(script), str(path), str(go), str(i), str(n)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(n)]
    time.sleep(2.0)       # let every worker import and reach the barrier
    go.write_text("go")
    outs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [0] * n, outs
    with MarketStore(path) as s:
        assert s.version == 1
        bars = s.read_bars("BTC/USDT", 1)
        assert bars["ts"].is_unique and bars["ts"].is_monotonic_increasing
        assert len(bars) == s.conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] > 0
        assert sum(r["n_bars"] for r in s.ranges("BTC/USDT", 1)) == len(bars)
        assert s.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


# ---- provenance (NT-221): text per source that writes never rebuild ---------------------------------------------
def test_provenance_is_kept_across_writes_and_replaced_by_set(tmp_path):
    with MarketStore(tmp_path / "m.sqlite") as s:
        assert s.provenance("X", 1) == {}
        s.set_provenance("X", 1, "a", "first")
        s.write_bars("X", 1, pd.DataFrame({"ts": [60_000, 120_000], "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 1.0}), "a")
        assert s.provenance("X", 1) == {"a": "first"}
        s.set_provenance("X", 1, "a", "second")
        assert s.provenance("X", 1) == {"a": "second"} and s.provenance("X", 5) == {}


def test_a_store_made_before_the_provenance_table_gains_it_on_open(tmp_path):
    import sqlite3
    p = tmp_path / "old.sqlite"
    with MarketStore(p):
        pass
    c = sqlite3.connect(p)
    c.execute("DROP TABLE provenance")
    c.commit()
    c.close()
    with MarketStore(p) as s:
        s.set_provenance("X", 1, "a", "t")
        assert s.provenance("X", 1) == {"a": "t"} and s.version == 1
