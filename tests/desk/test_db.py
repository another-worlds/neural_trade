"""Desk database (DESK-0.6): schema, round trips, concurrent writers, append-only journal, migrations."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from neural_trade.desk.core.model import (
    Fill, Liquidity, Order, OrderStatus, OrderType, Position, PositionSide, Side, TimeInForce,
)
from neural_trade.desk.store import db as dbmod
from neural_trade.desk.store.db import SchemaTooNew, open_db

SRC = Path(dbmod.__file__).resolve().parents[3]  # .../src of this checkout

# Copied by hand from CONTRACTS C4 (not derived from the code).
C4 = {
    "accounts": ["id", "kind", "quote", "cash", "cost_profile", "created_ms", "armed_until_ms"],
    "orders": ["client_id", "account_id", "instrument_id", "strategy_id", "bot_id", "side", "type", "qty", "size_frac",
               "price", "trigger_price", "tif", "reduce_only", "bracket", "status", "filled_qty", "avg_fill_price",
               "reason", "venue_id", "ts_created", "ts_updated", "tags"],
    "fills": ["trade_id", "client_id", "instrument_id", "account_id", "side", "qty", "price", "mid", "fee",
              "fee_currency", "liquidity", "ts"],
    "positions": ["position_id", "account_id", "instrument_id", "strategy_id", "side", "qty", "avg_price",
                  "realised_pnl", "fees", "ts_opened", "ts_closed"],
    "journal": ["id", "ts", "account_id", "kind", "payload"],
    "bots": ["id", "name", "strategy", "params", "instrument_id", "tf", "account_id", "model_id", "state", "risk",
             "created_ms", "last_bar_ms", "error"],
    "alerts": ["id", "name", "condition", "frequency", "state", "created_ms", "last_fired_ms", "message"],
    "alert_log": ["id", "alert_id", "ts", "payload"],
    "layouts": ["id", "name", "payload", "updated_ms"],
    "drawings": ["id", "layout_id", "instrument_id", "tool", "points", "style", "locked", "updated_ms"],
    "models": ["id", "run_dir", "bundle_dir", "symbol", "tf", "lookback", "horizons", "fingerprint", "dev_scores",
               "promoted_ms", "active_for"],
    "jobs": ["id", "kind", "params", "state", "pid", "gpu", "log_path", "created_ms", "started_ms", "finished_ms",
             "result", "error"],
    "backtests": ["id", "name", "strategy", "params", "instrument_id", "tf", "start_ms", "end_ms", "cost_profile",
                  "job_id", "report_path", "created_ms"],
    "settings": ["key", "value"],
}


@pytest.fixture
def db(tmp_path):
    d = open_db(tmp_path / "desk.sqlite")
    yield d
    d.close()


def _cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _order(cid="c1", **kw):
    base = dict(client_id=cid, instrument_id="BINANCE:ETHUSDT", account_id="acc1", strategy_id="s1", bot_id="b1",
                side=Side.SELL, type=OrderType.STOP_LIMIT, qty=1.5, price=101.25, trigger_price=100.5,
                tif=TimeInForce.IOC, reduce_only=True,
                bracket={"take_profit": 1.5, "stop_loss": 0.5, "tp_is_offset": True, "max_hold": 7},
                status=OrderStatus.PARTIALLY_FILLED, filled_qty=0.5, avg_fill_price=101.0, reason="r",
                venue_id="v9", ts_created=11, ts_updated=22, tags={"a": [1, 2], "b": "x"})
    base.update(kw)
    return Order(**base)


def test_schema_v1_has_exactly_the_c4_tables_and_columns(db):
    assert db.tables() == sorted(C4)
    for table, cols in C4.items():
        assert _cols(db.conn, table) == cols, table
    assert db.version == 1


def test_second_open_is_a_noop(tmp_path):
    p = tmp_path / "d.sqlite"
    a = open_db(p)
    a.orders.save(_order())
    a.close()
    schema1 = sqlite3.connect(p).execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    b = open_db(p)
    assert b.version == 1 and b.orders.get("c1") == _order()
    b.close()
    assert sqlite3.connect(p).execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema1


def test_wal_and_busy_timeout(db):
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_order_round_trip_all_fields(db):
    o = _order()
    db.orders.save(o)
    assert db.orders.get("c1") == o
    assert db.orders.get("nope") is None
    sized = _order("c2", qty=None, size_frac=0.25, type=OrderType.MARKET, reduce_only=False, bracket=None, price=None)
    db.orders.save(sized)
    assert db.orders.get("c2") == sized
    o.status, o.filled_qty = OrderStatus.FILLED, 1.5  # upsert
    db.orders.save(o)
    assert db.orders.get("c1").status is OrderStatus.FILLED
    assert [x.client_id for x in db.orders.list(account_id="acc1", open_only=True)] == ["c2"]
    assert [x.client_id for x in db.orders.list(status=OrderStatus.FILLED)] == ["c1"]


def test_fill_round_trip_all_fields(db):
    f = Fill(trade_id="t1", client_id="c1", instrument_id="BINANCE:ETHUSDT", side=Side.SELL, qty=0.7, price=99.5,
             mid=100.0, fee=0.0123, fee_currency="BNB", liquidity=Liquidity.MAKER, ts=1234)
    db.fills.save(f, "acc7")
    assert db.fills.get("t1") == (f, "acc7")
    assert db.fills.list(account_id="acc7") == [f] and db.fills.list(account_id="other") == []


def test_position_round_trip_all_fields(db):
    p = Position(position_id="pos-1", instrument_id="BINANCE:ETHUSDT", account_id="acc1", strategy_id="s1",
                 side=PositionSide.SHORT, qty=2.5, avg_price=3000.5, realised_pnl=-12.5, fees=1.25,
                 ts_opened=100, ts_closed=200)
    db.positions.save(p)
    assert db.positions.get("pos-1") == p
    assert db.positions.list(account_id="acc1") == [p]


# Rows for the tables without a model dataclass: every column set to a non-default value.
ROWS = {
    "accounts": {"id": "a1", "kind": "PAPER", "quote": "USDT", "cash": 1234.5, "cost_profile": {"fee": 0.001},
                 "created_ms": 5, "armed_until_ms": 9},
    "bots": {"id": "b1", "name": "bot", "strategy": "ma", "params": {"n": 3}, "instrument_id": "I", "tf": 5,
             "account_id": "a1", "model_id": "m1", "state": "RUNNING", "risk": {"max": 1.5}, "created_ms": 1,
             "last_bar_ms": 2, "error": "boom"},
    "alerts": {"id": "al1", "name": "n", "condition": {"op": ">", "x": 1}, "frequency": "once", "state": "ACTIVE",
               "created_ms": 1, "last_fired_ms": 2, "message": "hi"},
    "layouts": {"id": "l1", "name": "L", "payload": {"panes": [1, 2]}, "updated_ms": 3},
    "drawings": {"id": "d1", "layout_id": "l1", "instrument_id": "I", "tool": "trend", "points": [[1, 2], [3, 4]],
                 "style": {"color": "red"}, "locked": True, "updated_ms": 8},
    "models": {"id": "m1", "run_dir": "runs/x", "bundle_dir": "runs/x/b", "symbol": "BTC/USDT", "tf": 1,
               "lookback": 60, "horizons": [10, 15, 20], "fingerprint": "abc", "dev_scores": {"s": 0.5},
               "promoted_ms": 7, "active_for": ["BTC/USDT@1"]},
    "jobs": {"id": "j1", "kind": "train", "params": {"e": 1}, "state": "DONE", "pid": 42, "gpu": 1,
             "log_path": "l.log", "created_ms": 1, "started_ms": 2, "finished_ms": 3, "result": {"ok": True},
             "error": "e"},
    "backtests": {"id": "bt1", "name": "n", "strategy": "s", "params": {"p": 1}, "instrument_id": "I", "tf": 15,
                  "start_ms": 1, "end_ms": 2, "cost_profile": {"fee": 0.1}, "job_id": "j1", "report_path": "r.json",
                  "created_ms": 4},
}


@pytest.mark.parametrize("table", sorted(ROWS))
def test_dict_repo_round_trip_all_fields(db, table):
    row = ROWS[table]
    assert set(row) == set(C4[table])  # every column is set
    repo = getattr(db, table)
    repo.save(row)
    assert repo.get(row["id"]) == row
    assert repo.list() == [row]
    with pytest.raises(KeyError):
        repo.save({**row, "bogus": 1})
    assert repo.delete(row["id"]) and repo.get(row["id"]) is None


def test_settings_and_alert_log(db):
    db.settings.set("theme", {"mode": "dark", "n": [1, 2]})
    assert db.settings.get("theme") == {"mode": "dark", "n": [1, 2]} and db.settings.get("x", 5) == 5
    assert db.settings.all() == {"theme": {"mode": "dark", "n": [1, 2]}}
    i = db.alert_log.append("al1", {"price": 1.5}, ts=77)
    assert db.alert_log.list("al1") == [{"id": i, "alert_id": "al1", "ts": 77, "payload": {"price": 1.5}}]
    assert db.alert_log.list("zzz") == []


def test_journal_append_and_read(db):
    a = db.journal.append("acc1", "order", {"k": 1}, ts=10)
    b = db.journal.append("acc2", "fill", {"k": 2}, ts=11)
    assert b > a
    assert db.journal.read() == [
        {"id": a, "ts": 10, "account_id": "acc1", "kind": "order", "payload": {"k": 1}},
        {"id": b, "ts": 11, "account_id": "acc2", "kind": "fill", "payload": {"k": 2}},
    ]
    assert [r["id"] for r in db.journal.read(account_id="acc2")] == [b]
    assert [r["id"] for r in db.journal.read(after_id=a, limit=5)] == [b]


def test_journal_is_append_only(db):
    i = db.journal.append("acc1", "order", {"k": 1}, ts=10)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.conn.execute("UPDATE journal SET kind='x' WHERE id=?", (i,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.conn.execute("DELETE FROM journal WHERE id=?", (i,))
    # a second raw connection is refused too (the triggers live in the file, not in the API)
    raw = sqlite3.connect(db.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("DELETE FROM journal")
    raw.close()
    assert not hasattr(db.journal, "update") and not hasattr(db.journal, "delete")
    assert len(db.journal.read()) == 1 and db.journal.read()[0]["kind"] == "order"


def test_failed_transaction_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.tx() as c:
            c.execute("INSERT INTO settings (key, value) VALUES ('k', '1')")
            raise RuntimeError("x")
    assert db.settings.get("k") is None


def test_newer_schema_is_refused(tmp_path):
    p = tmp_path / "d.sqlite"
    open_db(p).close()
    raw = sqlite3.connect(p)
    raw.execute("PRAGMA user_version=2")
    raw.close()
    with pytest.raises(SchemaTooNew, match="version 2"):
        open_db(p)


def test_migrations_are_versioned_and_applied_in_order(tmp_path, monkeypatch):
    p = tmp_path / "d.sqlite"
    open_db(p).close()
    monkeypatch.setattr(dbmod, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(dbmod._MIGRATIONS, 2, ["ALTER TABLE settings ADD COLUMN note TEXT"])
    d = open_db(p)
    assert d.version == 2 and "note" in _cols(d.conn, "settings")
    d.close()
    open_db(p).close()  # re-open at 2 is a no-op (the ALTER would fail twice)


WRITER = textwrap.dedent("""
    import sys
    from neural_trade.desk.core.model import Order, OrderType, Side
    from neural_trade.desk.store.db import open_db
    db = open_db(sys.argv[1])
    for i in range(int(sys.argv[3])):
        db.orders.save(Order(client_id=f"{sys.argv[2]}-{i}", instrument_id="I", account_id="a", side=Side.BUY,
                             type=OrderType.MARKET, qty=1.0))
        db.journal.append("a", "order", {"i": i})
    db.close()
""")


def test_two_processes_write_orders_concurrently(tmp_path):
    p = tmp_path / "d.sqlite"
    open_db(p).close()  # migrated first; the writers race on writes (and a no-op migration check)
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "-1"}
    n = 150
    procs = [subprocess.Popen([sys.executable, str(script), str(p), tag, str(n)], env=env,
                              stderr=subprocess.PIPE, text=True) for tag in ("p1", "p2")]
    mine = open_db(p)
    for i in range(n):  # the test process itself is a third writer
        mine.orders.save(_order(f"me-{i}"))
    errs = [pr.communicate(timeout=120)[1] for pr in procs]
    assert [pr.returncode for pr in procs] == [0, 0], errs
    ids = {o.client_id for o in mine.orders.list()}
    assert ids == {f"{t}-{i}" for t in ("p1", "p2", "me") for i in range(n)}
    assert len(mine.journal.read()) == 2 * n
    mine.close()
