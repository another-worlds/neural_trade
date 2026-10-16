"""Desk database (CONTRACTS C4): sqlite in WAL mode, versioned migrations, one repository per table.

Standard library only. Schema version 1 is the C4 table list, columns verbatim; the version is kept in
``PRAGMA user_version`` and a database written by a newer schema is refused. Every write runs in its own
``BEGIN IMMEDIATE`` transaction so concurrent processes queue on the busy timeout instead of failing on a lock
upgrade. The journal is append-only: triggers make UPDATE and DELETE raise.

Gaps in C4 (minimum implemented, nothing invented): C4 types only some columns. Columns it types keep that type
(TEXT, REAL, INTEGER; "TEXT(JSON)" is TEXT holding JSON). Untyped C4 columns are declared TEXT when they are ids,
names, enums or paths and INTEGER when they are timestamps (``ts``, ``*_ms``) or counters (``pid``, ``gpu``); the
literal list is in tests/desk/test_db.py. ``Fill`` has no
``account_id`` field, so ``FillRepo.save`` takes it as an argument; ``AccountState`` carries no ``cost_profile``
or ``created_ms``, so accounts and the tables with no model dataclass use plain dict rows.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from neural_trade.desk.core.model import DeskError, Fill, Order, OrderStatus, Position

__all__ = ["DeskDb", "open_db", "SchemaTooNew", "SCHEMA_VERSION", "columns"]

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000


class SchemaTooNew(DeskError):
    pass


# Table -> column definitions, copied from C4 (untyped C4 columns are TEXT).
_TABLES: dict[str, str] = {
    "accounts": "id TEXT PRIMARY KEY, kind TEXT, quote TEXT, cash REAL, cost_profile TEXT, created_ms INTEGER, armed_until_ms INTEGER",
    "orders": "client_id TEXT PRIMARY KEY, account_id TEXT, instrument_id TEXT, strategy_id TEXT, bot_id TEXT, side TEXT, "
              "type TEXT, qty REAL, size_frac REAL, price REAL, trigger_price REAL, tif TEXT, reduce_only INTEGER, "
              "bracket TEXT, status TEXT, filled_qty REAL, avg_fill_price REAL, reason TEXT, venue_id TEXT, "
              "ts_created INTEGER, ts_updated INTEGER, tags TEXT",
    "fills": "trade_id TEXT PRIMARY KEY, client_id TEXT, instrument_id TEXT, account_id TEXT, side TEXT, qty REAL, "
             "price REAL, mid REAL, fee REAL, fee_currency TEXT, liquidity TEXT, ts INTEGER",
    "positions": "position_id TEXT PRIMARY KEY, account_id TEXT, instrument_id TEXT, strategy_id TEXT, side TEXT, qty REAL, "
                 "avg_price REAL, realised_pnl REAL, fees REAL, ts_opened INTEGER, ts_closed INTEGER",
    "journal": "id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, account_id TEXT, kind TEXT, payload TEXT",
    "bots": "id TEXT PRIMARY KEY, name TEXT, strategy TEXT, params TEXT, instrument_id TEXT, tf INTEGER, account_id TEXT, "
            "model_id TEXT, state TEXT, risk TEXT, created_ms INTEGER, last_bar_ms INTEGER, error TEXT",
    "alerts": "id TEXT PRIMARY KEY, name TEXT, condition TEXT, frequency TEXT, state TEXT, created_ms INTEGER, "
              "last_fired_ms INTEGER, message TEXT",
    "alert_log": "id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT, ts INTEGER, payload TEXT",
    "layouts": "id TEXT PRIMARY KEY, name TEXT, payload TEXT, updated_ms INTEGER",
    "drawings": "id TEXT PRIMARY KEY, layout_id TEXT, instrument_id TEXT, tool TEXT, points TEXT, style TEXT, "
                "locked INTEGER, updated_ms INTEGER",
    "models": "id TEXT PRIMARY KEY, run_dir TEXT, bundle_dir TEXT, symbol TEXT, tf INTEGER, lookback INTEGER, "
              "horizons TEXT, fingerprint TEXT, dev_scores TEXT, promoted_ms INTEGER, active_for TEXT",
    "jobs": "id TEXT PRIMARY KEY, kind TEXT, params TEXT, state TEXT, pid INTEGER, gpu INTEGER, log_path TEXT, "
            "created_ms INTEGER, started_ms INTEGER, finished_ms INTEGER, result TEXT, error TEXT",
    "backtests": "id TEXT PRIMARY KEY, name TEXT, strategy TEXT, params TEXT, instrument_id TEXT, tf INTEGER, start_ms INTEGER, "
                 "end_ms INTEGER, cost_profile TEXT, job_id TEXT, report_path TEXT, created_ms INTEGER",
    "settings": "key TEXT PRIMARY KEY, value TEXT",
}
# JSON columns (C4 "TEXT(JSON)"): stored as json text, returned as python values.
_JSON: dict[str, tuple[str, ...]] = {
    "accounts": ("cost_profile",), "orders": ("bracket", "tags"), "journal": ("payload",),
    "bots": ("params", "risk"), "alerts": ("condition",), "alert_log": ("payload",), "layouts": ("payload",),
    "drawings": ("points", "style"), "models": ("horizons", "dev_scores", "active_for"),
    "jobs": ("params", "result"), "backtests": ("params", "cost_profile"), "settings": ("value",),
}
_BOOL = {"orders": ("reduce_only",), "drawings": ("locked",)}

_MIGRATIONS: dict[int, list[str]] = {
    1: [f"CREATE TABLE {t} ({cols})" for t, cols in _TABLES.items()] + [
        "CREATE INDEX orders_account_status ON orders(account_id, status)",
        "CREATE INDEX fills_account ON fills(account_id, ts)",
        "CREATE INDEX journal_account ON journal(account_id, id)",
        "CREATE TRIGGER journal_no_update BEFORE UPDATE ON journal "
        "BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END",
        "CREATE TRIGGER journal_no_delete BEFORE DELETE ON journal "
        "BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END",
        # INSERT OR REPLACE would overwrite a row without firing the DELETE trigger (recursive_triggers is off).
        "CREATE TRIGGER journal_no_replace BEFORE INSERT ON journal "
        "WHEN NEW.id IS NOT NULL AND EXISTS (SELECT 1 FROM journal WHERE id = NEW.id) "
        "BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END",
    ],
}


def columns(table: str) -> list[str]:
    return [c.split()[0] for c in _TABLES[table].split(", ")]


def _enc(table: str, row: dict) -> dict:
    out = dict(row)
    for c in _JSON.get(table, ()):
        if out.get(c) is not None:
            out[c] = json.dumps(out[c], sort_keys=True)
    for c in _BOOL.get(table, ()):
        if out.get(c) is not None:
            out[c] = int(bool(out[c]))
    return out


def _dec(table: str, row: sqlite3.Row) -> dict:
    out = dict(row)
    for c in _JSON.get(table, ()):
        if out.get(c) is not None:
            out[c] = json.loads(out[c])
    for c in _BOOL.get(table, ()):
        if out.get(c) is not None:
            out[c] = bool(out[c])
    return out


class _Tx:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type: Any, *rest: Any) -> None:
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")


class DeskDb:
    """One connection to ``desk.sqlite``. Not shared across threads; open one per thread or process."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=busy_timeout_ms / 1000, isolation_level=None)
        try:
            self.conn.row_factory = sqlite3.Row
            self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            # The WAL switch and the first migration of a fresh file can race other processes; SQLite does not
            # always honour the busy timeout there, so retry on "locked" for a bounded time.
            self._retry(lambda: self.conn.execute("PRAGMA journal_mode=WAL").fetchone(), busy_timeout_ms)
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self._retry(self._migrate, busy_timeout_ms)
        except BaseException:
            self.conn.close()
            raise
        self.orders = OrderRepo(self)
        self.fills = FillRepo(self)
        self.positions = PositionRepo(self)
        self.journal = JournalRepo(self)
        self.alert_log = AlertLogRepo(self)
        self.settings = SettingsRepo(self)
        for name in ("accounts", "bots", "alerts", "layouts", "drawings", "models", "jobs", "backtests"):
            setattr(self, name, RowRepo(self, name, "id"))

    @staticmethod
    def _retry(fn: Any, budget_ms: int) -> Any:
        deadline = time.monotonic() + max(budget_ms, 1000) / 1000
        while True:
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) and "busy" not in str(e) or time.monotonic() > deadline:
                    raise
                time.sleep(0.01)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DeskDb":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        if self.version > SCHEMA_VERSION:
            ver = self.version
            raise SchemaTooNew(f"{self.path} has schema version {ver}; this code knows up to {SCHEMA_VERSION}")
        while self.version < SCHEMA_VERSION:
            nxt = self.version + 1
            with self.tx():
                if self.version >= nxt:  # another process migrated while we waited for the write lock
                    continue
                for stmt in _MIGRATIONS[nxt]:
                    self.conn.execute(stmt)
                self.conn.execute(f"PRAGMA user_version={nxt}")

    def tx(self) -> _Tx:
        return _Tx(self.conn)

    def tables(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return sorted(r[0] for r in rows)


class RowRepo:
    """Dict rows for a table, JSON columns decoded. ``save`` is an upsert on the primary key."""

    def __init__(self, db: DeskDb, table: str, pk: str) -> None:
        self.db, self.table, self.pk = db, table, pk
        self.cols = columns(table)

    def _save(self, row: dict) -> None:
        unknown = set(row) - set(self.cols)
        if unknown:
            raise KeyError(f"{self.table}: unknown columns {sorted(unknown)}")
        enc = _enc(self.table, {c: row.get(c) for c in self.cols})
        sql = f"INSERT OR REPLACE INTO {self.table} ({', '.join(self.cols)}) VALUES ({', '.join('?' * len(self.cols))})"
        with self.db.tx() as c:
            c.execute(sql, [enc[k] for k in self.cols])

    def _get(self, key: Any) -> Optional[dict]:
        r = self.db.conn.execute(f"SELECT * FROM {self.table} WHERE {self.pk}=?", (key,)).fetchone()
        return None if r is None else _dec(self.table, r)

    def _list(self, **where: Any) -> list[dict]:
        bad = set(where) - set(self.cols)
        if bad:
            raise KeyError(f"{self.table}: unknown columns {sorted(bad)}")
        clause = " AND ".join(f"{k}=?" for k in where)
        sql = f"SELECT * FROM {self.table}" + (f" WHERE {clause}" if where else "") + f" ORDER BY {self.cols[0]}"
        return [_dec(self.table, r) for r in self.db.conn.execute(sql, list(where.values()))]

    def delete(self, key: Any) -> bool:
        with self.db.tx() as c:
            return c.execute(f"DELETE FROM {self.table} WHERE {self.pk}=?", (key,)).rowcount > 0

    # dict-row API for tables without a model dataclass
    def save(self, row: dict) -> None:
        self._save(row)

    def get(self, key: Any) -> Optional[dict]:
        return self._get(key)

    def list(self, **where: Any) -> list[dict]:
        return self._list(**where)


class OrderRepo(RowRepo):
    def __init__(self, db: DeskDb) -> None:
        super().__init__(db, "orders", "client_id")

    def save(self, order: Order) -> None:  # type: ignore[override]
        self._save(order.to_dict())

    def get(self, client_id: str) -> Optional[Order]:  # type: ignore[override]
        row = self._get(client_id)
        return None if row is None else Order.from_dict(row)

    def list(self, *, account_id: Optional[str] = None, status: Optional[OrderStatus] = None,  # type: ignore[override]
             open_only: bool = False) -> list[Order]:
        where: dict = {}
        if account_id is not None:
            where["account_id"] = account_id
        if status is not None:
            where["status"] = OrderStatus(status).value
        orders = [Order.from_dict(r) for r in self._list(**where)]
        return [o for o in orders if not o.is_closed] if open_only else orders


class FillRepo(RowRepo):
    def __init__(self, db: DeskDb) -> None:
        super().__init__(db, "fills", "trade_id")

    def save(self, fill: Fill, account_id: str) -> None:  # type: ignore[override]
        self._save({**fill.to_dict(), "account_id": account_id})

    def get(self, trade_id: str) -> Optional[tuple[Fill, str]]:  # type: ignore[override]
        """Returns ``(fill, account_id)``."""
        row = self._get(trade_id)
        if row is None:
            return None
        acc = row.pop("account_id")
        return Fill.from_dict(row), acc

    def list(self, *, account_id: Optional[str] = None) -> list[Fill]:  # type: ignore[override]
        rows = self._list(**({"account_id": account_id} if account_id is not None else {}))
        return [Fill.from_dict({k: v for k, v in r.items() if k != "account_id"}) for r in rows]


class PositionRepo(RowRepo):
    def __init__(self, db: DeskDb) -> None:
        super().__init__(db, "positions", "position_id")

    def save(self, position: Position) -> None:  # type: ignore[override]
        self._save(position.to_dict())

    def get(self, position_id: str) -> Optional[Position]:  # type: ignore[override]
        row = self._get(position_id)
        return None if row is None else Position.from_dict(row)

    def list(self, *, account_id: Optional[str] = None) -> list[Position]:  # type: ignore[override]
        rows = self._list(**({"account_id": account_id} if account_id is not None else {}))
        return [Position.from_dict(r) for r in rows]


def _now_ms() -> int:
    return int(time.time() * 1000)


class JournalRepo:
    """Append-only event log; there is no update or delete API and the triggers refuse them in SQL."""

    def __init__(self, db: DeskDb) -> None:
        self.db = db

    def append(self, account_id: Optional[str], kind: str, payload: Any, *, ts: Optional[int] = None) -> int:
        ts = _now_ms() if ts is None else ts
        with self.db.tx() as c:
            cur = c.execute("INSERT INTO journal (ts, account_id, kind, payload) VALUES (?, ?, ?, ?)",
                            (ts, account_id, kind, json.dumps(payload, sort_keys=True)))
            return int(cur.lastrowid)

    def read(self, *, account_id: Optional[str] = None, kind: Optional[str] = None,
             after_id: int = 0, limit: Optional[int] = None) -> list[dict]:
        sql, args = "SELECT * FROM journal WHERE id>?", [after_id]
        for col, val in (("account_id", account_id), ("kind", kind)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        sql += " ORDER BY id" + (f" LIMIT {int(limit)}" if limit is not None else "")
        return [_dec("journal", r) for r in self.db.conn.execute(sql, args)]


class AlertLogRepo:
    def __init__(self, db: DeskDb) -> None:
        self.db = db

    def append(self, alert_id: str, payload: Any, *, ts: Optional[int] = None) -> int:
        ts = _now_ms() if ts is None else ts
        with self.db.tx() as c:
            cur = c.execute("INSERT INTO alert_log (alert_id, ts, payload) VALUES (?, ?, ?)",
                            (alert_id, ts, json.dumps(payload, sort_keys=True)))
            return int(cur.lastrowid)

    def list(self, alert_id: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM alert_log", []
        if alert_id is not None:
            sql, args = sql + " WHERE alert_id=?", [alert_id]
        return [_dec("alert_log", r) for r in self.db.conn.execute(sql + " ORDER BY id", args)]


class SettingsRepo:
    def __init__(self, db: DeskDb) -> None:
        self.db = db

    def set(self, key: str, value: Any) -> None:
        with self.db.tx() as c:
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))

    def get(self, key: str, default: Any = None) -> Any:
        r = self.db.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r[0])

    def all(self) -> dict[str, Any]:
        return {r[0]: json.loads(r[1]) for r in self.db.conn.execute("SELECT key, value FROM settings ORDER BY key")}


def open_db(path: str | Path, **kw: Any) -> DeskDb:
    return DeskDb(path, **kw)
