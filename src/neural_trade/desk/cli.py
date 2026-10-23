"""``neural-trade desk ...`` (PLAN A2). Reached through one lazy hook in ``neural_trade/cli.py``; importing this module
loads no TensorFlow, fastapi, nicegui or ccxt (heavy imports stay inside the command handlers).

    neural-trade desk init [--home DIR]
    neural-trade desk import-csv FILE [--home DIR] [--symbol S] [--tf MIN] [--src LABEL] [--provenance TEXT]
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

__all__ = ["add_subparser"]


def _settings(args):
    from neural_trade.desk.config import DeskSettings

    return DeskSettings.load(home=args.home)


def cmd_init(args) -> int:
    from neural_trade.desk.data.store import open_market
    from neural_trade.desk.store.db import DeskDb
    from neural_trade.utils.atomic import atomic_write_text

    home = _settings(args).home
    home.mkdir(parents=True, exist_ok=True)
    DeskDb(home / "desk.sqlite").close()
    open_market(home).close()
    token = home / "token"
    created = not token.exists()                        # never overwritten: sessions hold the token
    if created:
        atomic_write_text(token, secrets.token_urlsafe(32) + "\n")
    print(json.dumps({"home": str(home), "desk": "desk.sqlite", "market": "market.sqlite",  # noqa: T201 - the result
                      "token": "created" if created else "kept"}))
    return 0


def cmd_import_csv(args) -> int:
    from neural_trade.desk.data.importers import import_csv
    from neural_trade.desk.data.store import open_market

    home = _settings(args).home
    try:
        with open_market(home) as store:
            res = import_csv(store, args.file, symbol=args.symbol, tf=args.tf, src=args.src,
                             provenance=args.provenance)
    except ValueError as exc:
        print(f"import-csv: {exc}", file=sys.stderr)  # noqa: T201 - the command's error
        return 2
    print(json.dumps(res))  # noqa: T201 - the command's result
    return 0


def add_subparser(sub: "argparse._SubParsersAction") -> None:
    d = sub.add_parser("desk", help="the trading desk: init its home, import bars (D-067)")
    ds = d.add_subparsers(dest="desk_command", required=True)
    i = ds.add_parser("init", help="create desk.sqlite, market.sqlite and the token in the desk home")
    i.add_argument("--home", default=None, help="desk home (default: NT_DESK_HOME or var/desk)")
    i.set_defaults(func=cmd_init)
    c = ds.add_parser("import-csv", help="import a timestamp,open,high,low,close,volume CSV into market.sqlite")
    c.add_argument("file", type=Path)
    c.add_argument("--home", default=None, help="desk home (default: NT_DESK_HOME or var/desk)")
    c.add_argument("--symbol", default="BTC/USDT")
    c.add_argument("--tf", type=int, default=1, help="bar size in minutes")
    c.add_argument("--src", default=None, help="source label (default: repo:<file name>)")
    c.add_argument("--provenance", default=None, help="provenance text (default: the file name and span)")
    c.set_defaults(func=cmd_import_csv)
