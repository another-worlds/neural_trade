"""DESK-0.8: ``neural-trade desk`` (init, import-csv) and the lazy registration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from neural_trade.cli import build_parser, main

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _csv(path, n=5):
    ts = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"), "open": 1.0, "high": 2.0, "low": 0.5,
                  "close": 1.5, "volume": 3.0}).to_csv(path, index=False)
    return path


def test_build_parser_parses_import_csv_and_init():
    ns = build_parser().parse_args(["desk", "import-csv", "bars.csv", "--home", "h", "--tf", "1"])
    assert ns.file == Path("bars.csv") and ns.tf == 1 and ns.src is None
    assert build_parser().parse_args(["desk", "init", "--home", "h"]).home == "h"


def test_init_creates_both_sqlite_files_and_the_token(tmp_path, capsys):
    home = tmp_path / "home"
    assert main(["desk", "init", "--home", str(home)]) == 0
    assert (home / "desk.sqlite").is_file() and (home / "market.sqlite").is_file()
    assert len((home / "token").read_text().strip()) >= 32
    assert json.loads(capsys.readouterr().out)["token"] == "created"


def test_init_twice_is_a_noop_and_keeps_the_token(tmp_path, capsys):
    home = tmp_path / "home"
    main(["desk", "init", "--home", str(home)])
    tok = (home / "token").read_text()
    capsys.readouterr()
    assert main(["desk", "init", "--home", str(home)]) == 0
    assert (home / "token").read_text() == tok
    assert json.loads(capsys.readouterr().out)["token"] == "kept"


def test_import_csv_command_writes_bars_and_provenance(tmp_path, capsys):
    home, f = tmp_path / "home", _csv(tmp_path / "x.csv")
    assert main(["desk", "import-csv", str(f), "--home", str(home), "--provenance", "unit test"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_bars"] == 5 and out["src"] == "repo:x.csv"
    from neural_trade.desk.data.store import open_market
    with open_market(home) as s:
        assert s.provenance("BTC/USDT", 1) == {"repo:x.csv": "unit test"}


def test_import_csv_bad_file_exits_nonzero_with_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["desk", "import-csv", str(bad), "--home", str(tmp_path / "home")]) == 2
    assert "timestamp" in capsys.readouterr().err


def test_build_parser_loads_no_heavy_dependency():
    code = ("import sys\nfrom neural_trade.cli import build_parser\nbuild_parser()\n"
            "bad = [m for m in ('tensorflow', 'fastapi', 'nicegui', 'ccxt') if m in sys.modules]\n"
            "print('LOADED:' + ','.join(bad))\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip().splitlines()[-1] == "LOADED:", out
