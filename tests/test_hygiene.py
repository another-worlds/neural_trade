"""Code hygiene (plan B16): no print() in the package, logging reaches the current stdout, and
``import neural_trade`` stays cheap (no TensorFlow)."""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "neural_trade"


def test_no_print_calls_in_the_package():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "cli.py":  # a command's result goes to stdout by design
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"use logging instead of print: {offenders}"


def test_no_bare_or_silent_broad_excepts():
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ExceptHandler):
                continue
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception",
                                                                                               "BaseException"))
            silent = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
            if node.type is None or (broad and silent):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"bare `except:` or `except Exception: pass`: {offenders}"


def test_package_logger_writes_to_the_current_stdout(capsys):
    import neural_trade  # noqa: F401  (configures the package logger)

    logging.getLogger("neural_trade.some.module").info("progress line")
    assert "progress line" in capsys.readouterr().out


def test_configure_logging_can_hand_over_to_the_root_logger(caplog):
    from neural_trade.core.logging import configure_logging

    try:
        configure_logging(handler=False)
        with caplog.at_level(logging.INFO, logger="neural_trade"):
            logging.getLogger("neural_trade.x").info("to root")
        assert "to root" in caplog.text
    finally:
        configure_logging()


def test_import_is_cheap_and_does_not_load_tensorflow():
    code = "import sys, neural_trade; print('tensorflow' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False"
