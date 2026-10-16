"""Fixtures for the trading desk tests (D-067). Desk state never touches the real ``var/desk``."""
from __future__ import annotations

from pathlib import Path

import pytest

from neural_trade.desk.config import ENV_CONFIG, ENV_HOME, ENV_HOST, ENV_PLUGIN_DIRS, ENV_PORT


@pytest.fixture(autouse=True)
def _clean_desk_env(monkeypatch):
    """A developer's own NT_DESK_* variables must not leak into a test."""
    for name in (ENV_HOME, ENV_HOST, ENV_PORT, ENV_PLUGIN_DIRS, ENV_CONFIG, "NT_DESK_LIVE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def desk_home(tmp_path, monkeypatch) -> Path:
    """A temporary desk home, also set as ``NT_DESK_HOME`` so ``DeskSettings.load()`` finds it."""
    home = tmp_path / "desk_home"
    home.mkdir()
    monkeypatch.setenv(ENV_HOME, str(home))
    return home


def pytest_collection_modifyitems(items):
    """Mark every test under tests/desk with ``desk`` (and e2e/network by directory) so ``-m desk`` selects them."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        if "desk" in parts and "tests" in parts:
            item.add_marker(pytest.mark.desk)
            if "e2e" in parts:
                item.add_marker(pytest.mark.e2e)
            if "network" in parts:
                item.add_marker(pytest.mark.network)
