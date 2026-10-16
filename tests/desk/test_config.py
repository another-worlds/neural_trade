"""DESK-0.2: ``DeskSettings.load()`` precedence is argument > environment > YAML > defaults."""
from __future__ import annotations

import os

import pytest

from neural_trade.desk.config import DESK_HOME, PROJECT_ROOT, DeskSettings


def _yaml(tmp_path, text):
    p = tmp_path / "desk.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_without_yaml_or_env(tmp_path):
    s = DeskSettings.load(yaml_path=tmp_path / "missing.yaml", env={})
    assert s.home == DESK_HOME
    assert (s.host, s.port) == ("127.0.0.1", 8080)
    assert s.plugin_dirs == [PROJECT_ROOT / "plugins_desk"]
    assert s.cost_profiles["research_zero"]["fee_bps_taker"] == 0.0
    assert s.accounts[0]["kind"] == "paper"


def test_yaml_overrides_defaults_and_resolves_relative_paths(tmp_path):
    p = _yaml(tmp_path, "home: somewhere/desk\nport: 9001\nplugin_dirs: [my_plugins]\n")
    s = DeskSettings.load(yaml_path=p, env={})
    assert s.home == PROJECT_ROOT / "somewhere" / "desk"
    assert s.port == 9001 and s.host == "127.0.0.1"
    assert s.plugin_dirs == [PROJECT_ROOT / "my_plugins"]


def test_env_overrides_yaml_and_argument_overrides_env(tmp_path):
    p = _yaml(tmp_path, f"home: {(tmp_path / 'from_yaml').as_posix()}\nport: 9001\n")
    env = {"NT_DESK_HOME": str(tmp_path / "from_env"), "NT_DESK_PORT": "9002", "NT_DESK_HOST": "0.0.0.0",
           "NT_DESK_PLUGIN_DIRS": os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])}
    s = DeskSettings.load(yaml_path=p, env=env)
    assert s.home == tmp_path / "from_env" and s.port == 9002 and s.host == "0.0.0.0"
    assert s.plugin_dirs == [tmp_path / "a", tmp_path / "b"]
    s2 = DeskSettings.load(home=tmp_path / "from_flag", yaml_path=p, env=env)
    assert s2.home == tmp_path / "from_flag" and s2.port == 9002


def test_config_path_from_env_and_process_environment(tmp_path, monkeypatch):
    p = _yaml(tmp_path, "port: 9100\n")
    monkeypatch.setenv("NT_DESK_CONFIG", str(p))
    monkeypatch.setenv("NT_DESK_HOME", str(tmp_path / "h"))
    s = DeskSettings.load()  # reads os.environ
    assert s.port == 9100 and s.home == tmp_path / "h"


def test_unknown_key_and_bad_top_level_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown desk setting"):
        DeskSettings.load(yaml_path=_yaml(tmp_path, "prot: 1\n"), env={})
    with pytest.raises(ValueError, match="mapping"):
        DeskSettings.load(yaml_path=_yaml(tmp_path, "- 1\n"), env={})


def test_shipped_configs_desk_yaml_loads_and_loading_creates_nothing(tmp_path, desk_home):
    s = DeskSettings.load()
    assert s.home == desk_home
    assert list(desk_home.iterdir()) == []
