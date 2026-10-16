"""``DeskSettings``: the desk's configuration, a plain dataclass loaded from defaults, ``configs/desk.yaml`` and
the environment (docs/desk/PLAN.md A2). Precedence, highest first: explicit arguments (``--home``), environment
variables, the YAML file, the defaults. Secrets (API keys) are never settings: they are read from the environment
by the broker that needs them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# The default home is relative to the project root so a desk started from any directory finds its state.
DESK_HOME = PROJECT_ROOT / "var" / "desk"
DEFAULT_YAML = PROJECT_ROOT / "configs" / "desk.yaml"

ENV_HOME = "NT_DESK_HOME"
ENV_HOST = "NT_DESK_HOST"
ENV_PORT = "NT_DESK_PORT"
ENV_PLUGIN_DIRS = "NT_DESK_PLUGIN_DIRS"   # os.pathsep separated
ENV_CONFIG = "NT_DESK_CONFIG"             # a YAML file to use instead of configs/desk.yaml


def _default_cost_profiles() -> Dict[str, Dict[str, float]]:
    # research_zero: the research yardstick (D-044). binance_spot_taker: a stated assumption for desk accounts
    # (0.1% taker fee per side, no spread or slippage); change it in configs/desk.yaml, not here.
    return {
        "research_zero": {"fee_bps_taker": 0.0, "fee_bps_maker": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0},
        "binance_spot_taker": {"fee_bps_taker": 10.0, "fee_bps_maker": 10.0, "half_spread_bps": 0.0,
                               "slippage_bps": 0.0},
    }


def _default_accounts() -> List[Dict[str, Any]]:
    return [{"id": "paper", "kind": "paper", "quote": "USDT", "cash": 10_000.0, "cost_profile": "research_zero"}]  # owner question 3 (desk costs) is open; D-044 default is 0


@dataclass
class DeskSettings:
    home: Path = DESK_HOME
    host: str = "127.0.0.1"
    port: int = 8080
    plugin_dirs: List[Path] = field(default_factory=lambda: [PROJECT_ROOT / "plugins_desk"])
    accounts: List[Dict[str, Any]] = field(default_factory=_default_accounts)
    cost_profiles: Dict[str, Dict[str, float]] = field(default_factory=_default_cost_profiles)

    @classmethod
    def load(cls, *, home: Optional[os.PathLike] = None, yaml_path: Optional[os.PathLike] = None,
             env: Optional[Mapping[str, str]] = None) -> "DeskSettings":
        """Defaults, then the YAML file (if it exists), then the environment, then ``home`` (the ``--home`` flag)."""
        env = os.environ if env is None else env
        values: Dict[str, Any] = {}
        path = Path(yaml_path) if yaml_path is not None else Path(env.get(ENV_CONFIG) or DEFAULT_YAML)
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: expected a mapping at the top level, got {type(loaded).__name__}")
            known = {f.name for f in fields(cls)}
            unknown = sorted(set(loaded) - known)
            if unknown:
                raise ValueError(f"{path}: unknown desk setting(s) {unknown}; known: {sorted(known)}")
            values.update(loaded)
        if env.get(ENV_HOME):
            values["home"] = env[ENV_HOME]
        if env.get(ENV_HOST):
            values["host"] = env[ENV_HOST]
        if env.get(ENV_PORT):
            values["port"] = env[ENV_PORT]
        if env.get(ENV_PLUGIN_DIRS):
            values["plugin_dirs"] = [p for p in env[ENV_PLUGIN_DIRS].split(os.pathsep) if p]
        if home is not None:
            values["home"] = home
        return cls._from_values(values)

    @classmethod
    def _from_values(cls, values: Dict[str, Any]) -> "DeskSettings":
        s = cls(**values)
        s.home = _resolve(s.home)
        s.port = int(s.port)
        s.plugin_dirs = [_resolve(p) for p in s.plugin_dirs]
        return s

    def paths(self) -> Tuple[Path, ...]:
        """The directories the desk may create under its home (nothing is created by loading)."""
        return (self.home,)


def _resolve(p: os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
