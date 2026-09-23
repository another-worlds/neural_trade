"""Strategy parameters from YAML/dicts, with unknown knobs rejected (a typo must not silently
fall back to a default)."""
from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.strategy.backtest import BacktestConfig
from neural_trade.strategy.strategies import Strategies, Strategy


def _check_keys(cls, params: Mapping[str, Any], what: str) -> None:
    known = {f.name for f in dataclasses.fields(cls)}
    for key in params:
        if key not in known:
            hint = difflib.get_close_matches(key, sorted(known), n=1)
            raise InvalidConfigurationError(
                f"unknown {what} parameter {key!r}" + (f" (did you mean {hint[0]!r}?)" if hint else ""))


def build_strategy(name: str, params: Optional[Mapping[str, Any]] = None) -> Strategy:
    cls = Strategies.get(name)
    params = dict(params or {})
    _check_keys(cls, params, f"strategy {name!r}")
    return cls(**params)


def build_backtest_config(params: Optional[Mapping[str, Any]] = None) -> BacktestConfig:
    params = dict(params or {})
    _check_keys(BacktestConfig, params, "backtest")
    return BacktestConfig(**params)


def load_params(path: Union[str, Path]) -> Dict[str, Any]:
    """``{strategy: name, params: {...}, backtest: {...}}`` from a YAML file."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(data) - {"strategy", "params", "backtest"}
    if unknown:
        raise InvalidConfigurationError(f"unknown top-level keys in {path}: {sorted(unknown)}")
    return data


def from_file(path: Union[str, Path]):
    """``(strategy, backtest_config)`` built from a params YAML."""
    data = load_params(path)
    return (build_strategy(data.get("strategy", Strategies.default), data.get("params")),
            build_backtest_config(data.get("backtest")))
