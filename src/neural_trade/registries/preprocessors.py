"""Preprocessors registry (registry 8 of 9): DataFrame -> DataFrame steps.

Components are ``step(df, config) -> DataFrame``; ``run_preprocessors`` applies
``Config.PREPROCESSORS`` in order. Scalers are intentionally absent (see
neural_trade.data.preprocessors): scaling before the split leaks test statistics.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Iterable, Optional, Tuple

import pandas as pd

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.registry import BaseRegistry


class Preprocessors(BaseRegistry):
    registry = {}
    strict = True
    default = "standardize_ohlcv"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.data.preprocessors",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and params[:2] == ["df", "config"]


def run_preprocessors(df: pd.DataFrame, config, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Apply ``names`` (default Config.PREPROCESSORS) in order; each must return a DataFrame."""
    for name in list(names if names is not None else getattr(config, "PREPROCESSORS", []) or []):
        out = Preprocessors.build(name, df, config)
        if not isinstance(out, pd.DataFrame):
            raise ComponentValidationError(f"Preprocessors: '{name}' returned {type(out).__name__}")
        df = out
    return df


from neural_trade.data import preprocessors as _pp  # noqa: E402

for _name, _tags in (("standardize_ohlcv", ["cleaning", "default"]), ("sort_dedupe", ["cleaning", "default"]),
                     ("resample_bars", ["resampling", "default"]), ("drop_missing_close", ["cleaning", "default"]),
                     ("strip_currency_symbols", ["cleaning"]), ("log_returns", ["transform", "finance"]),
                     ("add_time_features", ["features", "calendar"])):
    Preprocessors.register(name=_name, tags=_tags)(getattr(_pp, _name))
Preprocessors._initialized = True
