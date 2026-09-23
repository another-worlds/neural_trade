"""DataLoaders registry (registry 5 of 9): where raw market data comes from.

Components are ``loader(config, **kwargs) -> pandas.DataFrame`` returning the RAW frame; the
Preprocessors pipeline standardises it and ``validate_ohlcv_frame`` checks the result.

Deferred (named in REGISTRY_SPECIFICATIONS.md, no implementation exists yet):
``binance_api``, ``postgres``, ``ccxt``.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Tuple

from neural_trade.core.registry import BaseRegistry


class DataLoaders(BaseRegistry):
    registry = {}
    strict = True
    default = "csv"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.data.loaders",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and bool(params) and params[0] == "config"


from neural_trade.data import loaders as _loaders  # noqa: E402

DataLoaders.register(name="csv", tags=["file", "default"])(_loaders.load_csv)
DataLoaders.register(name="parquet", tags=["file"], dependencies=["pyarrow"])(_loaders.load_parquet)
DataLoaders.register(name="dataframe", tags=["memory"])(_loaders.load_dataframe)
DataLoaders._initialized = True
