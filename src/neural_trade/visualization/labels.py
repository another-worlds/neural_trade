"""Figure labels from the dataset spec (NT-041, D-022): the instrument, the quote currency and the bar size.

No figure module writes 'BTC' or a '$' currency sign: a title, an axis or a hover names the instrument and
the quote currency of the setup that was run. The labels in force are the module's current :class:`Labels`:
the reference setup (BTC/USDT, one-minute bars) until a caller names another with :func:`use`::

    labels.use(config)                  # a Config or a DatasetSpec: symbol, quote currency, bar size
    with labels.using(config): ...      # for the duration of a block
    labels.quote()                      # 'USDT'
    labels.money(1234.5)                # '1,235 USDT'
    labels.axis_money(tickformat=',.0f')   # tickformat / ticksuffix keywords of a quote-currency axis

A currency amount is written ``1,234 USDT`` (number, space, quote code), also in a Plotly hover template and a
tick format, where the d3 ``$`` prefix is a US-dollar sign by definition and so cannot be used.
"""
from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict

from neural_trade.core.dataset_spec import DatasetSpec, bar_label


@dataclass(frozen=True)
class Labels:
    symbol: str
    quote_currency: str
    bar_minutes: float

    @property
    def base(self) -> str:
        """The base asset: ``BTC`` of ``BTC/USDT``."""
        return self.symbol.split("/")[0]

    @property
    def bar(self) -> str:
        return bar_label(self.bar_minutes)

    @property
    def tag(self) -> str:
        """``BTC/USDT 1-minute``."""
        return f"{self.symbol} {self.bar}"


def _reference() -> Labels:
    """The reference setup's labels: the Config defaults (the instrument is configuration, D-022)."""
    from neural_trade.core.config import Config

    specs = Config.field_specs()
    return Labels(str(specs["SYMBOL"].default), str(specs["QUOTE_CURRENCY"].default),
                  float(specs["RESAMPLE_MINUTES"].default))


_CURRENT = _reference()


def _labels_of(obj) -> Labels:
    if isinstance(obj, Labels):
        return obj
    spec = obj if isinstance(obj, DatasetSpec) else DatasetSpec.from_config(obj)
    return Labels(spec.symbol, spec.quote_currency, spec.bar_minutes)


def current() -> Labels:
    return _CURRENT


def use(obj) -> Labels:
    """Make a Config, a DatasetSpec or a Labels the labels in force; returns them."""
    global _CURRENT
    _CURRENT = _labels_of(obj)
    return _CURRENT


@contextmanager
def using(obj):
    """Labels of ``obj`` for the duration of the block."""
    global _CURRENT
    old = _CURRENT
    _CURRENT = _labels_of(obj)
    try:
        yield _CURRENT
    finally:
        _CURRENT = old


def labelled(fn):
    """Decorator for a figure or table function with a ``config`` parameter: the labels of that Config (symbol,
    quote currency, bar size) are in force while the function builds its output, then the previous ones return.
    A call without a Config (or with an object that is not one) keeps the labels in force."""
    names = list(inspect.signature(fn).parameters)
    pos = names.index("config") if "config" in names else None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cfg = kwargs.get("config", args[pos] if pos is not None and len(args) > pos else None)
        if cfg is None or not hasattr(cfg, "SYMBOL"):
            return fn(*args, **kwargs)
        with using(cfg):
            return fn(*args, **kwargs)

    return wrapper


def quote() -> str:
    """The quote currency code of the current setup (``USDT``)."""
    return _CURRENT.quote_currency


def symbol() -> str:
    return _CURRENT.symbol


def base() -> str:
    return _CURRENT.base


def tag() -> str:
    """Instrument and bar size: ``BTC/USDT 1-minute``."""
    return _CURRENT.tag


def money(value, decimals: int = 0, *, signed: bool = False) -> str:
    """``1,234 USDT`` / ``-1,234 USDT`` / ``+1,234 USDT`` (``signed``)."""
    v = float(value)
    sign = "-" if v < 0 else ("+" if signed else "")
    return f"{sign}{abs(v):,.{decimals}f} {quote()}"


def amount_suffix() -> str:
    """`` USDT``: the suffix after a number in a hover template or a tick (``%{y:,.0f}`` + this)."""
    return " " + quote()


def axis_money(**kw: Any) -> Dict[str, Any]:
    """Plotly axis keywords for a quote-currency axis: ``tickformat`` (default ``,.0f``) and the ``ticksuffix``."""
    kw.setdefault("tickformat", ",.0f")
    kw["ticksuffix"] = amount_suffix()
    return kw


__all__ = ["Labels", "amount_suffix", "axis_money", "base", "current", "labelled", "money", "quote", "symbol", "tag",
           "use", "using"]
