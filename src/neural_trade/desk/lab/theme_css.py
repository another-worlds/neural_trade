"""DESK-5.4: the desk's CSS variables, from the project theme tokens (CONTRACTS C5 ``GET /system/theme``).

A pure function: no file is read or written. ``visualization.theme`` is imported inside the functions so that
importing the desk stays cheap. ``theme.py`` defines one dark palette only, so there is no light variant here
(``color-scheme: dark`` says so to the browser). The candle tokens are the desk's own (CONTRACTS lists none;
the values are the minimum, in ``DESK_TOKENS``): candles must not use the GOOD / CRITICAL status colours, which
keep their meaning for outcomes.
"""
from __future__ import annotations

from typing import Dict

#: the desk's own tokens: candle body colours (not green / red)
DESK_TOKENS: Dict[str, str] = {
    "candle-up": "#5ab4ac",
    "candle-down": "#c26a8a",
}


def theme_tokens() -> Dict[str, str]:
    """Token name -> value, in a fixed order: surfaces and ink, horizons, status, long / short, candles, orders."""
    from neural_trade.visualization import theme as T

    tokens: Dict[str, str] = {
        "paper": T.PAPER, "surface": T.SURFACE, "ink": T.INK, "ink-2": T.INK_2, "muted": T.MUTED,
        "grid": T.GRID, "axis": T.AXIS, "neutral": T.NEUTRAL,
    }
    for i, h in enumerate(T.HORIZONS):
        tokens[f"horizon-{h}"] = T.SERIES[i]
    tokens.update({"good": T.GOOD, "warning": T.WARNING, "serious": T.SERIOUS, "critical": T.CRITICAL,
                   "long": T.LONG_COLOR, "short": T.SHORT_COLOR, "up": T.UP_COLOR, "down": T.DOWN_COLOR})
    tokens.update(DESK_TOKENS)
    # order lines: buy / sell take the long / short pair, protective orders the status colours
    tokens.update({"order-buy": T.LONG_COLOR, "order-sell": T.SHORT_COLOR, "order-stop": T.CRITICAL,
                   "order-take": T.GOOD, "order-pending": T.MUTED})
    return tokens


def theme_css() -> str:
    """The ``:root`` block of CSS custom properties (``--nt-<token>``), deterministic, ending in a newline."""
    lines = [":root {", "  color-scheme: dark;"]
    lines += [f"  --nt-{name}: {value};" for name, value in theme_tokens().items()]
    lines += ["}", ""]
    return "\n".join(lines)
