"""Shared helpers of the tactical report generator: theme, data loading, example windows, HTML snippets."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAC = HERE.parent
ROOT = TAC.parent.parent
OUT = TAC / "webdash" / "out"
sys.path.insert(0, str(TAC / "common"))
import cutoff as C  # noqa: E402  (hard cutoff 2025-07-20 23:59)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("nt_theme", ROOT / "src" / "neural_trade" / "visualization" / "theme.py")
NF = _load("nt_novel_features", TAC / "novel" / "features.py")
T.register()

# colour roles (theme.py): horizons h0-h2 are reserved for horizons and never used here
UP, DOWN = T.UP_COLOR, T.DOWN_COLOR            # candles: violet up, yellow down (always with the candle shape)
S = T.OTHER_SERIES                             # gold, pink, tan, violet, red, light grey, grey
PINK, GOLD, TAN, VIOLET, SALMON = S[1], S[0], S[2], S[3], S[4]
INK, INK2, MUTED, GRID, SURFACE, PAPER, AXIS = T.INK, T.INK_2, T.MUTED, T.GRID, T.SURFACE, T.PAPER, T.AXIS


def jload(name):
    return json.load(open(OUT / f"{name}.json", encoding="utf-8"))


def jfile(rel):
    return json.load(open(TAC / rel, encoding="utf-8"))


# ------------------------------------------------------------------ data and the example windows
def load_bars():
    """OHLCV of the cut file from 2024-07-01 to the cutoff, as a DataFrame with a DatetimeIndex."""
    csv = C.ensure_cut_csv()
    d = pd.read_csv(csv, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    C.assert_cut(d["timestamp"], "report bars")
    d = d[d["timestamp"] >= "2024-07-01"].reset_index(drop=True)
    C.assert_cut(d["timestamp"], "report bars (look period)")
    return d


def pick_windows(d):
    """Three example 60-bar windows (calm, trending, volatile) that also have 60 later bars before the cutoff.
    Returns [(label, start_index, W[60,5])] - W = open, high, low, close, volume."""
    A = d[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
    c = A[:, 3]
    starts = np.arange(0, len(A) - 130, 30)
    lr = np.diff(np.log(c))
    sd = np.array([lr[s:s + 59].std() for s in starts])
    net = np.array([abs(np.log(c[s + 59] / c[s])) for s in starts])
    path = np.array([np.abs(lr[s:s + 59]).sum() for s in starts])
    er = net / np.maximum(path, 1e-12)
    vol = np.array([A[s:s + 60, 4].sum() for s in starts])
    live = np.array([(A[st:st + 60, 1] > A[st:st + 60, 2]).mean() for st in starts])
    act = (vol > np.median(vol) * 0.5) & (sd > 0) & (live >= 0.85)       # real activity: few flat O=H=L=C bars
    # calm: volatility at the 8% quantile; trending: highest efficiency among windows with median-or-higher volatility;
    # volatile: highest volatility
    q05 = np.quantile(sd[act], 0.08)
    i_calm = np.flatnonzero(act)[np.argmin(np.abs(sd[act] - q05))]
    ok = act & (sd >= np.median(sd[act]))
    i_trend = np.flatnonzero(ok)[np.argmax(er[ok])]
    i_vol = np.flatnonzero(act)[np.argmax(sd[act])]
    out = []
    for lab, i in (("Спокойное", i_calm), ("Трендовое", i_trend), ("Волатильное", i_vol)):
        s = int(starts[i])
        out.append({"label": lab, "i": s, "W": A[s:s + 60].copy(), "next": A[s + 60:s + 120].copy(),
                    "t0": d["timestamp"][s], "t1": d["timestamp"][s + 59], "sd_bps": float(sd[i] * 1e4),
                    "er": float(er[i]), "net_bps": float(net[i] * 1e4)})
        assert d["timestamp"][s + 119] <= pd.Timestamp(C.CUTOFF)
    return out


def span_text(w):
    return f"{w['t0']:%Y-%m-%d %H:%M}-{w['t1']:%H:%M} UTC"


# ------------------------------------------------------------------ html snippets
def fig_html(fig, key, height=None, title=None, subtitle=None, lift=0):
    """lift: pixels to move the top legend above a row of subplot titles (multi-panel figures)."""
    if title is not None or subtitle is not None:
        T.apply(fig, title=title, subtitle=subtitle, height=height)
    else:
        T.apply(fig, height=height)
    if lift:
        top = 56 + lift + 30
        h = height or fig.layout.height or 500
        fig.update_layout(margin=dict(t=top), legend=dict(orientation="h", yanchor="bottom", xanchor="left", x=0.0,
                                                          y=1.0 + (lift + 22) / max(h - top - 48, 100)))
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"f_{key}",
                       config={"displaylogo": False, "responsive": True, "scrollZoom": False})


GLYPH = {"good": ("&#10004;", T.GOOD), "bad": ("&#10008;", T.CRITICAL), "part": ("&#9680;", T.WARNING),
         "info": ("&#9679;", T.NEUTRAL)}


def badge(kind, text):
    g, col = GLYPH[kind]
    return f'<span class="badge" style="border-color:{col};color:{col}"><b>{g}</b> {text}</span>'


JV = {"effect": ("good", "есть эффект"), "partial": ("part", "частично"), "none": ("bad", "эффекта нет"),
      "failed": ("bad", "не подтвердилось"), "info": ("info", "справка")}


def jbadge(v):
    k, t = JV[v]
    return badge(k, t)


def table(rows, header, cls="rt"):
    h = "".join(f"<th>{x}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def card(title, body, note=None, wide=True):
    n = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="card"><h3>{title}</h3>{n}{body}</div>'


def ru(x, nd=3, sign=False):
    """Russian decimal comma."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")
