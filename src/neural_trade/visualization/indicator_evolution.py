"""How the 18 learned indicator periods evolved over training.

* :func:`indicator_evolution` (registry entry) - one panel per indicator family (moving averages,
  MACD fast / slow / signal, RSI, Bollinger bands), each period's path over the epochs, plus the
  relative change from epoch 1 of every period and each period's correlation with the validation
  loss across epochs.
* :func:`indicator_summary` - the table behind it: start, end, min, max, change, coefficient of
  variation and correlation with val loss for every period.

``data``: a metrics.jsonl / indicator_params_history.csv path, a list of metrics rows, or a
DataFrame with the period columns (ma_period_*, macd_*_{fast,slow,signal}, rsi_period_*, bb_period_*).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from neural_trade.visualization import theme as T

PREFIXES = ("ma_period_", "macd_", "rsi_period_", "bb_period_")
INDEX_COLORS = (T.SERIES[6], T.SERIES[4], T.SERIES[3])      # indicator copy 0 / 1 / 2 (not horizon colours)
MACD_DASH = {"fast": "solid", "slow": "dash", "signal": "dot"}


def _frame(data) -> pd.DataFrame:
    if isinstance(data, str) or hasattr(data, "read_text"):
        path = str(data)
        if path.endswith(".jsonl"):
            from neural_trade.telemetry.epoch_logger import read_metrics

            data = read_metrics(path)
        else:
            return pd.read_csv(path)
    if isinstance(data, list):
        rows = []
        for r in data:
            row = {k.split("/", 1)[1]: v for k, v in r.items() if k.startswith("period/")}
            row.update({k: r.get(k) for k in ("epoch", "val_loss", "loss") if k in r})
            rows.append(row)
        return pd.DataFrame(rows)
    df = pd.DataFrame(data).copy()
    return df.rename(columns={c: c.split("/", 1)[1] for c in df.columns if c.startswith("period/")})


def _period_cols(df):
    return [c for c in df.columns if c.startswith(PREFIXES) and not c.startswith("change_")]


def _val_loss(df):
    for k in ("val_loss", "log_val_loss"):
        if k in df.columns:
            return pd.to_numeric(df[k], errors="coerce").to_numpy(float)
    return None


def indicator_summary(data) -> pd.DataFrame:
    """One row per learned period: first, last, min, max, change %, CV, corr with val loss."""
    df = _frame(data)
    vl = _val_loss(df)
    rows = {}
    for c in _period_cols(df):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        corr = np.nan
        if vl is not None and np.nanstd(v) > 0 and np.nanstd(vl) > 0:
            corr = float(pd.Series(v).corr(pd.Series(vl)))
        rows[c] = {"epoch 1": v[0], "last": v[-1], "min": np.nanmin(v), "max": np.nanmax(v),
                   "change %": 100 * (v[-1] - v[0]) / v[0] if v[0] else np.nan,
                   "CV": float(np.nanstd(v) / np.nanmean(v)) if np.nanmean(v) else np.nan,
                   "corr with val loss": corr}
    return pd.DataFrame(rows).T.sort_values("change %", key=np.abs, ascending=False)


def indicator_evolution(data, config=None, *, height: int = 820, **_):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _frame(data)
    cols = _period_cols(df)
    x = (pd.to_numeric(df["epoch"], errors="coerce") + 1).tolist() if "epoch" in df else list(range(1, len(df) + 1))
    titles = ("Moving-average periods", "MACD periods (fast solid, slow dashed, signal dotted)", "RSI periods",
              "Bollinger-band periods", "Change from epoch 1 (%)", "Correlation with val loss across epochs")
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, vertical_spacing=0.16, horizontal_spacing=0.07)
    panel = {"ma_period_": (1, 1), "macd_": (1, 2), "rsi_period_": (1, 3), "bb_period_": (2, 1)}
    first = T.legend_once()
    for c in cols:
        m = re.search(r"_(\d)(?:_(fast|slow|signal))?$", c)
        idx, role = (int(m.group(1)), m.group(2)) if m else (0, None)
        color = INDEX_COLORS[idx % 3]
        r, k = next(v for p, v in panel.items() if c.startswith(p))
        y = pd.to_numeric(df[c], errors="coerce")
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if len(x) <= 40 else "lines", name=f"copy {idx}",
                                 legendgroup=f"idx{idx}", showlegend=first(f"idx{idx}"),
                                 line=dict(color=color, width=2, dash=MACD_DASH.get(role, "solid")),
                                 marker=dict(size=4), hovertemplate=f"{c}: %{{y:.2f}} bars<extra></extra>"), r, k)
        base = y.iloc[0]
        fig.add_trace(go.Scatter(x=x, y=100 * (y - base) / base if base else y * np.nan, mode="lines",
                                 name=c, showlegend=False, legendgroup=f"idx{idx}",
                                 line=dict(color=T.rgba(color, 0.7), width=1, dash=MACD_DASH.get(role, "solid")),
                                 hovertemplate=f"{c}: %{{y:+.1f}}%<extra></extra>"), 2, 2)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=2, col=2)
    summary = indicator_summary(df) if cols else pd.DataFrame()
    if len(summary) and summary["corr with val loss"].notna().any():
        s = summary["corr with val loss"].dropna().sort_values()
        fig.add_trace(go.Bar(x=s.to_numpy(), y=list(s.index), orientation="h", showlegend=False,
                             marker=dict(color=[T.SERIES[0] if v < 0 else T.SERIES[1] for v in s.to_numpy()]),
                             hovertemplate="%{y}: r = %{x:+.2f}<extra></extra>"), 2, 3)
        fig.update_xaxes(range=[-1, 1], row=2, col=3, title_text="r across epochs (mostly a shared trend with training time)")
        fig.update_yaxes(tickfont=dict(size=9), row=2, col=3)
    for r, k in ((1, 1), (1, 2), (1, 3), (2, 1)):
        fig.update_yaxes(title_text="bars" if k == 1 else None, row=r, col=k)
    for k in (1, 2):
        fig.update_xaxes(title_text="epoch", row=2, col=k)
    T.note_on_empty(fig)
    T.apply(fig, title="Learned indicator periods", height=height,
            subtitle="each indicator exists in three copies (0 / 1 / 2) with its own learned period(s)")
    fig.update_layout(margin=dict(t=110), legend=dict(y=1.05))
    return fig
