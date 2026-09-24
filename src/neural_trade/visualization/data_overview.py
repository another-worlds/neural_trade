"""The data and its purged split: where each block sits in time and what its labels look like."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.theme import apply

BLOCKS = ("train", "val", "cal", "test")
# Blocks are named on the figure, so they need no hue: a neutral tint, alternating strength so
# neighbours separate. (Hues would collide with the fixed roles: h0 blue, status green, long violet.)
BLOCK_OPACITY = {"train": 0.07, "val": 0.17, "cal": 0.07, "test": 0.17}
BLOCK_COLORS = {name: T.NEUTRAL for name in BLOCKS}
_LABEL_ROW_PX = 17          # the second row of block labels sits this far above the first


def _timestamps(df) -> pd.Series:
    return pd.to_datetime(df["timestamp"] if "timestamp" in df else df.iloc[:, 0])


def split_table(blocks, config) -> pd.DataFrame:
    """One row per block (``blocks`` from data.processor.split_arrays): size, time span, share of
    labels inside the deadband, up-rate outside it per horizon, realised 1-bar volatility."""
    from neural_trade.metrics.direction_labels import direction_labels_np

    df = blocks["df"]
    ts = _timestamps(df)
    rows = {}
    for name in BLOCKS:
        b = blocks[name]
        labels = direction_labels_np(b["y"], b["last_close"], float(config.DIR_DEADBAND_BPS))
        anchors = b["anchor_bar"]
        row = {"sequences": len(anchors), "from": ts.iloc[int(anchors[0])], "to": ts.iloc[int(anchors[-1])],
               "1-bar vol $": float(np.mean(np.std(np.diff(b["X"], axis=1), axis=1)))}
        for h, (lab, mask) in labels.items():
            row[f"in deadband {h}"] = float(1 - mask.mean())
            row[f"up-rate {h}"] = float(lab[mask].mean()) if mask.any() else float("nan")
        rows[name] = row
    return pd.DataFrame(rows).T


def split_overview_figure(blocks, config, *, title: Optional[str] = None, max_points: int = 6000):
    """Close price with the train / val / cal / test blocks shaded and named (the thin unshaded
    strips between them are the purge gaps)."""
    import plotly.graph_objects as go

    df = blocks["df"]
    ts = _timestamps(df)
    close = df["Close"].to_numpy(float) if "Close" in df else df["close"].to_numpy(float)
    idx = S.thin(len(close), max_points)
    fig = go.Figure(go.Scatter(x=ts.iloc[idx], y=close[idx].astype(np.float32), name="close", mode="lines",
                               line=dict(width=1.2, color=T.INK_2),
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>close $%{y:,.0f}<extra></extra>"))
    for k, name in enumerate(BLOCKS):
        a = blocks[name]["anchor_bar"]
        t0, t1 = ts.iloc[int(a[0])], ts.iloc[int(a[-1])]
        fig.add_vrect(x0=t0, x1=t1, fillcolor=T.NEUTRAL, opacity=BLOCK_OPACITY[name], line_width=0, layer="below")
        # the block's name and size above the plot area, from its left edge (inside, the price line would
        # cross it). val and cal are short blocks side by side: their labels alternate between two rows,
        # so they cannot touch at any notebook width
        fig.add_annotation(x=t0, xanchor="left", y=1.0, yref="paper", yanchor="bottom", showarrow=False, align="left",
                           yshift=_LABEL_ROW_PX * (k % 2), text=f"<b>{name}</b> {len(a):,}",
                           font=dict(color=T.INK_2, size=11))
    fold = blocks.get("fold")
    gap = getattr(fold, "gap", "?")
    sub = "<br>".join((
        f"fold {getattr(fold, 'fold', '?')}: train | val | cal | test in time order, {gap} sequences purged between blocks",
        "(no bar is both a training label and an evaluation input)",
        f"{len(df):,} one-minute bars, {ts.iloc[0]:%Y-%m-%d %H:%M} to {ts.iloc[-1]:%Y-%m-%d %H:%M} UTC",
        "shading spans each block's anchor bars; the number after each block is its count of sequences"))
    apply(fig, title=title or "Purged walk-forward split", subtitle=sub, height=500)
    fig.update_layout(showlegend=False, margin=dict(t=172))
    fig.update_yaxes(title_text="close ($)", tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(title_text="time (UTC)")
    return fig


# ------------------------------------------------------------------ registry entry (data, config)
def split_overview(data, config=None, **kw):
    """``data``: the dict returned by data.processor.split_arrays."""
    return split_overview_figure(data, config, **kw)
