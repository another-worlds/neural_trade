"""The data and its purged split: where each block sits in time and what its labels look like."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

BLOCK_COLORS = {"train": "#1d4ed8", "val": "#b45309", "cal": "#7c3aed", "test": "#15803d"}


def split_table(blocks, config) -> pd.DataFrame:
    """One row per block (``blocks`` from data.processor.split_arrays): size, time span, share of
    labels inside the deadband, up-rate outside it per horizon, realised 1-bar volatility."""
    from neural_trade.metrics.direction_labels import direction_labels_np

    df = blocks["df"]
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df else df.iloc[:, 0])
    rows = {}
    for name in ("train", "val", "cal", "test"):
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
    """Close price with the train / val / cal / test blocks shaded (gaps between them are the purge)."""
    import plotly.graph_objects as go

    df = blocks["df"]
    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df else df.iloc[:, 0])
    close = df["Close"].to_numpy(float)
    step = max(1, len(close) // max_points)
    fig = go.Figure(go.Scatter(x=ts.iloc[::step], y=close[::step], name="close", line=dict(width=1, color="#374151")))
    for name, color in BLOCK_COLORS.items():
        a = blocks[name]["anchor_bar"]
        fig.add_vrect(x0=ts.iloc[int(a[0])], x1=ts.iloc[int(a[-1])], fillcolor=color, opacity=0.15, line_width=0,
                      annotation_text=name, annotation_position="top left")
    fold = blocks.get("fold")
    fig.update_layout(title=title or f"Purged split, fold {getattr(fold, 'fold', '?')} "
                                     f"(gap {getattr(fold, 'gap', '?')} sequences between blocks)",
                      height=380, yaxis_title="close", showlegend=False)
    return fig


# ------------------------------------------------------------------ registry entry (data, config)
def split_overview(data, config=None, **kw):
    """``data``: the dict returned by data.processor.split_arrays."""
    return split_overview_figure(data, config, **kw)
