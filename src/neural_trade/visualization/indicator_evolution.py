"""How the 18 learned indicator periods evolved over training."""
from __future__ import annotations

import pandas as pd

PREFIXES = ("ma_period_", "macd_", "rsi_period_", "bb_period_")


def _frame(data) -> pd.DataFrame:
    if isinstance(data, (str,)) or hasattr(data, "read_text"):
        path = str(data)
        if path.endswith(".jsonl"):
            from neural_trade.telemetry.epoch_logger import read_metrics

            rows = read_metrics(path)
            return pd.DataFrame([{k.split("/", 1)[1]: v for k, v in r.items() if k.startswith("period/")}
                                 | {"epoch": r.get("epoch")} for r in rows])
        return pd.read_csv(path)
    return pd.DataFrame(data)


def indicator_evolution(data, config=None, **_):
    """Plotly line chart of every learned period per epoch.

    ``data``: an indicator_params_history.csv / metrics.jsonl path, or a DataFrame with the
    period columns (ma_period_*, macd_*_{fast,slow,signal}, rsi_period_*, bb_period_*).
    """
    import plotly.graph_objects as go

    df = _frame(data)
    cols = [c for c in df.columns if c.startswith(PREFIXES) and not c.startswith("change_")]
    x = df["epoch"] if "epoch" in df else list(range(len(df)))
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=x, y=df[c], mode="lines+markers", name=c))
    fig.update_layout(title="Learned indicator periods", xaxis_title="epoch", yaxis_title="period (bars)",
                      height=480, legend=dict(font=dict(size=9)))
    return fig
