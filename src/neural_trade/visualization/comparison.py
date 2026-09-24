"""Figures for comparing runs (experiments.compare.compare_runs) and ablation verdicts."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

VERDICT_COLORS = {"VALUE": "#15803d", "HARMFUL": "#b91c1c", "NEUTRAL": "#6b7280", "INCONCLUSIVE": "#b45309"}


def runs_comparison_figure(df, metrics: Optional[Sequence[str]] = None, *, title: Optional[str] = None):
    """One small panel per metric, one bar per run (``df`` from compare_runs)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = [m for m in (metrics or [c for c in df.columns if c not in ("seed", "tags")]) if m in df.columns]
    cols = min(3, max(1, len(metrics)))
    rows = int(np.ceil(len(metrics) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics)
    labels = [str(i)[-28:] for i in df.index]
    for k, m in enumerate(metrics):
        fig.add_trace(go.Bar(x=labels, y=df[m].astype(float), name=m, showlegend=False,
                             hovertext=list(df.index)), row=k // cols + 1, col=k % cols + 1)
    fig.update_layout(title=title or f"{len(df)} runs", height=300 * rows)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def ablation_deltas_figure(analysis, *, title: Optional[str] = None):
    """Mean paired delta (+/- sd over seeds and periods) per term, mode and primary metric, coloured
    by that comparison's verdict. Positive = the term helps."""
    import plotly.graph_objects as go

    rows = []
    for term, t in analysis.get("terms", {}).items():
        for mode, m in t["modes"].items():
            for c in m["metrics"]:
                rows.append((f"{term.replace('LAMBDA_', '')} · {mode.replace('leave_one_', 'one ')} · "
                             f"{c['metric'].split('/')[-1]}", c["mean_delta"], c.get("sd_delta"), c["verdict"]))
    for c in (analysis.get("family") or {}).get("metrics", []):
        rows.append((f"family · {c['metric'].split('/')[-1]}", c["mean_delta"], c.get("sd_delta"), c["verdict"]))
    fig = go.Figure()
    for verdict, color in VERDICT_COLORS.items():
        sel = [r for r in rows if r[3] == verdict]
        if sel:
            fig.add_trace(go.Bar(y=[r[0] for r in sel], x=[r[1] for r in sel], orientation="h", name=verdict,
                                 marker_color=color,
                                 error_x=dict(type="data", array=[r[2] if r[2] == r[2] else 0 for r in sel])))
    fig.add_vline(x=0, line_color="#9ca3af")
    fig.update_layout(title=title or "Ablation: paired deltas (positive = the term helps)", barmode="overlay",
                      height=max(360, 22 * len(rows)), xaxis_title="mean delta over (seed, period) pairs")
    return fig


# ------------------------------------------------------------------ registry entries (data, config)
def runs_comparison(data, config=None, **kw):
    return runs_comparison_figure(data, **kw)


def ablation_deltas(data, config=None, **kw):
    return ablation_deltas_figure(data, **kw)
