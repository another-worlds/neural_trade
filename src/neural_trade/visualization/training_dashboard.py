"""Training dashboard: every per-epoch metric the trainer logs, in one figure, plus a health card.

The trainer logs ~260 values per epoch (metrics.jsonl): total and component losses for train and
validation, direction metrics per horizon for both the direction head and the Gaussian readout of
the price head, calibration (ECE, PIT-KS), prediction bias, the six physics terms, gradient norm,
learning rates and the 18 learned indicator periods. This module draws them.

* :func:`training_dashboard_figure` - 6x2 panels, validation solid / training dotted, horizons in
  fixed colours, best epoch marked, early-stopping patience in the title.
* :func:`training_health` / :func:`training_health_html` - the verdicts a person scans first:
  convergence, stability, generalisation gap, train/val agreement, progress, patience, class
  collapse per horizon, non-finite gradients.
* :func:`batch_loss_figure` - the live within-epoch loss (running mean per batch).

``history``: a list of epoch dicts (metrics.jsonl rows or Keras epoch logs), a Keras History,
a {key: [values]} dict, a DataFrame, or a path to metrics.jsonl.
"""
from __future__ import annotations

import html
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neural_trade.visualization import theme as T

LOSS_COMPONENTS: Tuple[Tuple[str, str], ...] = (
    ("point_loss", "point"), ("trend_loss", "trend"), ("dir_loss", "direction"), ("nll_loss", "NLL"),
    ("crps_loss", "CRPS"), ("soft_ece_loss", "soft ECE"), ("vol_loss", "volatility"), ("reg_loss", "regulariser"),
)
PHYSICS_TERMS: Tuple[Tuple[str, str], ...] = (
    ("t_perp_loss", "T-perp"), ("casimir_loss", "Casimir"), ("hd_loss", "HD"), ("ife_loss", "IFE"),
    ("vac_loss", "vacuum"), ("vac_overflow_loss", "vac overflow"),
)

PANELS = (
    "Total loss", "Loss components (unweighted, log)",
    "Direction head MCC", "Price head MCC (Gaussian readout)",
    "Balanced accuracy", "Direction ECE",
    "Variance calibration: PIT-KS (0 = calibrated)", "Predicted vs true up-rate",
    "Physics terms (log)", "Learning rate (log)",
    "Gradient global norm (pre-clip)", "Seconds per epoch",
)


# ------------------------------------------------------------------ data
def history_frame(history) -> pd.DataFrame:
    """Normalise any history-like input to a DataFrame with one row per epoch."""
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        if isinstance(history, str) or hasattr(history, "read_text"):
            from neural_trade.telemetry.epoch_logger import read_metrics

            history = read_metrics(str(history))
        if hasattr(history, "history") and not isinstance(history, (list, dict)):
            history = history.history
        if isinstance(history, dict):
            df = pd.DataFrame(history)
        else:
            df = pd.DataFrame(list(history or []))
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(len(df))
    df = df.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    if "seconds" in df.columns and "epoch_seconds" not in df.columns:
        df["epoch_seconds"] = df["seconds"]
    return df


def _col(df, key) -> Optional[np.ndarray]:
    if key not in df.columns:
        return None
    v = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
    return v if np.isfinite(v).any() else None


# ------------------------------------------------------------------ health
def training_health(history, config=None) -> Dict[str, dict]:
    """Named checks, each ``{"value": str, "status": good|warning|critical|info, "detail": str}``."""
    df = history_frame(history)
    out: Dict[str, dict] = {}
    n = len(df)
    planned = int(getattr(config, "EPOCHS", n) or n) if config is not None else n
    loss, val = _col(df, "loss"), _col(df, "val_loss")
    out["epochs"] = dict(value=f"{n} / {planned}", status="info", detail="epochs finished / planned")
    if n == 0 or val is None:
        return out
    best = int(np.nanargmin(val))
    patience = int(getattr(config, "EARLY", 0) or 0) if config is not None else 0
    since = n - 1 - best
    out["best"] = dict(value=f"{val[best]:.4f} @ epoch {best + 1}", status="info", detail="lowest validation loss")
    if patience:
        frac = since / patience
        out["patience"] = dict(value=f"{since} / {patience}",
                               status="good" if frac < 0.5 else ("warning" if frac < 1 else "critical"),
                               detail="epochs since the best val loss / early-stopping patience")
    if n >= 3:
        slope = float(np.polyfit(np.arange(3), val[-3:], 1)[0])
        rel = slope / abs(np.nanmean(val[-3:]))
        state = "improving" if rel < -0.002 else ("worsening" if rel > 0.002 else "plateau")
        out["convergence"] = dict(value=f"{100 * rel:+.2f}% / epoch ({state})",
                                  status={"improving": "good", "plateau": "warning", "worsening": "critical"}[state],
                                  detail="val-loss slope over the last 3 epochs")
        cv = float(np.nanstd(val[-5:]) / abs(np.nanmean(val[-5:])))
        out["stability"] = dict(value=f"CV {100 * cv:.2f}%",
                                status="good" if cv < 0.01 else ("warning" if cv < 0.03 else "critical"),
                                detail="spread of the last 5 val losses")
    if loss is not None:
        gap = val - loss
        trend = gap[-1] - gap[max(0, n - 5)]
        out["gap"] = dict(value=f"{gap[-1]:+.4f} ({'widening' if trend > 0.02 * abs(val[-1]) else 'steady'})",
                          status="warning" if trend > 0.02 * abs(val[-1]) else "good",
                          detail="val loss - train loss (train is a dropout-on running mean); a widening gap "
                                 "over the last 5 epochs is the overfitting signal")
        if n >= 3:
            k = min(n, 10)
            agree = float(np.mean(np.sign(np.diff(loss[-k:])) == np.sign(np.diff(val[-k:]))))
            out["agreement"] = dict(value=f"{100 * agree:.0f}%",
                                    status="good" if agree >= 0.7 else ("warning" if agree >= 0.5 else "critical"),
                                    detail="epochs (last 10) where train and val loss moved the same way")
    prog = (val[0] - val[-1]) / abs(val[0]) if val[0] else 0.0
    out["progress"] = dict(value=f"{100 * prog:+.1f}%", status="good" if prog > 0 else "critical",
                           detail="val-loss change since epoch 1")
    for h in T.HORIZONS:
        mcc = _col(df, f"val_dir_mcc_{h}")
        up = _col(df, f"val_pred_up_rate_{h}")
        if mcc is None:
            continue
        collapsed = up is not None and (up[-1] < 0.03 or up[-1] > 0.97)
        status = "critical" if collapsed else ("good" if mcc[-1] > 0.02 else ("warning" if mcc[-1] > -0.02
                                                                                else "critical"))
        extra = f", predicts up {100 * up[-1]:.0f}%" if up is not None else ""
        out[f"direction {h}"] = dict(value=f"MCC {mcc[-1]:+.3f} (best {np.nanmax(mcc):+.3f}){extra}",
                                     status=status,
                                     detail="validation MCC of the direction head; critical when it predicts one "
                                            "class almost always (collapse)")
    nf = _col(df, "nonfinite_grad_steps")
    if nf is not None:
        total = float(np.nansum(nf))
        out["gradients"] = dict(value=f"{int(total)} non-finite steps", status="good" if total == 0 else "critical",
                                detail="training steps whose gradient had a NaN/inf (skipped by the guard)")
    lr = _col(df, "lr")
    if lr is not None:
        cuts = int(np.sum(np.diff(lr[np.isfinite(lr)]) < 0))
        out["learning rate"] = dict(value=f"{lr[-1]:.2e} ({cuts} reductions)", status="info",
                                    detail="ReduceLROnPlateau cuts so far")
    sec = _col(df, "epoch_seconds")
    if sec is not None:
        left = max(0, planned - n)
        out["time"] = dict(value=f"{np.nanmedian(sec):.0f} s / epoch" + (f", ~{left * np.nanmedian(sec) / 60:.0f} min left"
                                                                         if left else ""),
                           status="info", detail="median epoch time")
    return out


_ICON = {"good": "&#10003;", "warning": "!", "critical": "&#10007;", "info": "i"}
_STATUS_COLOR = {"good": T.GOOD, "warning": T.WARNING, "critical": T.CRITICAL, "info": T.MUTED}


def training_health_html(history, config=None, *, title: str = "Training health") -> str:
    """The checks from :func:`training_health` as a grid of tiles (icon + label + colour)."""
    checks = training_health(history, config)
    tiles = []
    for name, c in checks.items():
        color = _STATUS_COLOR[c["status"]]
        tiles.append(
            f"<div title='{html.escape(c['detail'])}' style='background:{T.SURFACE};border:1px solid {T.AXIS};"
            f"border-left:3px solid {color};border-radius:6px;padding:8px 10px;min-width:0'>"
            f"<div style='color:{T.MUTED};font-size:11px;letter-spacing:.04em;text-transform:uppercase'>"
            f"{html.escape(name)}</div>"
            f"<div style='color:{T.INK};font-size:14px;margin-top:2px;font-variant-numeric:tabular-nums'>"
            f"<span style='color:{color};font-weight:700;margin-right:6px'>{_ICON[c['status']]}</span>"
            f"{html.escape(c['value'])}</div></div>")
    return (f"<div style='font-family:{T.FONT};background:{T.PAPER};padding:12px;border-radius:8px'>"
            f"<div style='color:{T.INK};font-weight:600;margin-bottom:8px'>{html.escape(title)}</div>"
            f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px'>"
            + "".join(tiles) + "</div></div>")


# ------------------------------------------------------------------ figure
def _line(fig, x, y, *, row, col, name, color, dash=T.VAL_DASH, group=None, show=True, width=2, hover=None,
          legend="legend"):
    import plotly.graph_objects as go

    if y is None:
        return
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if len(x) <= 40 else "lines", name=name,
                             legendgroup=group or name, showlegend=show, legend=legend,
                             line=dict(color=color, dash=dash, width=width), marker=dict(size=5, color=color),
                             hovertemplate=hover or f"{name}: %{{y:.4g}}<extra></extra>"),
                  row=row, col=col)


def _log_ticks(fig, row, col, values):
    """Label a log axis at the values actually present (e.g. the learning-rate steps)."""
    v = np.unique(np.round(values[np.isfinite(values) & (values > 0)], 12))
    if len(v):
        fig.update_yaxes(tickvals=v, ticktext=[f"{a:.1e}" for a in v], row=row, col=col)


def training_dashboard_figure(history, config=None, *, title: Optional[str] = None, height: int = 1720):
    """All logged per-epoch metrics in 12 panels (see :data:`PANELS`)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = history_frame(history)
    x = (df["epoch"].to_numpy() + 1).tolist()   # epochs shown 1-based, as the trainer prints them
    fig = make_subplots(rows=6, cols=2, shared_xaxes=True, subplot_titles=PANELS,
                        vertical_spacing=0.05, horizontal_spacing=0.07)
    first = T.legend_once()

    # key for the whole figure: the line styles, then the horizons
    for label, dash in (("validation", T.VAL_DASH), ("training", T.TRAIN_DASH)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label, line=dict(color=T.INK_2, dash=dash),
                                 legendgroup=f"style-{label}", hoverinfo="skip"), 2, 1)

    # 1: total loss with the best epoch
    val = _col(df, "val_loss")
    _line(fig, x, _col(df, "loss"), row=1, col=1, name="train", color=T.INK_2, dash=T.TRAIN_DASH, group="total",
          legend="legend2")
    _line(fig, x, val, row=1, col=1, name="validation", color=T.INK, group="total-val", legend="legend2")
    if val is not None:
        b = int(np.nanargmin(val))
        fig.add_trace(go.Scatter(x=[x[b]], y=[val[b]], mode="markers", name="best", legend="legend2",
                                 marker=dict(symbol="star", size=14, color=T.WARNING, line=dict(color=T.PAPER, width=1)),
                                 hovertemplate=f"best val loss {val[b]:.4f} at epoch {x[b]}<extra></extra>"), 1, 1)

    # 2: loss components, val solid / train dotted, same colour
    for i, (key, label) in enumerate(LOSS_COMPONENTS):
        color = T.SERIES[i % len(T.SERIES)]
        v, t = T.positive(_col(df, f"val_{key}")), T.positive(_col(df, key))
        if v is None and t is None:
            continue
        _line(fig, x, v, row=1, col=2, name=label, color=color, group=f"c-{key}", width=1.5, legend="legend3")
        _line(fig, x, t, row=1, col=2, name=f"{label} (train)", color=color, dash=T.TRAIN_DASH, group=f"c-{key}",
              show=v is None, width=1.5, legend="legend3")
    fig.update_yaxes(type="log", row=1, col=2)

    # 3-8: per-horizon metrics: val solid, train dotted
    def horizons(row, col, val_key, train_key=None, *, alt_key=None):
        for h in T.HORIZONS:
            c = T.HORIZON_COLORS[h]
            _line(fig, x, _col(df, val_key.format(h=h)), row=row, col=col, name=T.horizon_label(h, config),
                  color=c, group=h, show=first(h))
            if train_key:
                _line(fig, x, _col(df, train_key.format(h=h)), row=row, col=col, name=f"{h} train", color=c,
                      dash=T.TRAIN_DASH, group=h, show=False, width=1.5)
            if alt_key:
                _line(fig, x, _col(df, alt_key.format(h=h)), row=row, col=col, name=f"{h} true", color=c,
                      dash=T.ALT_DASH, group=h, show=False, width=1.5)

    horizons(2, 1, "val_dir_mcc_{h}", "train_dir_mcc_{h}")
    horizons(2, 2, "val_gauss_dir_mcc_{h}", "train_gauss_dir_mcc_{h}")
    horizons(3, 1, "val_dir_bal_acc_{h}", "train_dir_bal_acc_{h}")
    horizons(3, 2, "val_dir_ece_{h}", "train_dir_ece_{h}")
    horizons(4, 1, "val_pit_ks_{h}", "pit_ks_{h}")
    horizons(4, 2, "val_pred_up_rate_{h}", alt_key="val_true_up_rate_{h}")
    for r, c, y in ((2, 1, 0.0), (2, 2, 0.0), (3, 1, 0.5), (4, 2, 0.5)):
        fig.add_hline(y=y, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=r, col=c)

    # 9: physics terms
    for i, (key, label) in enumerate(PHYSICS_TERMS):
        v, t = T.positive(_col(df, f"val_{key}")), T.positive(_col(df, key))
        v_ok = v is not None and np.isfinite(v).any()
        t_ok = t is not None and np.isfinite(t).any()
        if not (v_ok or t_ok):
            continue   # a term switched off (lambda 0) is left out rather than drawn at zero on a log axis
        color = T.SERIES[(i + 3) % len(T.SERIES)]
        _line(fig, x, v if v_ok else None, row=5, col=1, name=label, color=color, group=f"p-{key}", width=1.5,
              legend="legend4")
        _line(fig, x, t if t_ok else None, row=5, col=1, name=f"{label} (train)", color=color, dash=T.TRAIN_DASH,
              group=f"p-{key}", show=not v_ok, width=1.5, legend="legend4")
    fig.update_yaxes(type="log", row=5, col=1)

    # 10: learning rates
    lr, lri = _col(df, "lr"), _col(df, "lr_indicator")
    _line(fig, x, lr, row=5, col=2, name="network", color=T.SERIES[6], group="lr", legend="legend5")
    _line(fig, x, lri, row=5, col=2, name="indicator periods", color=T.SERIES[4], group="lri", legend="legend5")
    fig.update_yaxes(type="log", row=5, col=2)
    present = [a for a in (lr, lri) if a is not None]
    if present:
        _log_ticks(fig, 5, 2, np.concatenate(present))

    # 11: gradient norm, with non-finite steps flagged
    gn = _col(df, "grad_global_norm")
    _line(fig, x, gn, row=6, col=1, name="grad norm", color=T.SERIES[0], group="gn", show=False)
    nf = _col(df, "nonfinite_grad_steps")
    if nf is not None and gn is not None and np.nansum(nf) > 0:
        bad = np.where(nf > 0)[0]
        fig.add_trace(go.Scatter(x=[x[i] for i in bad], y=gn[bad], mode="markers", name="non-finite steps",
                                 marker=dict(symbol="x", size=11, color=T.CRITICAL),
                                 hovertext=[f"{int(nf[i])} non-finite steps" for i in bad]), 6, 1)

    # 12: epoch time
    _line(fig, x, _col(df, "epoch_seconds"), row=6, col=2, name="s / epoch", color=T.SERIES[3], group="sec",
          show=False)

    for c in (1, 2):
        fig.update_xaxes(title_text="epoch", row=6, col=c)
    T.note_on_empty(fig)
    T.apply(fig, title=title or "Training dashboard", subtitle=_title_line(df, config), height=height)
    for legend_id, (r, c) in (("legend2", (1, 1)), ("legend3", (1, 2)), ("legend4", (5, 1)), ("legend5", (5, 2))):
        T.panel_legend(fig, legend_id, r, c, PANELS[(r - 1) * 2 + (c - 1)])
    fig.update_layout(hovermode="x unified", margin=dict(t=130), legend=dict(y=1.03))
    return fig


def _title_line(df, config) -> str:
    checks = training_health(df, config)
    parts = [f"epoch {checks['epochs']['value']}"]
    for k in ("best", "patience", "learning rate", "time"):
        if k in checks:
            parts.append(f"{k} {checks[k]['value']}")
    return " · ".join(parts) + " · solid = validation, dotted = training"


def batch_loss_figure(points: Sequence[Tuple[float, float]], val_points: Sequence[Tuple[float, float]] = (),
                      *, height: int = 240):
    """Live strip: the running-mean training loss after each batch (x in fractional epochs) and the
    validation loss at each epoch end."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if points:
        xs, ys = zip(*points)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="train (running mean, per batch)",
                                 line=dict(color=T.INK_2, width=1.5)))
    if val_points:
        xs, ys = zip(*val_points)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="validation (epoch end)",
                                 line=dict(color=T.INK, width=2)))
    T.apply(fig, title="Loss, batch by batch", height=height)
    fig.update_layout(margin=dict(t=56, b=36), xaxis_title="epoch")
    return fig


# ------------------------------------------------------------------ registry entries (data, config)
def training_dashboard(data, config=None, **kw):
    return training_dashboard_figure(data, config, **kw)

