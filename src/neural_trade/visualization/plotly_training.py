"""Training-curve figures and the notebook's live dashboard callback (moved from model.py in B12).

* :func:`training_curves_figure` - loss and direction-metric curves from a history dict
  (``{"loss": [...], "val_loss": [...], "dir_acc_avg": [...], ...}``). Pure: no widgets.
* :func:`plotly_interactive` - registry entry; accepts a history dict, a Keras History, or
  a list of per-epoch dicts (metrics.jsonl rows) and adds the plotting aliases first.
* :func:`make_interactive_plot_callback` - the ipywidgets dashboard used during training
  in the notebook; it draws with :func:`training_curves_figure`.
"""
from __future__ import annotations

import logging

import time

import numpy as np

from neural_trade.visualization.aliases import add_plot_aliases
from neural_trade.visualization.qbox_dashboard import _qbox_dashboard_html

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from neural_trade.core.config import Config

_log = logging.getLogger(__name__)


def training_curves_figure(history, *, title=None, n_epochs=None):
    """Two-panel Plotly figure: epoch losses and direction metrics (dark theme)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    title_text = title or "Training curves"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Epoch Loss", "Epoch Direction Metrics"),
                        vertical_spacing=0.12)
    epochs = list(range(1, (n_epochs if n_epochs is not None else len(history.get('loss', []))) + 1))
    if 'loss' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['loss'], mode='lines+markers', name='loss', line=dict(color='#64B5F6')), row=1, col=1)
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], mode='lines+markers', name='val_loss', line=dict(color='#42A5F5')), row=1, col=1)
    if 'dir_acc_avg' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['dir_acc_avg'], mode='lines+markers', name='dir_acc_avg', line=dict(color='#81C784')), row=2, col=1)
    if 'val_dir_acc_avg' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_dir_acc_avg'], mode='lines+markers', name='val_dir_acc_avg', line=dict(color='#66BB6A')), row=2, col=1)
    # Balanced Accuracy: (Sensitivity + Specificity) / 2
    # Range [0, 1], 50% = random, class-imbalance robust (unlike accuracy)
    if 'bal_acc_avg' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['bal_acc_avg'], mode='lines+markers', name='bal_acc_avg', line=dict(color='#FFB74D')), row=2, col=1)
    if 'val_bal_acc_avg' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_bal_acc_avg'], mode='lines+markers', name='val_bal_acc_avg', line=dict(color='#FFA726')), row=2, col=1)

    # Add 50% dotted lines to metrics subplot (row 2)
    if epochs:
        fig.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=2, col=1, annotation_text="50%", annotation_position="right")

    # Calculate axis range with padding
    if epochs:
        x_min, x_max = min(epochs), max(epochs)
        x_padding = max(0.5, (x_max - x_min) * 0.05)  # 5% padding
    else:
        x_min, x_max, x_padding = 0, 1, 0.1

    # Dark theme styling with proper margins
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color='#e0e0e0'), x=0.5, xanchor='center'),
        height=560,
        showlegend=True,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0d0d0d',
        font=dict(color='#e0e0e0'),
        margin=dict(l=60, r=40, t=70, b=40),
        xaxis_showgrid=True,
        xaxis_gridwidth=1,
        xaxis_gridcolor='#333333',
        yaxis_showgrid=True,
        yaxis_gridwidth=1,
        yaxis_gridcolor='#333333',
        xaxis2_showgrid=True,
        xaxis2_gridwidth=1,
        xaxis2_gridcolor='#333333',
        yaxis2_showgrid=True,
        yaxis2_gridwidth=1,
        yaxis2_gridcolor='#333333',
    )

    # Set x-axis range with padding
    fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])

    # Set y-axis range for metrics subplot with padding
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)

    # Update axes styling
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)

    # Update subplot titles color
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#e0e0e0', size=12)
    return fig


def _as_history(data):
    """Normalise a Keras History, a history dict or a list of epoch dicts to {key: [values]}."""
    if hasattr(data, "history"):
        data = data.history
    if isinstance(data, list):
        keys = sorted({k for row in data for k in row})
        return {k: [row.get(k) for row in data] for k in keys}
    return dict(data or {})


def plotly_interactive(data, config=None, *, primary_horizon="h1", prefer_gauss=True, **_):
    """Registry entry: training curves figure from any history-like ``data``."""
    hist = _as_history(data)
    n = len(hist.get("loss", []))
    rows = [{k: v[i] for k, v in hist.items() if i < len(v)} for i in range(n)]
    for row in rows:
        add_plot_aliases(row, primary_horizon=primary_horizon, prefer_gauss=prefer_gauss)
    return training_curves_figure(_as_history(rows), n_epochs=n)


def make_interactive_plot_callback(
    *,
    config: 'Config',
    loss_output,
    metrics_output,
    progress_widget,
    total_epochs: int,
    batch_metrics_output=None,
    primary_horizon: str = "h1",
    prefer_gauss: bool = True,
    should_pause=None,
    should_stop=None,
    batch_update_interval: int = 1,
    epoch_info_widget=None,
):
    """Notebook-friendly interactive Plotly callback.

    This is a rewrite/encapsulation of the notebook's Cell 3 callback so notebooks can
    depend on `model.py` as the single source of truth.
    """
    import tensorflow as tf
    from IPython.display import clear_output, display
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    def _bool_call(maybe_callable) -> bool:
        try:
            return bool(maybe_callable()) if callable(maybe_callable) else bool(maybe_callable)
        except Exception:
            return False

    class _InteractivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {}
            self.epoch_count = 0
            self.batch_history = {}
            self.total_batches = None
            self.batch_count = 0
            self.batch_update_interval = max(1, int(batch_update_interval))

        def on_epoch_begin(self, epoch, logs=None):
            self.batch_history.clear()
            self.batch_count = 0
            self.total_batches = self.params.get('steps') if self.params is not None else None
            if batch_metrics_output is not None:
                with batch_metrics_output:
                    clear_output(wait=True)

        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            logs = add_plot_aliases(logs, primary_horizon=primary_horizon, prefer_gauss=prefer_gauss)
            batch_idx = (batch or 0) + 1
            self.batch_history.setdefault('batch', []).append(batch_idx)
            for key, value in logs.items():
                if value is None:
                    continue
                try:
                    self.batch_history.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue

            # Batch plot (loss + a couple key metrics)
            if batch_metrics_output is not None and self.batch_count % self.batch_update_interval == 0:
                with batch_metrics_output:
                    clear_output(wait=True)
                    batches = self.batch_history.get('batch', [])
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("Batch Loss", "Batch Direction Metrics"),
                                        vertical_spacing=0.12)
                    if 'loss' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['loss'], mode='lines', name='loss', line=dict(color='#64B5F6')), row=1, col=1)
                    if 'dir_acc' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['dir_acc'], mode='lines', name='dir_acc', line=dict(color='#81C784')), row=2, col=1)
                    if 'f1' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['f1'], mode='lines', name='f1', line=dict(color='#FFB74D')), row=2, col=1)
                    
                    # Add 50% dotted lines to metrics subplot (row 2)
                    if batches:
                        fig.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=2, col=1, annotation_text="50%", annotation_position="right")
                    
                    # Calculate axis range with padding to prevent data touching borders
                    if batches:
                        x_min, x_max = min(batches), max(batches)
                        x_padding = max(1, (x_max - x_min) * 0.03)  # 3% padding
                    else:
                        x_min, x_max, x_padding = 0, 1, 0.1
                    
                    # Dark theme styling with proper margins
                    fig.update_layout(
                        height=450,
                        showlegend=True,
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#0d0d0d',
                        font=dict(color='#e0e0e0'),
                        margin=dict(l=60, r=40, t=40, b=40),
                        xaxis_showgrid=True,
                        xaxis_gridwidth=1,
                        xaxis_gridcolor='#333333',
                        yaxis_showgrid=True,
                        yaxis_gridwidth=1,
                        yaxis_gridcolor='#333333',
                        xaxis2_showgrid=True,
                        xaxis2_gridwidth=1,
                        xaxis2_gridcolor='#333333',
                        yaxis2_showgrid=True,
                        yaxis2_gridwidth=1,
                        yaxis2_gridcolor='#333333',
                    )
                    
                    # Set x-axis range with padding (shared x-axis, only xaxis2 controls both)
                    fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
                    
                    # Set y-axis range for metrics subplot with padding
                    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
                    
                    # Update axes styling
                    fig.update_xaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                    fig.update_yaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                    
                    # Update subplot titles color
                    for annotation in fig['layout']['annotations']:
                        annotation['font'] = dict(color='#e0e0e0', size=12)
                    
                    display(fig)

            self.batch_count += 1

        def on_epoch_end(self, epoch, logs=None):
            # Optional notebook controls.
            # Keep this inside the callback so the notebook can remain a thin UI wrapper.
            if _bool_call(should_stop):
                self.model.stop_training = True
                return

            # Cooperative pause loop (safe no-op if not provided)
            while _bool_call(should_pause) and not _bool_call(should_stop):
                time.sleep(0.1)
            if _bool_call(should_stop):
                self.model.stop_training = True
                return

            logs = add_plot_aliases(logs or {}, primary_horizon=primary_horizon, prefer_gauss=prefer_gauss)
            self.epoch_count += 1
            for k, v in (logs or {}).items():
                if v is None:
                    continue
                try:
                    self.history.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    continue

            try:
                progress_widget.value = min(total_epochs, epoch + 1)
                progress_widget.description = f'Epoch ({epoch + 1}/{total_epochs}):'
            except Exception:
                _log.debug("dashboard widget update failed", exc_info=True)

            # Update epoch info widget if provided
            try:
                if epoch_info_widget is not None:
                    # Compute patience
                    patience_used_info = 0
                    if 'val_loss' in self.history and len(self.history['val_loss']) > 1:
                        best_idx = np.argmin(self.history['val_loss'])
                        patience_used_info = len(self.history['val_loss']) - 1 - best_idx
                    patience_max_info = getattr(config, 'PATIENCE', total_epochs)
                    
                    # Get current metrics
                    curr_loss = logs.get('loss', 0)
                    curr_val_loss = logs.get('val_loss', 0)
                    curr_dir_acc = logs.get('val_dir_acc_avg', 0)
                    curr_f1 = logs.get('val_f1_avg', 0)
                    
                    # Determine status color
                    if patience_used_info > patience_max_info * 0.8:
                        patience_color = "#EF5350"  # Red - close to stopping
                    elif patience_used_info > patience_max_info * 0.5:
                        patience_color = "#FFB74D"  # Orange - warning
                    else:
                        patience_color = "#81C784"  # Green - good
                    
                    epoch_info_widget.value = f"""
                    <div style="font-family: monospace; color: #e0e0e0; background-color: #1a1a1a; 
                                padding: 12px 20px; border-radius: 5px; text-align: center; 
                                border: 1px solid #333; margin-bottom: 10px;">
                        <span style="font-size: 18px; font-weight: bold; color: #64B5F6;">
                            🔄 Epoch {self.epoch_count}/{total_epochs}
                        </span>
                        <span style="color: {patience_color}; margin-left: 20px;">
                            ⏳ Patience: {patience_used_info}/{patience_max_info}
                        </span>
                        <span style="color: #64B5F6; margin-left: 20px;">
                            📉 Loss: {curr_loss:.4f}
                        </span>
                        <span style="color: #42A5F5; margin-left: 15px;">
                            Val: {curr_val_loss:.4f}
                        </span>
                        <span style="color: #81C784; margin-left: 20px;">
                            🎯 Acc: {curr_dir_acc:.1%}
                        </span>
                        <span style="color: #FFB74D; margin-left: 15px;">
                            F1: {curr_f1:.3f}
                        </span>
                    </div>
                    """
            except Exception:
                _log.debug("dashboard widget update failed", exc_info=True)

            # Epoch plots
            with loss_output:
                clear_output(wait=True)
                
                # Compute patience estimation (epochs since best val_loss)
                patience_used = 0
                if 'val_loss' in self.history and len(self.history['val_loss']) > 1:
                    best_val_loss_idx = np.argmin(self.history['val_loss'])
                    patience_used = len(self.history['val_loss']) - 1 - best_val_loss_idx
                patience_max = getattr(config, 'PATIENCE', total_epochs)
                
                # Compute key metrics for header
                current_loss = self.history.get('loss', [0])[-1] if 'loss' in self.history else 0
                current_val_loss = self.history.get('val_loss', [0])[-1] if 'val_loss' in self.history else 0
                current_dir_acc = self.history.get('val_dir_acc_avg', [0])[-1] if 'val_dir_acc_avg' in self.history else 0
                
                # Build title with epoch progress and metrics
                title_text = (f"<b>Epoch {self.epoch_count}/{total_epochs}</b> │ "
                             f"Patience: {patience_used}/{patience_max} │ "
                             f"Loss: {current_loss:.4f} │ Val Loss: {current_val_loss:.4f} │ "
                             f"Val Dir Acc: {current_dir_acc:.1%}")
                
                fig = training_curves_figure(self.history, title=title_text,
                                             n_epochs=self.epoch_count)
                display(fig)

            with metrics_output:
                clear_output(wait=True)
                
                # Compute convergence metrics
                loss_hist = self.history.get('loss', [])
                val_loss_hist = self.history.get('val_loss', [])
                
                # Convergence: rate of loss decrease (last 3 epochs)
                convergence_rate = 0.0
                if len(loss_hist) >= 3:
                    recent_losses = loss_hist[-3:]
                    convergence_rate = (recent_losses[0] - recent_losses[-1]) / (len(recent_losses) - 1) if len(recent_losses) > 1 else 0
                
                # Stability: std of recent validation losses
                stability = 0.0
                if len(val_loss_hist) >= 3:
                    stability = 1.0 - min(1.0, np.std(val_loss_hist[-5:]) * 10)  # Higher = more stable
                
                # Generalization gap: difference between train and val loss
                gen_gap = 0.0
                if loss_hist and val_loss_hist:
                    gen_gap = val_loss_hist[-1] - loss_hist[-1]
                
                # Coherence: How well train/val losses track each other (moving in same direction)
                # High coherence (>0.8) = model generalizes well, losses move together
                # Low/negative coherence = overfitting (train improves, val doesn't) or noise
                # Note: This measures train/val alignment, not cross-horizon consistency
                coherence = 0.0
                if len(loss_hist) >= 3 and len(val_loss_hist) >= 3:
                    try:
                        # Use direction agreement instead of correlation for robustness
                        # Direction: did loss increase or decrease between epochs?
                        train_diffs = np.diff(loss_hist[-10:])
                        val_diffs = np.diff(val_loss_hist[-10:])
                        if len(train_diffs) > 0 and len(val_diffs) > 0:
                            # Direction agreement: both increasing or both decreasing
                            train_dirs = np.sign(train_diffs)
                            val_dirs = np.sign(val_diffs)
                            agreement = np.mean(train_dirs == val_dirs)
                            coherence = agreement  # Range [0, 1], 1 = perfect agreement
                        else:
                            coherence = 0.5  # Neutral if not enough data
                    except Exception:
                        coherence = 0.0
                
                # Learning progress: improvement from initial
                progress = 0.0
                if len(val_loss_hist) >= 2:
                    progress = (val_loss_hist[0] - val_loss_hist[-1]) / val_loss_hist[0] if val_loss_hist[0] > 0 else 0
                
                # Build HTML output for dark theme visibility
                from IPython.display import HTML
                
                conv_status = "↓ converging" if convergence_rate > 0.001 else ("→ plateau" if abs(convergence_rate) < 0.001 else "↑ diverging")
                conv_color = "#81C784" if convergence_rate > 0.001 else ("#FFB74D" if abs(convergence_rate) < 0.001 else "#EF5350")
                
                stab_status = "stable" if stability > 0.8 else ("moderate" if stability > 0.5 else "unstable")
                stab_color = "#81C784" if stability > 0.8 else ("#FFB74D" if stability > 0.5 else "#EF5350")
                
                gap_status = "good" if gen_gap < 0.1 else ("warning" if gen_gap < 0.3 else "overfitting")
                gap_color = "#81C784" if gen_gap < 0.1 else ("#FFB74D" if gen_gap < 0.3 else "#EF5350")
                
                coh_status = "aligned" if coherence > 0.8 else ("moderate" if coherence > 0.5 else "misaligned")
                coh_color = "#81C784" if coherence > 0.8 else ("#FFB74D" if coherence > 0.5 else "#EF5350")
                
                html_content = f"""
                <div style="font-family: monospace; color: #e0e0e0; background-color: #0d0d0d; padding: 15px; border-radius: 5px;">
                    <div style="text-align: center; font-size: 16px; font-weight: bold; border-bottom: 2px solid #444; padding-bottom: 10px; margin-bottom: 15px;">
                        📊 EPOCH METRICS DASHBOARD
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="color: #64B5F6; font-weight: bold; margin-bottom: 8px;">📉 LOSSES</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">loss:</span> <span style="color: #64B5F6;">{logs.get('loss', 0):.6f}</span><br>
                            <span style="display: inline-block; width: 180px;">val_loss:</span> <span style="color: #42A5F5;">{logs.get('val_loss', 0):.6f}</span><br>
                            {'<span style="display: inline-block; width: 180px;">point_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('point_loss', 0):.6f}" + '</span><br>' if 'point_loss' in logs else ''}
                            {'<span style="display: inline-block; width: 180px;">dir_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('dir_loss', 0):.6f}" + '</span><br>' if 'dir_loss' in logs else ''}
                            {'<span style="display: inline-block; width: 180px;">nll_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('nll_loss', 0):.6f}" + '</span><br>' if 'nll_loss' in logs else ''}
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="color: #81C784; font-weight: bold; margin-bottom: 8px;">🎯 DIRECTION METRICS</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">dir_acc_avg:</span> <span style="color: #81C784;">{logs.get('dir_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">val_dir_acc_avg:</span> <span style="color: #66BB6A;">{logs.get('val_dir_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">bal_acc_avg:</span> <span style="color: #FFB74D;">{logs.get('bal_acc_avg', 0):.4f}</span> <span style="color: #888;">(50%=random)</span><br>
                            <span style="display: inline-block; width: 180px;">val_bal_acc_avg:</span> <span style="color: #FFA726;">{logs.get('val_bal_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">brier_avg:</span> <span style="color: #CE93D8;">{logs.get('brier_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">ece_avg:</span> <span style="color: #BA68C8;">{logs.get('ece_avg', 0):.4f}</span><br>
                        </div>
                    </div>
                    
                    {_qbox_dashboard_html(logs)}
                    
                    <div style="margin-bottom: 10px;">
                        <div style="color: #CE93D8; font-weight: bold; margin-bottom: 8px;">📈 TRAINING HEALTH</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">Convergence:</span> <span style="color: {conv_color};">{convergence_rate:+.6f} ({conv_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Stability:</span> <span style="color: {stab_color};">{stability:.4f} ({stab_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Gen. Gap:</span> <span style="color: {gap_color};">{gen_gap:+.6f} ({gap_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Coherence:</span> <span style="color: {coh_color};">{coherence:.4f} ({coh_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Progress:</span> <span style="color: {'#81C784' if progress > 0 else '#EF5350'};">{progress*100:+.2f}%</span><br>
                        </div>
                    </div>
                </div>
                """
                display(HTML(html_content))

    return _InteractivePlotCallback()
