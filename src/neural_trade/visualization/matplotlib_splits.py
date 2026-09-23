"""Walk-forward split plot (moved from DataProcessor.plot_splits in B9)."""
from __future__ import annotations

import numpy as np


def plot_splits(df, start_idx, tscv, X_seq_len):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='BTC Close Price', alpha=0.8)
    split_boundaries = [0]
    for train_idx, test_idx in tscv.split(np.arange(X_seq_len)):
        split_boundaries.append(test_idx[0])
    split_boundaries.append(X_seq_len)
    colors = ['#fff8b0', '#d2f8d2']
    labels = ['Train', 'Test']
    used = set()
    for i in range(len(split_boundaries)-1):
        s = start_idx + split_boundaries[i]
        e = start_idx + split_boundaries[i+1]
        color = colors[i % 2]
        label = labels[i % 2] if labels[i % 2] not in used else ""
        used.add(labels[i % 2])
        ax.axvspan(df['Date'].iloc[s], df['Date'].iloc[e-1], color=color, alpha=0.2, label=label)
    ax.set_title('BTC Price with Walk-Forward Validation (Train=Yellow, Test=Green)')
    ax.set_xlabel('Date')
    ax.set_ylabel('BTC Price (USD)')
    ax.legend()
    plt.tight_layout()
    plt.show()
