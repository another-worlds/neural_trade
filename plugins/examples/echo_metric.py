"""Example plugin: a numpy-tier metric that reports how many samples it was given."""
import numpy as np

from neural_trade.registries.metrics import Metrics


@Metrics.register(name="n_samples", tags=["diagnostic", "plugin"])
def n_samples(y_true, y_pred, *, mask=None) -> float:
    """Number of (optionally masked) samples; useful to report n next to every metric."""
    n = len(np.asarray(y_true).reshape(-1))
    if mask is not None:
        n = int(np.asarray(mask).reshape(-1).astype(bool)[:n].sum())
    return float(n)
