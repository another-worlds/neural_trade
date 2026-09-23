"""Template: a numpy-tier evaluation metric. Copy, rename, implement; select it in Config.METRICS."""
import numpy as np

from neural_trade.registries.metrics import Metrics


@Metrics.register(name="my_metric", tags=["custom"])
def my_metric(y_true, y_pred, *, mask=None) -> float:
    """One line describing what the metric measures."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
