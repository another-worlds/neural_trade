"""neural_trade: multi-horizon BTC forecasting with learnable indicators.

Importing the package is cheap and never imports TensorFlow; heavy modules
(training, models, serving) are imported on demand by the caller:

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate
"""
__version__ = "0.3.0"

__all__ = ["__version__"]
