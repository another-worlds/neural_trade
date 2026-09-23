"""PricePredictor: legacy facade over the Models registry and the dataset builder."""
from __future__ import annotations

from neural_trade.data.datasets import create_datasets
from neural_trade.registries.models import Models

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from neural_trade.core.config import Config


class PricePredictor:
    def __init__(self, config: Config):
        self.config = config

    def build_model(self):
        """Build the configured architecture (Models registry, Config.MODEL_NAME)."""
        return Models.build(getattr(self.config, 'MODEL_NAME', None), self.config)

    def create_datasets(self, X_train, y_train, last_close_train, extended_trends_train,
                        X_test, y_test, last_close_test, extended_trends_test):
        return create_datasets(self.config, X_train, y_train, last_close_train, extended_trends_train,
                               X_test, y_test, last_close_test, extended_trends_test)
