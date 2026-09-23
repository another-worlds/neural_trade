"""The pre-package ``model.py`` API in one import (Phase B12).

``import model`` / ``from model import *`` keep working through the root ``model.py`` shim,
which aliases this module. New code should import from the owning modules instead:

    neural_trade.core.config.Config                    neural_trade.training.trainer.train_and_evaluate
    neural_trade.data.processor.DataProcessor          neural_trade.training.custom_model.CustomTrainModel
    neural_trade.models.layers.LearnableIndicators     neural_trade.visualization.plotly_training
"""
import math  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import warnings  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import tensorflow as tf  # noqa: F401
from tensorflow.keras import callbacks, initializers, layers, models, optimizers, regularizers  # noqa: F401

from neural_trade.calibration import CalibrationPipeline  # noqa: F401
from neural_trade.core.config import Config
from neural_trade.core.outputs import LossComponents, PredictiveOutputs
from neural_trade.data.datasets import create_datasets
from neural_trade.data.processor import DataProcessor
from neural_trade.data.splits import FoldIndices, make_purged_splits
from neural_trade.metrics.evaluate import _compute_all_horizon_metrics
from neural_trade.models.facade import PricePredictor
from neural_trade.models.layers import EnergyGate, LearnableIndicators, PositionalEncodingLayer, VacuumSaturationNoise
from neural_trade.registries.losses import Losses
from neural_trade.training.callbacks import ParamsLogger, TqdmCallback
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.trainer import (TrainResult, _apply_config_overrides, _calibration_coverage_report,
                                           _predict_heads, _temperature_of, train_and_evaluate, train_model)
from neural_trade.visualization.aliases import _first_present, _mean_present, _sum_present, add_plot_aliases
from neural_trade.visualization.plotly_training import make_interactive_plot_callback, training_curves_figure
from neural_trade.visualization.qbox_dashboard import _qbox_dashboard_html

__all__ = [
    "Config", "LossComponents", "PredictiveOutputs", "DataProcessor", "FoldIndices", "make_purged_splits",
    "create_datasets", "PricePredictor", "LearnableIndicators", "PositionalEncodingLayer",
    "VacuumSaturationNoise", "EnergyGate", "CustomTrainModel", "TrainResult", "train_and_evaluate",
    "train_model", "TqdmCallback", "ParamsLogger", "make_interactive_plot_callback",
    "training_curves_figure", "add_plot_aliases", "Losses", "CalibrationPipeline",
    "np", "pd", "tf", "layers", "models", "optimizers", "callbacks", "initializers", "regularizers",
]
