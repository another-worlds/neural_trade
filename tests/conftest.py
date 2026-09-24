"""Shared pytest configuration and fixtures.

A bare ``pytest`` from the repository root works because ``pyproject.toml`` sets
``pythonpath = ["src", "."]``; no test module should touch ``sys.path``.

Markers (declared in pyproject.toml):
  tf        auto-applied to every test in a module that imports tensorflow
  slow      trains a model / takes more than ~30 s
  gpu       needs a GPU; never selected in CI
  data      needs the bundled binance_btcusdt_1min_ccxt.csv (skipped if absent)
  notebook  needs ipywidgets / IPython
"""
from __future__ import annotations

import functools
import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CSV = REPO_ROOT / "binance_btcusdt_1min_ccxt.csv"

# Environment that must be in place before TensorFlow is imported anywhere.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ.get("NEURAL_TRADE_FORCE_CPU") == "1":  # set by CI; opt-in locally
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_TF_IMPORT_RE = re.compile(r"^\s*(?:import tensorflow\b|from tensorflow\b)", re.MULTILINE)
_TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


@functools.lru_cache(maxsize=None)
def _module_imports_tensorflow(path: str) -> bool:
    try:
        return bool(_TF_IMPORT_RE.search(Path(path).read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return False


def pytest_ignore_collect(collection_path, config):
    """Without TensorFlow installed, skip collecting modules that import it.

    This is what lets ``pytest -m "not tf"`` run cleanly on a machine without TF
    instead of failing at collection.
    """
    if _TF_AVAILABLE:
        return None
    p = Path(str(collection_path))
    if p.suffix == ".py" and _module_imports_tensorflow(str(p)):
        return True
    return None


def pytest_collection_modifyitems(config, items):
    """Auto-mark ``tf`` tests and skip ``data`` tests when the CSV is missing."""
    for item in items:
        if _module_imports_tensorflow(str(item.fspath)) and "tf" not in item.keywords:
            item.add_marker(pytest.mark.tf)
        if "data" in item.keywords and not BUNDLED_CSV.exists():
            item.add_marker(pytest.mark.skip(reason=f"{BUNDLED_CSV.name} is not present"))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tf():
    """Import TensorFlow lazily; tests that need it skip cleanly when it is absent."""
    return pytest.importorskip("tensorflow")


@pytest.fixture
def make_loss_model(tf):
    """Factory for a ``CustomTrainModel`` that can evaluate the loss functions.

    In production ``CustomTrainModel`` is always built as a *functional* Keras model
    (``inputs=base.inputs, outputs=base.outputs``). The first time that happens Keras
    permanently rewrites the class's bases to include ``Functional``
    (``keras.engine.functional.inject_functional_model_class``), after which the bare
    ``CustomTrainModel(base_model=None, ...)`` form raises

        TypeError: Functional.__init__() missing 2 required positional arguments

    That makes any test using the bare form pass alone and fail inside a suite that
    trains anything first - a pure test-ordering artefact. Building the tiny functional
    graph below keeps these tests order-independent and matches production construction.
    The graph is never called; the loss tests pass the nine heads in explicitly.
    """
    from neural_trade.core.config import Config
    from neural_trade.training.custom_model import CustomTrainModel

    def _factory(pred_scale=261.0, pred_mean=3.2, *, config=None, lookback=60, **kwargs):
        inp = tf.keras.Input(shape=(lookback,), name="close_window")
        outs = [tf.keras.layers.Dense(1, name=f"head_{i}")(inp) for i in range(10)]
        base = tf.keras.Model(inp, outs, name="tiny_base")
        return CustomTrainModel(
            base_model=base, pred_scale=pred_scale, pred_mean=pred_mean,
            config=config if config is not None else Config(),
            inputs=base.inputs, outputs=base.outputs, **kwargs,
        )

    return _factory


@pytest.fixture(scope="session")
def realistic_scales() -> dict:
    """The scale regime the production data actually has.

    ``pred_scale``/``pred_mean`` are the std/mean of the raw training deltas
    (model.py: pred_scale = np.std(y_train)); ``last_close`` is a 2025 BTC price.
    ``tests/test_losses.py`` uses pred_scale=1, last_close=1 — the one parameterisation
    under which the trend-loss saturation bug is invisible.
    """
    return {"pred_scale": 261.0, "pred_mean": 3.2, "last_close": 110_000.0}


@pytest.fixture(scope="session")
def synthetic_close() -> np.ndarray:
    """A seeded geometric random walk: 3,000 one-minute closes around 110,000 with ~8 bps minute vol."""
    rng = np.random.default_rng(0)
    log_returns = rng.normal(0.0, 8e-4, size=3_000)
    return (110_000.0 * np.exp(np.cumsum(log_returns))).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_bars(synthetic_close: np.ndarray) -> pd.DataFrame:
    """OHLCV bars derived from ``synthetic_close`` in the bundled CSV's column layout."""
    rng = np.random.default_rng(1)
    close = synthetic_close.astype(np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 15.0, size=close.shape[0]))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.gamma(shape=2.0, scale=20.0, size=close.shape[0])
    timestamps = pd.date_range("2025-10-11 02:30:00", periods=close.shape[0], freq="1min", tz="UTC")
    return pd.DataFrame(
        {"datetime": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture(scope="session")
def real_slice() -> pd.DataFrame:
    """First 5,000 rows of the bundled Binance CSV (mark the consuming test ``data``)."""
    if not BUNDLED_CSV.exists():
        pytest.skip(f"{BUNDLED_CSV.name} is not present")
    return pd.read_csv(BUNDLED_CSV, nrows=5_000)


@pytest.fixture
def tiny_config(tf):
    """A ``Config`` sized for smoke tests. Requires TensorFlow (the smoke tests build models)."""
    from neural_trade.core.config import Config

    cfg = Config()
    cfg.EPOCHS = 1
    cfg.BATCH_SIZE = 16
    cfg.MAX_SEQUENCE_COUNT = 600
    cfg.PATIENCE = 1
    cfg.EARLY = 1
    cfg.validate()
    return cfg

# ---------------------------------------------------------------------------- dashboards (no TensorFlow)
VIZ_HORIZONS = ("h0", "h1", "h2")


@pytest.fixture(scope="session")
def viz_frame():
    from neural_trade.evaluation.frame import PredictionFrame

    rng = np.random.default_rng(0)
    n = 3000
    sigma = rng.uniform(60, 240, (n, 3))
    signal = rng.normal(0, 1, (n, 3))
    y = 0.2 * signal * sigma + rng.normal(0, 1, (n, 3)) * sigma       # a weak, real edge
    p = 1 / (1 + np.exp(-0.4 * signal))
    pred_scale = 100.0
    lo_hi = {h: (0.3 * signal[:, i] * sigma[:, i] - 1.645 * sigma[:, i],
                 0.3 * signal[:, i] * sigma[:, i] + 1.645 * sigma[:, i]) for i, h in enumerate(VIZ_HORIZONS)}
    return PredictionFrame(
        y=y, last_close=np.full(n, 100_000.0),
        delta={h: 0.3 * signal[:, i] * sigma[:, i] for i, h in enumerate(VIZ_HORIZONS)},
        direction_prob={h: p[:, i] for i, h in enumerate(VIZ_HORIZONS)},
        variance_scaled={h: (sigma[:, i] / pred_scale) ** 2 for i, h in enumerate(VIZ_HORIZONS)},
        pred_scale=pred_scale, direction_prob_calibrated={h: 0.5 + 0.8 * (p[:, i] - 0.5) for i, h in enumerate(VIZ_HORIZONS)},
        intervals=lo_hi)


@pytest.fixture(scope="session")
def viz_config():
    from neural_trade.core.config import Config

    return Config(EPOCHS=6, EARLY=3)


@pytest.fixture(scope="session")
def viz_history():
    """A factory: viz_history(n=6, full=True) -> epoch rows with the keys the trainer logs."""
    return _viz_history


def _viz_history(n=6, *, full=True):
    rng = np.random.default_rng(1)
    rows = []
    for e in range(n):
        r = {"epoch": e, "loss": 7 - 0.2 * e, "val_loss": 6 - 0.1 * e + 0.05 * rng.normal(), "seconds": 20.0}
        if full:
            for pre in ("", "val_"):
                for k in ("point_loss", "trend_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss",
                          "t_perp_loss", "casimir_loss", "hd_loss"):
                    r[pre + k] = float(rng.uniform(0.01, 1))
            for h in VIZ_HORIZONS:
                for pre in ("train_", "val_"):
                    for k in ("dir_mcc", "gauss_dir_mcc", "dir_bal_acc", "dir_ece"):
                        r[f"{pre}{k}_{h}"] = float(rng.uniform(0, 0.6))
                r[f"val_pred_up_rate_{h}"], r[f"val_true_up_rate_{h}"] = 0.48, 0.5
                r[f"val_pit_ks_{h}"], r[f"pit_ks_{h}"] = 0.1, 0.05
            r.update(lr=1e-3 if e < 4 else 5e-4, lr_indicator=5e-3, grad_global_norm=8.0, nonfinite_grad_steps=0.0)
        rows.append(r)
    return rows


@pytest.fixture(scope="session")
def viz_backtest(viz_frame):
    from neural_trade.strategy import Bars, SignalFrame, backtest, build_backtest_config, build_strategy

    rng = np.random.default_rng(2)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, len(viz_frame))))
    bars = Bars.from_close(close)
    import copy
    fr = copy.deepcopy(viz_frame)
    fr.last_close = close
    sig = SignalFrame.build(fr, 1.0)
    strat = build_strategy("calibrated_quantile", None, calibration=sig)
    res = backtest(sig, bars, strat, build_backtest_config({"random_seeds": 0}))
    assert res.summary["n_trades"] > 5
    return res, bars, sig, strat
