# Registry-Based Modular Architecture Specifications
## Neural Trade Project

**Version:** 1.0
**Date:** 2026-01-14
**Status:** Planning Phase

---

## Executive Summary

This document specifies a comprehensive registry-based architecture for the neural_trade codebase, enabling drop-in auto-integrated modular structure for all major code segments. The design builds upon the existing `losses.py` registry pattern and extends it across the entire codebase.

### Goals
1. **Modularity**: Each component type has its own registry and module
2. **Extensibility**: Adding new components requires minimal code changes
3. **Discoverability**: All available components are easily enumerable
4. **Consistency**: Uniform API across all registries
5. **Backward Compatibility**: Existing code continues to work
6. **Testing**: Each registry is independently testable

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Registry Core Specification](#registry-core-specification)
3. [Component Registry Specifications](#component-registry-specifications)
4. [Integration Patterns](#integration-patterns)
5. [Auto-Discovery Mechanism](#auto-discovery-mechanism)
6. [Testing Strategy](#testing-strategy)
7. [Migration Path](#migration-path)
8. [Implementation Plan](#implementation-plan)

---

## Architecture Overview

### Directory Structure

```
neural_trade/
├── core/
│   ├── __init__.py                      # Core exports
│   ├── registry.py                      # Base registry class
│   ├── config.py                        # Central configuration
│   └── exceptions.py                    # Custom exceptions
│
├── registries/
│   ├── __init__.py                      # Auto-discovery loader
│   ├── losses.py                        # Loss functions (EXISTING)
│   ├── models.py                        # Model architectures
│   ├── optimizers.py                    # Optimizer configurations
│   ├── metrics.py                       # Evaluation metrics
│   ├── callbacks.py                     # Training callbacks
│   ├── data_loaders.py                  # Data loading strategies
│   ├── visualizations.py                # Visualization backends
│   ├── layers.py                        # Custom layer components
│   └── preprocessors.py                 # Data preprocessing pipelines
│
├── models/
│   ├── __init__.py
│   ├── base.py                          # Base model interface
│   ├── gru_attention.py                 # Current architecture (extracted)
│   ├── lstm_transformer.py              # Alternative architecture
│   └── conv_net.py                      # CNN-based architecture
│
├── data/
│   ├── __init__.py
│   ├── processor.py                     # DataProcessor (extracted from model.py)
│   ├── loaders/
│   │   ├── csv_loader.py
│   │   ├── database_loader.py
│   │   └── api_loader.py
│   └── preprocessors/
│       ├── scalers.py
│       ├── feature_engineering.py
│       └── sequence_generators.py
│
├── visualization/
│   ├── __init__.py
│   ├── plotly_backend.py                # Current implementation
│   ├── matplotlib_backend.py            # Alternative backend
│   └── dashboard.py                     # Live training dashboard
│
├── utils/
│   ├── __init__.py
│   ├── metrics_utils.py                 # EXISTING
│   └── validation.py                    # Input validation utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_losses.py                   # EXISTING
│   ├── test_models_registry.py
│   ├── test_optimizers_registry.py
│   ├── test_metrics_registry.py
│   ├── test_callbacks_registry.py
│   ├── test_data_loaders_registry.py
│   ├── test_visualizations_registry.py
│   ├── test_layers_registry.py
│   └── test_integration.py              # End-to-end tests
│
├── model.py                             # Legacy entry point (refactored)
├── losses.py                            # EXISTING (move to registries/)
├── metrics_utils.py                     # EXISTING (move to utils/)
└── config.yaml                          # NEW: YAML-based configuration
```

### Design Principles

1. **Single Responsibility**: Each registry manages one component type
2. **Open/Closed**: Open for extension, closed for modification
3. **Dependency Inversion**: Depend on abstractions (registries), not concrete implementations
4. **Don't Repeat Yourself**: Shared patterns extracted to base registry
5. **Convention over Configuration**: Sensible defaults with override capability

---

## Registry Core Specification

### Base Registry Class

**File:** `core/registry.py`

```python
"""
Base registry implementation for all component types.
Provides consistent API and functionality across all registries.
"""

from typing import Dict, Callable, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import warnings

T = TypeVar('T')

@dataclass
class RegistryEntry(Generic[T]):
    """Metadata for a registered component."""
    name: str
    component: T
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    registered_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
            "dependencies": self.dependencies,
        }


class BaseRegistry(ABC):
    """
    Abstract base class for all registries.

    Provides:
    - Component registration via decorator
    - Component retrieval with validation
    - Listing and filtering
    - Metadata management
    - Auto-discovery support
    """

    registry: Dict[str, RegistryEntry] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        author: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        override: bool = False,
    ):
        """
        Decorator to register a component.

        Args:
            name: Component name (defaults to function/class name)
            description: Human-readable description
            tags: Categorization tags
            version: Semantic version
            author: Component author
            dependencies: Required dependencies
            override: Allow overriding existing registration

        Example:
            @Models.register(name="gru_attention", tags=["rnn", "attention"])
            def build_gru_attention_model(config):
                ...
        """
        def decorator(component):
            component_name = name or component.__name__

            # Check for duplicates
            if component_name in cls.registry and not override:
                warnings.warn(
                    f"{cls.__name__}: '{component_name}' already registered. "
                    f"Use override=True to replace.",
                    UserWarning
                )
                return component

            # Extract description from docstring if not provided
            component_description = description
            if component_description is None and component.__doc__:
                component_description = component.__doc__.split('\n')[0].strip()

            # Create registry entry
            entry = RegistryEntry(
                name=component_name,
                component=component,
                description=component_description,
                tags=tags or [],
                version=version,
                author=author,
                dependencies=dependencies or [],
            )

            cls.registry[component_name] = entry

            # Add metadata to component
            component._registry_name = component_name
            component._registry_entry = entry

            return component

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """
        Retrieve a registered component.

        Args:
            name: Component name
            **kwargs: Additional arguments passed to component if callable

        Returns:
            The registered component (optionally instantiated)

        Raises:
            KeyError: If component not found
        """
        if name not in cls.registry:
            available = ", ".join(cls.list_names())
            raise KeyError(
                f"{cls.__name__}: '{name}' not found. "
                f"Available: {available}"
            )

        entry = cls.registry[name]
        component = entry.component

        # If kwargs provided and component is callable, call it
        if kwargs and callable(component):
            return component(**kwargs)

        return component

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """List all registered components with metadata."""
        return [entry.to_dict() for entry in cls.registry.values()]

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered component names."""
        return sorted(cls.registry.keys())

    @classmethod
    def filter_by_tag(cls, tag: str) -> List[str]:
        """Filter components by tag."""
        return [
            name for name, entry in cls.registry.items()
            if tag in entry.tags
        ]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if component is registered."""
        return name in cls.registry

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove a registered component (for testing)."""
        if name in cls.registry:
            del cls.registry[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered components (for testing)."""
        cls.registry.clear()
        cls._initialized = False

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a registered component."""
        if name not in cls.registry:
            raise KeyError(f"{cls.__name__}: '{name}' not found")
        return cls.registry[name].to_dict()

    @classmethod
    @abstractmethod
    def validate_component(cls, component: Any) -> bool:
        """
        Validate that a component meets registry requirements.

        To be implemented by subclasses.
        """
        pass

    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover and register components from module.

        To be implemented by subclasses if auto-discovery is needed.
        """
        pass


class RegistryMixin:
    """
    Mixin to add registry functionality to existing classes.

    Example:
        class CustomTrainModel(Model, RegistryMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.loss_fn = self.get_from_registry(Losses, config.loss_name)
    """

    @staticmethod
    def get_from_registry(registry: BaseRegistry, name: str, **kwargs):
        """Get component from specified registry."""
        return registry.get(name, **kwargs)

    @staticmethod
    def list_from_registry(registry: BaseRegistry) -> List[str]:
        """List all components in specified registry."""
        return registry.list_names()
```

### Custom Exceptions

**File:** `core/exceptions.py`

```python
"""Custom exceptions for registry system."""

class RegistryError(Exception):
    """Base exception for registry errors."""
    pass

class ComponentNotFoundError(RegistryError):
    """Raised when a component is not found in registry."""
    pass

class ComponentValidationError(RegistryError):
    """Raised when a component fails validation."""
    pass

class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register duplicate component without override."""
    pass

class DependencyError(RegistryError):
    """Raised when component dependencies are not met."""
    pass
```

---

## Component Registry Specifications

### 1. Models Registry

**File:** `registries/models.py`

**Purpose:** Register model architecture builders

**Component Type:** Functions that return Keras/TF models

**Signature:**
```python
def model_builder(config: Config) -> tf.keras.Model:
    """
    Build and return a Keras model.

    Args:
        config: Configuration object with hyperparameters

    Returns:
        Compiled or uncompiled Keras model
    """
    pass
```

**Example Registration:**
```python
@Models.register(
    name="gru_attention",
    description="Bi-GRU with multi-head attention and transformer blocks",
    tags=["rnn", "attention", "transformer", "default"],
    version="2.0.0"
)
def build_gru_attention_model(config: Config) -> tf.keras.Model:
    """Current production architecture."""
    # Implementation from PricePredictor.build_model()
    ...

@Models.register(
    name="lstm_transformer",
    tags=["rnn", "transformer"]
)
def build_lstm_transformer_model(config: Config) -> tf.keras.Model:
    """LSTM-based alternative with pure transformer decoder."""
    ...

@Models.register(
    name="conv1d_attention",
    tags=["cnn", "attention", "fast"]
)
def build_conv1d_model(config: Config) -> tf.keras.Model:
    """Lightweight CNN-based architecture for fast inference."""
    ...
```

**Validation:**
- Must accept `config: Config` as first argument
- Must return `tf.keras.Model` instance
- Model must have expected input/output shapes

**Default:** `gru_attention` (current implementation)

---

### 2. Optimizers Registry

**File:** `registries/optimizers.py`

**Purpose:** Register optimizer configurations

**Component Type:** Functions that return configured optimizers

**Signature:**
```python
def optimizer_builder(config: Config) -> tf.keras.optimizers.Optimizer:
    """
    Build and return a configured optimizer.

    Args:
        config: Configuration object with learning rate, etc.

    Returns:
        Configured Keras optimizer
    """
    pass
```

**Example Registration:**
```python
@Optimizers.register(
    name="adam",
    description="Adam optimizer with optional weight decay",
    tags=["adaptive", "default"]
)
def build_adam(config: Config) -> tf.keras.optimizers.Optimizer:
    """Standard Adam optimizer."""
    return tf.keras.optimizers.Adam(
        learning_rate=config.LR,
        beta_1=config.ADAM_BETA1,
        beta_2=config.ADAM_BETA2,
        epsilon=config.ADAM_EPSILON,
        clipnorm=config.GRAD_CLIP_NORM if config.GRAD_CLIP_NORM else None,
    )

@Optimizers.register(name="adamw", tags=["adaptive", "weight_decay"])
def build_adamw(config: Config) -> tf.keras.optimizers.Optimizer:
    """Adam with decoupled weight decay."""
    return tf.keras.optimizers.AdamW(
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        clipnorm=config.GRAD_CLIP_NORM if config.GRAD_CLIP_NORM else None,
    )

@Optimizers.register(name="sgd_momentum", tags=["sgd", "momentum"])
def build_sgd_momentum(config: Config) -> tf.keras.optimizers.Optimizer:
    """SGD with momentum and optional Nesterov acceleration."""
    return tf.keras.optimizers.SGD(
        learning_rate=config.LR,
        momentum=config.SGD_MOMENTUM,
        nesterov=config.SGD_NESTEROV,
        clipnorm=config.GRAD_CLIP_NORM if config.GRAD_CLIP_NORM else None,
    )
```

**Validation:**
- Must accept `config: Config` as first argument
- Must return `tf.keras.optimizers.Optimizer` instance

**Default:** `adam`

---

### 3. Metrics Registry

**File:** `registries/metrics.py`

**Purpose:** Register evaluation metrics

**Component Type:** Functions that compute metrics from predictions and targets

**Signature:**
```python
def metric_function(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Compute metric value.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        **kwargs: Additional metric-specific parameters

    Returns:
        Metric value (scalar)
    """
    pass
```

**Example Registration:**
```python
@Metrics.register(name="mse", tags=["regression", "loss"])
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

@Metrics.register(name="safe_mape", tags=["regression", "percentage"])
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """Safe MAPE that handles zeros."""
    # Implementation from metrics_utils.py
    ...

@Metrics.register(name="direction_accuracy", tags=["classification", "direction"])
def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of predicting price movement direction."""
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    return np.mean(direction_true == direction_pred)

@Metrics.register(name="mcc", tags=["classification", "correlation"])
def matthews_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient for direction prediction."""
    # Implementation from _compute_all_horizon_metrics()
    ...
```

**Validation:**
- Must accept `y_true` and `y_pred` as first two arguments
- Must return numeric scalar value
- Should handle edge cases (empty arrays, zeros, etc.)

**Auto-Discovery:** Import from `metrics_utils.py`

---

### 4. Callbacks Registry

**File:** `registries/callbacks.py`

**Purpose:** Register custom training callbacks

**Component Type:** Keras callback classes or factory functions

**Signature:**
```python
def callback_builder(config: Config, **kwargs) -> tf.keras.callbacks.Callback:
    """
    Build and return a configured callback.

    Args:
        config: Configuration object
        **kwargs: Additional callback-specific parameters

    Returns:
        Configured Keras callback instance
    """
    pass
```

**Example Registration:**
```python
@Callbacks.register(
    name="tqdm_progress",
    description="TQDM progress bar for training",
    tags=["progress", "console"]
)
def build_tqdm_callback(config: Config, **kwargs) -> tf.keras.callbacks.Callback:
    """TQDM callback for progress visualization."""
    # Return TqdmCallback instance from model.py
    return TqdmCallback(**kwargs)

@Callbacks.register(name="interactive_plot", tags=["visualization", "plotly"])
def build_interactive_plot_callback(config: Config, **kwargs) -> tf.keras.callbacks.Callback:
    """Interactive Plotly visualization during training."""
    # Return callback from make_interactive_plot_callback()
    ...

@Callbacks.register(name="params_logger", tags=["logging", "mlflow"])
def build_params_logger(config: Config, **kwargs) -> tf.keras.callbacks.Callback:
    """Log hyperparameters to MLflow or Weights & Biases."""
    return ParamsLogger(**kwargs)

@Callbacks.register(name="early_stopping_ensemble", tags=["regularization"])
def build_multi_monitor_early_stopping(config: Config) -> List[tf.keras.callbacks.Callback]:
    """Multiple early stopping callbacks with different monitors."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_custom_loss',
            patience=config.EARLY_STOPPING_PATIENCE // 2,
            restore_best_weights=False
        ),
    ]
```

**Validation:**
- Must return `tf.keras.callbacks.Callback` instance or list of callbacks
- Callbacks must implement required Keras callback methods

**Default:** Includes all callbacks from current training setup

---

### 5. Data Loaders Registry

**File:** `registries/data_loaders.py`

**Purpose:** Register data loading strategies

**Component Type:** Functions that load and return raw data

**Signature:**
```python
def data_loader(config: Config, **kwargs) -> pd.DataFrame:
    """
    Load data from source.

    Args:
        config: Configuration object with data paths, etc.
        **kwargs: Loader-specific parameters

    Returns:
        DataFrame with OHLCV data and timestamp index
    """
    pass
```

**Example Registration:**
```python
@DataLoaders.register(
    name="csv",
    description="Load data from CSV file",
    tags=["file", "default"]
)
def load_from_csv(config: Config, file_path: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    path = file_path or config.DATA_PATH
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

@DataLoaders.register(name="postgres", tags=["database", "sql"])
def load_from_postgres(config: Config, query: Optional[str] = None) -> pd.DataFrame:
    """Load data from PostgreSQL database."""
    from sqlalchemy import create_engine
    engine = create_engine(config.DATABASE_URL)
    query = query or f"SELECT * FROM {config.TABLE_NAME} ORDER BY timestamp"
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

@DataLoaders.register(name="binance_api", tags=["api", "exchange"])
def load_from_binance(config: Config, symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Load data from Binance API."""
    import ccxt
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=config.TIMEFRAME)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df
```

**Validation:**
- Must return `pd.DataFrame`
- DataFrame must have timestamp index
- Must contain required columns (OHLCV)

**Default:** `csv` (current implementation)

---

### 6. Visualizations Registry

**File:** `registries/visualizations.py`

**Purpose:** Register visualization backends and components

**Component Type:** Functions that create plots or dashboards

**Signature:**
```python
def visualization_function(data: Any, config: Config, **kwargs) -> Any:
    """
    Create visualization.

    Args:
        data: Data to visualize (predictions, metrics, etc.)
        config: Configuration object
        **kwargs: Visualization-specific parameters

    Returns:
        Plot object, figure, or None (if displaying inline)
    """
    pass
```

**Example Registration:**
```python
@Visualizations.register(
    name="plotly_interactive",
    description="Interactive Plotly visualization",
    tags=["plotly", "interactive", "default"]
)
def create_plotly_plot(data: Dict, config: Config, **kwargs):
    """Interactive plot using Plotly."""
    # Implementation from make_interactive_plot_callback()
    ...

@Visualizations.register(name="matplotlib_static", tags=["matplotlib", "publication"])
def create_matplotlib_plot(data: Dict, config: Config, **kwargs):
    """Static publication-quality plot using Matplotlib."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Plot predictions, errors, direction, calibration
    ...
    return fig

@Visualizations.register(name="tensorboard", tags=["tensorboard", "logging"])
def log_to_tensorboard(data: Dict, config: Config, **kwargs):
    """Log metrics and plots to TensorBoard."""
    # Create TensorBoard callback with custom scalars
    ...

@Visualizations.register(name="wandb_dashboard", tags=["wandb", "cloud"])
def create_wandb_dashboard(data: Dict, config: Config, **kwargs):
    """Create Weights & Biases dashboard."""
    import wandb
    # Log metrics, predictions, and custom plots to W&B
    ...
```

**Validation:**
- Must accept data dictionary as first argument
- Should handle various data formats gracefully

**Default:** `plotly_interactive`

---

### 7. Layers Registry

**File:** `registries/layers.py`

**Purpose:** Register custom Keras layers

**Component Type:** Keras Layer classes or factory functions

**Signature:**
```python
def layer_builder(**kwargs) -> tf.keras.layers.Layer:
    """
    Build and return a custom layer.

    Args:
        **kwargs: Layer-specific configuration

    Returns:
        Keras Layer instance
    """
    pass
```

**Example Registration:**
```python
@Layers.register(
    name="learnable_indicators",
    description="Learnable technical indicators with meta-learning",
    tags=["indicators", "meta_learning", "default"]
)
def build_learnable_indicators(**kwargs) -> tf.keras.layers.Layer:
    """LearnableIndicators layer from model.py."""
    return LearnableIndicators(**kwargs)

@Layers.register(name="positional_encoding", tags=["attention", "transformer"])
def build_positional_encoding(**kwargs) -> tf.keras.layers.Layer:
    """Transformer-style positional encoding."""
    return PositionalEncodingLayer(**kwargs)

@Layers.register(name="wavenet_causal", tags=["cnn", "causal", "time_series"])
def build_wavenet_layer(**kwargs) -> tf.keras.layers.Layer:
    """WaveNet-style causal dilated convolution."""
    # New implementation
    ...

@Layers.register(name="squeeze_excitation", tags=["attention", "channel"])
def build_squeeze_excitation(**kwargs) -> tf.keras.layers.Layer:
    """Squeeze-and-Excitation attention for channels."""
    # New implementation
    ...
```

**Validation:**
- Must return `tf.keras.layers.Layer` instance or subclass
- Layer must be serializable for model saving

**Default:** Layers from current architecture

---

### 8. Preprocessors Registry

**File:** `registries/preprocessors.py`

**Purpose:** Register data preprocessing pipelines

**Component Type:** Functions that transform DataFrames

**Signature:**
```python
def preprocessor(df: pd.DataFrame, config: Config, **kwargs) -> pd.DataFrame:
    """
    Preprocess raw data.

    Args:
        df: Raw DataFrame
        config: Configuration object
        **kwargs: Preprocessor-specific parameters

    Returns:
        Preprocessed DataFrame
    """
    pass
```

**Example Registration:**
```python
@Preprocessors.register(
    name="standard_scaler",
    description="StandardScaler normalization",
    tags=["scaling", "default"]
)
def apply_standard_scaler(df: pd.DataFrame, config: Config, fit: bool = True) -> pd.DataFrame:
    """Apply StandardScaler to features."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if fit:
        scaler.fit(df)
    return pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns
    )

@Preprocessors.register(name="minmax_scaler", tags=["scaling", "bounded"])
def apply_minmax_scaler(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply MinMaxScaler to features."""
    ...

@Preprocessors.register(name="robust_scaler", tags=["scaling", "outlier_robust"])
def apply_robust_scaler(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply RobustScaler (median/IQR based)."""
    ...

@Preprocessors.register(name="log_returns", tags=["transform", "finance"])
def compute_log_returns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Convert prices to log returns."""
    ...
```

**Validation:**
- Must accept and return `pd.DataFrame`
- Must preserve index
- Should handle missing values appropriately

**Default:** Current preprocessing from DataProcessor

---

## Integration Patterns

### Pattern 1: Direct Registry Access

**Use Case:** Simple component retrieval in configuration phase

```python
from registries import Models, Optimizers, Losses

# Get model builder
model_builder = Models.get("gru_attention")
model = model_builder(config)

# Get optimizer
optimizer = Optimizers.get("adam", config=config)

# Get loss function
loss_fn = Losses.get("custom_loss")
```

### Pattern 2: Config-Driven Selection

**Use Case:** Dynamic component selection from configuration

```python
# In Config dataclass
@dataclass
class Config:
    MODEL_NAME: str = "gru_attention"
    OPTIMIZER_NAME: str = "adam"
    LOSS_NAME: str = "custom_loss"
    METRICS: List[str] = field(default_factory=lambda: ["mse", "safe_mape", "direction_accuracy"])
    CALLBACKS: List[str] = field(default_factory=lambda: ["tqdm_progress", "early_stopping_ensemble"])
    DATA_LOADER: str = "csv"
    VISUALIZATION: str = "plotly_interactive"

# In training script
def train_model(config: Config):
    # Load data
    data_loader = DataLoaders.get(config.DATA_LOADER)
    df = data_loader(config)

    # Build model
    model_builder = Models.get(config.MODEL_NAME)
    model = model_builder(config)

    # Get optimizer
    optimizer = Optimizers.get(config.OPTIMIZER_NAME, config=config)

    # Get loss
    loss_fn = Losses.get(config.LOSS_NAME)

    # Compile
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Get callbacks
    callbacks_list = []
    for callback_name in config.CALLBACKS:
        callback = Callbacks.get(callback_name, config=config)
        if isinstance(callback, list):
            callbacks_list.extend(callback)
        else:
            callbacks_list.append(callback)

    # Train
    model.fit(train_data, callbacks=callbacks_list, ...)
```

### Pattern 3: Registry Mixin in Custom Classes

**Use Case:** Classes that need to access multiple registries

```python
from core.registry import RegistryMixin
from registries import Losses, Metrics

class CustomTrainModel(tf.keras.Model, RegistryMixin):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Get components from registries
        self.loss_fn = self.get_from_registry(Losses, config.LOSS_NAME)
        self.metrics_fns = {
            name: self.get_from_registry(Metrics, name)
            for name in config.METRICS
        }

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)

        # Compute gradients and update
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute metrics
        metrics = {name: fn(y, y_pred) for name, fn in self.metrics_fns.items()}

        return {"loss": loss, **metrics}
```

### Pattern 4: Auto-Discovery on Import

**Use Case:** Automatically register all components when module is imported

```python
# In registries/__init__.py
"""
Auto-discovery and registration of all components.
Import this module to automatically register all available components.
"""

# Import all registry modules to trigger registration
from . import losses
from . import models
from . import optimizers
from . import metrics
from . import callbacks
from . import data_loaders
from . import visualizations
from . import layers
from . import preprocessors

# Export all registries
from .losses import Losses
from .models import Models
from .optimizers import Optimizers
from .metrics import Metrics
from .callbacks import Callbacks
from .data_loaders import DataLoaders
from .visualizations import Visualizations
from .layers import Layers
from .preprocessors import Preprocessors

__all__ = [
    "Losses",
    "Models",
    "Optimizers",
    "Metrics",
    "Callbacks",
    "DataLoaders",
    "Visualizations",
    "Layers",
    "Preprocessors",
]

# Print registered components summary
def print_registry_summary():
    """Print summary of all registered components."""
    registries = [
        ("Models", Models),
        ("Optimizers", Optimizers),
        ("Losses", Losses),
        ("Metrics", Metrics),
        ("Callbacks", Callbacks),
        ("Data Loaders", DataLoaders),
        ("Visualizations", Visualizations),
        ("Layers", Layers),
        ("Preprocessors", Preprocessors),
    ]

    print("=" * 60)
    print("NEURAL TRADE - REGISTRY SUMMARY")
    print("=" * 60)
    for name, registry in registries:
        count = len(registry.list_names())
        print(f"{name:20s} {count:3d} registered")
    print("=" * 60)

# Auto-print summary on import (optional, controlled by env var)
import os
if os.getenv("NEURAL_TRADE_SHOW_REGISTRY_SUMMARY", "false").lower() == "true":
    print_registry_summary()
```

### Pattern 5: CLI for Registry Inspection

**Use Case:** Command-line tool to list and inspect available components

```python
# In scripts/registry_cli.py
"""
Command-line interface for registry inspection.

Usage:
    python -m scripts.registry_cli list models
    python -m scripts.registry_cli info models gru_attention
    python -m scripts.registry_cli search --tag rnn
"""

import argparse
from registries import *

def list_components(registry_name: str):
    """List all components in a registry."""
    registry = globals()[registry_name.capitalize()]
    print(f"\n{registry_name.upper()} ({len(registry.list_names())} registered)\n")
    for entry in registry.list():
        tags_str = f"[{', '.join(entry['tags'])}]" if entry['tags'] else ""
        print(f"  • {entry['name']:30s} {tags_str}")
        if entry['description']:
            print(f"    {entry['description']}")
        print()

def show_info(registry_name: str, component_name: str):
    """Show detailed info about a component."""
    registry = globals()[registry_name.capitalize()]
    metadata = registry.get_metadata(component_name)
    print(f"\n{component_name.upper()}\n")
    for key, value in metadata.items():
        print(f"  {key:15s}: {value}")

def search_by_tag(tag: str):
    """Search all registries for components with a tag."""
    registries = [Models, Optimizers, Losses, Metrics, Callbacks,
                  DataLoaders, Visualizations, Layers, Preprocessors]

    print(f"\nComponents tagged with '{tag}':\n")
    for registry in registries:
        matches = registry.filter_by_tag(tag)
        if matches:
            print(f"  {registry.__name__}:")
            for name in matches:
                print(f"    • {name}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Trade Registry CLI")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("registry", help="Registry name")

    # Info command
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("registry", help="Registry name")
    info_parser.add_argument("component", help="Component name")

    # Search command
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--tag", required=True, help="Tag to search for")

    args = parser.parse_args()

    if args.command == "list":
        list_components(args.registry)
    elif args.command == "info":
        show_info(args.registry, args.component)
    elif args.command == "search":
        search_by_tag(args.tag)
```

---

## Auto-Discovery Mechanism

### Module-Level Registration

**Pattern:** Register components at module import time

```python
# In registries/models.py

from core.registry import BaseRegistry
from core.config import Config
import tensorflow as tf

class Models(BaseRegistry):
    """Registry for model architectures."""

    registry = {}

    @classmethod
    def validate_component(cls, component):
        """Validate model builder function."""
        import inspect
        sig = inspect.signature(component)
        # Check that first parameter is 'config'
        params = list(sig.parameters.keys())
        if not params or params[0] != 'config':
            raise ValueError(f"Model builder must have 'config' as first parameter")
        return True


# Register all models
# These decorators execute at import time

@Models.register(name="gru_attention", tags=["rnn", "attention", "default"])
def build_gru_attention_model(config: Config) -> tf.keras.Model:
    """Current production architecture."""
    # Implementation
    pass

@Models.register(name="lstm_transformer", tags=["rnn", "transformer"])
def build_lstm_transformer_model(config: Config) -> tf.keras.Model:
    """Alternative architecture."""
    # Implementation
    pass

# More registrations...
```

### Dynamic Discovery from Directory

**Pattern:** Scan directory for modules and import them

```python
# In registries/__init__.py

import os
import importlib
from pathlib import Path

def auto_discover_registries():
    """
    Auto-discover and import all registry modules.

    Scans the registries/ directory for Python modules and imports them,
    triggering all @register decorators.
    """
    registry_dir = Path(__file__).parent

    for file_path in registry_dir.glob("*.py"):
        if file_path.name.startswith("_"):
            continue

        module_name = file_path.stem
        try:
            importlib.import_module(f".{module_name}", package="registries")
        except Exception as e:
            print(f"Warning: Failed to import registry module '{module_name}': {e}")

# Execute auto-discovery on import
auto_discover_registries()
```

### Plugin System for External Components

**Pattern:** Load components from external plugins directory

```python
# In core/plugin_loader.py

import os
import sys
import importlib.util
from pathlib import Path
from typing import List

def load_plugins(plugin_dir: str = "plugins") -> List[str]:
    """
    Load external plugins from a directory.

    Plugins should be Python modules that register components
    with existing registries.

    Args:
        plugin_dir: Directory containing plugin modules

    Returns:
        List of loaded plugin names
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        return []

    loaded_plugins = []

    for plugin_file in plugin_path.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue

        plugin_name = plugin_file.stem

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            loaded_plugins.append(plugin_name)
            print(f"Loaded plugin: {plugin_name}")

        except Exception as e:
            print(f"Warning: Failed to load plugin '{plugin_name}': {e}")

    return loaded_plugins

# Example plugin structure:
# plugins/
# ├── custom_models.py        # Registers additional models
# ├── experimental_losses.py  # Registers experimental losses
# └── special_metrics.py      # Registers domain-specific metrics
```

### Example External Plugin

```python
# In plugins/custom_models.py

"""
External plugin that registers custom model architectures.
"""

from registries import Models
from core.config import Config
import tensorflow as tf

@Models.register(
    name="wavenet_predictor",
    description="WaveNet-style dilated causal convolution model",
    tags=["cnn", "causal", "plugin"],
    author="External Contributor"
)
def build_wavenet_model(config: Config) -> tf.keras.Model:
    """WaveNet architecture for time series prediction."""
    # Custom implementation
    inputs = tf.keras.Input(shape=(config.LOOKBACK, config.NUM_FEATURES))

    x = inputs
    skip_connections = []

    # Dilated causal convolutions
    for i in range(config.WAVENET_LAYERS):
        dilation_rate = 2 ** i
        conv = tf.keras.layers.Conv1D(
            filters=config.WAVENET_FILTERS,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='tanh'
        )(x)

        gate = tf.keras.layers.Conv1D(
            filters=config.WAVENET_FILTERS,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='sigmoid'
        )(x)

        x = tf.keras.layers.Multiply()([conv, gate])
        skip_connections.append(x)

    # Combine skip connections
    x = tf.keras.layers.Add()(skip_connections)
    x = tf.keras.layers.Activation('relu')(x)

    # Output heads (same as other models)
    # ... output layer implementation ...

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# This registration happens automatically when plugin is loaded!
```

---

## Testing Strategy

### 1. Registry Unit Tests

**File:** `tests/test_registry_base.py`

```python
"""
Unit tests for base registry functionality.
"""

import pytest
from core.registry import BaseRegistry, RegistryEntry

class TestRegistry(BaseRegistry):
    """Test registry implementation."""
    registry = {}

    @classmethod
    def validate_component(cls, component):
        return callable(component)

def test_register_component():
    """Test component registration."""
    TestRegistry.clear()

    @TestRegistry.register(name="test_func")
    def test_function():
        return 42

    assert TestRegistry.has("test_func")
    assert TestRegistry.get("test_func")() == 42

def test_duplicate_registration_warning():
    """Test warning on duplicate registration."""
    TestRegistry.clear()

    @TestRegistry.register(name="duplicate")
    def func1():
        return 1

    with pytest.warns(UserWarning):
        @TestRegistry.register(name="duplicate")
        def func2():
            return 2

    # First registration should be kept
    assert TestRegistry.get("duplicate")() == 1

def test_override_registration():
    """Test overriding registration."""
    TestRegistry.clear()

    @TestRegistry.register(name="override")
    def func1():
        return 1

    @TestRegistry.register(name="override", override=True)
    def func2():
        return 2

    # Second registration should replace first
    assert TestRegistry.get("override")() == 2

def test_component_not_found():
    """Test error when component not found."""
    TestRegistry.clear()

    with pytest.raises(KeyError, match="not found"):
        TestRegistry.get("nonexistent")

def test_list_components():
    """Test listing registered components."""
    TestRegistry.clear()

    @TestRegistry.register(name="func1")
    def f1():
        pass

    @TestRegistry.register(name="func2")
    def f2():
        pass

    names = TestRegistry.list_names()
    assert "func1" in names
    assert "func2" in names

def test_filter_by_tag():
    """Test filtering components by tag."""
    TestRegistry.clear()

    @TestRegistry.register(name="func1", tags=["tag_a"])
    def f1():
        pass

    @TestRegistry.register(name="func2", tags=["tag_b"])
    def f2():
        pass

    @TestRegistry.register(name="func3", tags=["tag_a", "tag_b"])
    def f3():
        pass

    tag_a_components = TestRegistry.filter_by_tag("tag_a")
    assert "func1" in tag_a_components
    assert "func3" in tag_a_components
    assert "func2" not in tag_a_components

def test_metadata():
    """Test metadata storage and retrieval."""
    TestRegistry.clear()

    @TestRegistry.register(
        name="documented_func",
        description="Test function",
        tags=["test"],
        version="1.0.0",
        author="Test Author"
    )
    def func():
        """Docstring."""
        pass

    metadata = TestRegistry.get_metadata("documented_func")
    assert metadata["name"] == "documented_func"
    assert metadata["description"] == "Test function"
    assert "test" in metadata["tags"]
    assert metadata["version"] == "1.0.0"
    assert metadata["author"] == "Test Author"
```

### 2. Component Registry Tests

Each registry should have its own test file:

- `tests/test_models_registry.py` - Test model registration and retrieval
- `tests/test_optimizers_registry.py` - Test optimizer registration
- `tests/test_metrics_registry.py` - Test metric computation
- `tests/test_callbacks_registry.py` - Test callback instantiation
- etc.

**Example:** `tests/test_models_registry.py`

```python
"""Tests for Models registry."""

import pytest
import tensorflow as tf
from registries import Models
from core.config import Config

@pytest.fixture
def config():
    """Provide test configuration."""
    return Config()

def test_models_registered():
    """Test that expected models are registered."""
    models = Models.list_names()
    assert "gru_attention" in models
    # Add more assertions as models are added

def test_build_model(config):
    """Test building a model from registry."""
    model_builder = Models.get("gru_attention")
    model = model_builder(config)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape is not None
    assert model.output_shape is not None

def test_model_signature(config):
    """Test model has correct input/output shapes."""
    model_builder = Models.get("gru_attention")
    model = model_builder(config)

    # Check input shape
    expected_input_shape = (None, config.LOOKBACK, config.NUM_FEATURES)
    assert model.input_shape == expected_input_shape

    # Check output shapes (3 horizons × 3 outputs each)
    # ... shape assertions ...

def test_filter_models_by_tag():
    """Test filtering models by tag."""
    rnn_models = Models.filter_by_tag("rnn")
    assert "gru_attention" in rnn_models or "lstm_transformer" in rnn_models
```

### 3. Integration Tests

**File:** `tests/test_integration.py`

```python
"""
End-to-end integration tests for registry system.
"""

import pytest
from registries import Models, Optimizers, Losses, Callbacks
from core.config import Config

def test_end_to_end_model_training():
    """Test complete training pipeline using registries."""
    config = Config()

    # Get all components from registries
    model = Models.get(config.MODEL_NAME, config=config)
    optimizer = Optimizers.get(config.OPTIMIZER_NAME, config=config)
    loss_fn = Losses.get(config.LOSS_NAME)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Get callbacks
    callbacks_list = []
    for callback_name in config.CALLBACKS:
        callback = Callbacks.get(callback_name, config=config)
        if isinstance(callback, list):
            callbacks_list.extend(callback)
        else:
            callbacks_list.append(callback)

    # Create dummy data
    import numpy as np
    x_train = np.random.randn(100, config.LOOKBACK, config.NUM_FEATURES)
    y_train = np.random.randn(100, 9)  # 3 horizons × 3 outputs

    # Train for 1 epoch
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=1,
        callbacks=callbacks_list,
        verbose=0
    )

    assert history.history is not None
    assert "loss" in history.history

def test_plugin_loading():
    """Test loading external plugins."""
    from core.plugin_loader import load_plugins

    # Load plugins (if any exist)
    loaded = load_plugins("tests/test_plugins")

    # Verify plugin components are registered
    # (if test plugins exist)
    # ...
```

### 4. Test Coverage Goals

- **Base Registry:** 100% coverage
- **Component Registries:** 95%+ coverage
- **Integration Tests:** Cover all major workflows
- **Plugin System:** Test loading, errors, conflicts

---

## Migration Path

### Phase 1: Preparation (Non-Breaking)

**Goal:** Set up infrastructure without breaking existing code

1. Create `core/` directory with base registry
2. Create `registries/` directory structure
3. Move `losses.py` to `registries/losses.py` (update imports)
4. Create import aliases in root for backward compatibility:
   ```python
   # In root losses.py
   from registries.losses import Losses, focal_loss, dice_loss, ...
   ```
5. Update tests to use new import paths

**Deliverables:**
- [ ] `core/registry.py` with BaseRegistry
- [ ] `core/exceptions.py` with custom exceptions
- [ ] `registries/__init__.py` with auto-discovery
- [ ] Migration of `losses.py` complete
- [ ] All existing tests passing
- [ ] Backward compatibility verified

### Phase 2: Core Registries (Incremental)

**Goal:** Implement essential registries one by one

**2.1: Models Registry**
1. Create `registries/models.py`
2. Extract `PricePredictor.build_model()` to `build_gru_attention_model()`
3. Register as default model
4. Update `model.py` to use `Models.get(config.MODEL_NAME)`
5. Write tests in `tests/test_models_registry.py`

**2.2: Optimizers Registry**
1. Create `registries/optimizers.py`
2. Register `adam`, `adamw`, `sgd_momentum`
3. Update training code to use `Optimizers.get(config.OPTIMIZER_NAME)`
4. Write tests

**2.3: Metrics Registry**
1. Create `registries/metrics.py`
2. Move functions from `metrics_utils.py` to registry
3. Register all metrics
4. Update evaluation code to use registry
5. Write tests

**Deliverables per sub-phase:**
- [ ] Registry module created
- [ ] Default components registered
- [ ] Integration with existing code
- [ ] Tests written and passing
- [ ] Documentation updated

### Phase 3: Data & Visualization (Enhancement)

**Goal:** Add registries for data pipeline and visualization

**3.1: Data Loaders Registry**
1. Create `registries/data_loaders.py`
2. Extract data loading from `DataProcessor`
3. Register CSV loader (default)
4. Add database and API loaders (optional)
5. Update `DataProcessor` to use registry

**3.2: Preprocessors Registry**
1. Create `registries/preprocessors.py`
2. Extract preprocessing steps from `DataProcessor`
3. Register scaling, feature engineering, etc.
4. Make preprocessing pipeline configurable

**3.3: Visualizations Registry**
1. Create `registries/visualizations.py`
2. Extract `make_interactive_plot_callback()` to registry
3. Add alternative backends (matplotlib, etc.)
4. Update callbacks to use registry

**Deliverables:**
- [ ] Data loading fully configurable
- [ ] Preprocessing pipeline modular
- [ ] Multiple visualization backends available
- [ ] Tests and documentation

### Phase 4: Advanced Features (Extension)

**Goal:** Add advanced functionality and external plugins

**4.1: Callbacks Registry**
1. Create `registries/callbacks.py`
2. Register all custom callbacks
3. Make callback list fully configurable

**4.2: Layers Registry**
1. Create `registries/layers.py`
2. Register `LearnableIndicators`, `PositionalEncodingLayer`
3. Enable layer swapping in model architectures

**4.3: Plugin System**
1. Create `core/plugin_loader.py`
2. Create `plugins/` directory
3. Document plugin creation process
4. Add example plugins

**Deliverables:**
- [ ] All callbacks registered
- [ ] Custom layers modular
- [ ] Plugin system functional
- [ ] Developer documentation

### Phase 5: Cleanup & Optimization (Polish)

**Goal:** Refactor, optimize, and finalize

1. Refactor `model.py` to use all registries
2. Split monolithic `model.py` into focused modules:
   - `models/base.py` - Base classes
   - `models/gru_attention.py` - Default architecture
   - `data/processor.py` - DataProcessor class
   - `training/trainer.py` - Training orchestration
3. Remove all backward compatibility shims
4. Optimize import times
5. Add comprehensive documentation
6. Create migration guide for users

**Deliverables:**
- [ ] Codebase fully modularized
- [ ] No monolithic files
- [ ] All registries tested and documented
- [ ] Migration guide published
- [ ] Performance benchmarks

---

## Implementation Plan

### Phased Rollout Schedule

```
Week 1-2:   Phase 1 - Preparation
Week 3-4:   Phase 2.1 - Models Registry
Week 5:     Phase 2.2 - Optimizers Registry
Week 6:     Phase 2.3 - Metrics Registry
Week 7-8:   Phase 3 - Data & Visualization
Week 9-10:  Phase 4 - Advanced Features
Week 11-12: Phase 5 - Cleanup & Optimization
```

### Success Criteria

#### Phase 1
- [x] Losses registry migrated without breaking tests
- [x] Base registry infrastructure in place
- [ ] Backward compatibility maintained

#### Phase 2
- [ ] At least 2 model architectures registered
- [ ] At least 3 optimizers registered
- [ ] At least 10 metrics registered
- [ ] Config-driven component selection works
- [ ] All unit tests pass

#### Phase 3
- [ ] Data loading supports CSV, DB, and API
- [ ] At least 3 preprocessing pipelines registered
- [ ] At least 2 visualization backends available
- [ ] End-to-end tests pass

#### Phase 4
- [ ] All custom callbacks registered
- [ ] Custom layers extracted to registry
- [ ] Plugin system loads external components
- [ ] Example plugins provided

#### Phase 5
- [ ] `model.py` < 1000 lines
- [ ] Import time < 2 seconds
- [ ] 95%+ test coverage
- [ ] Documentation complete
- [ ] Zero breaking changes from Phase 4

### Risk Mitigation

**Risk:** Breaking existing notebooks and scripts
**Mitigation:** Maintain backward compatibility shims until Phase 5

**Risk:** Import time increases with auto-discovery
**Mitigation:** Lazy loading, optional auto-discovery, profiling

**Risk:** Registry overhead in training loops
**Mitigation:** Cache retrieved components, benchmark performance

**Risk:** Testing complexity increases
**Mitigation:** Comprehensive unit tests per registry, integration tests

**Risk:** Plugin conflicts and security
**Mitigation:** Plugin validation, namespace isolation, optional plugins

---

## Configuration Schema

### YAML Configuration Support

**File:** `config.yaml`

```yaml
# Neural Trade Configuration
# Registry-based modular configuration

# Component Selection
model:
  name: gru_attention
  architecture:
    lookback: 60
    num_features: 5
    gru_units: 128
    attention_heads: 4
    transformer_layers: 2

optimizer:
  name: adam
  learning_rate: 0.001
  beta_1: 0.9
  beta_2: 0.999
  epsilon: 1e-7
  gradient_clip_norm: 1.0

loss:
  name: custom_loss
  weights:
    point_loss: 1.0
    trend_loss: 0.5
    direction_loss: 0.3
    variance_loss: 0.2

metrics:
  - mse
  - safe_mape
  - direction_accuracy
  - mcc

callbacks:
  - tqdm_progress
  - early_stopping_ensemble
  - interactive_plot
  - params_logger

data:
  loader: csv
  path: ./data/price_data.csv
  preprocessors:
    - standard_scaler
    - log_returns

visualization:
  backend: plotly_interactive
  update_freq: 1  # Update every N epochs

# Data Configuration
data_config:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  batch_size: 32
  shuffle: true

# Training Configuration
training:
  epochs: 100
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

# Paths
paths:
  data_dir: ./data
  models_dir: ./models
  logs_dir: ./logs
  plugins_dir: ./plugins
```

### Config Loader

**File:** `core/config.py`

```python
"""Configuration management with YAML support."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

@dataclass
class Config:
    """Central configuration with registry-based component selection."""

    # Component names (registry keys)
    MODEL_NAME: str = "gru_attention"
    OPTIMIZER_NAME: str = "adam"
    LOSS_NAME: str = "custom_loss"
    METRICS: List[str] = field(default_factory=lambda: ["mse", "safe_mape", "direction_accuracy"])
    CALLBACKS: List[str] = field(default_factory=lambda: ["tqdm_progress", "early_stopping_ensemble"])
    DATA_LOADER: str = "csv"
    PREPROCESSORS: List[str] = field(default_factory=lambda: ["standard_scaler"])
    VISUALIZATION: str = "plotly_interactive"

    # Model architecture
    LOOKBACK: int = 60
    NUM_FEATURES: int = 5
    GRU_UNITS: int = 128
    ATTENTION_HEADS: int = 4
    TRANSFORMER_LAYERS: int = 2

    # Optimizer parameters
    LR: float = 0.001
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.999
    ADAM_EPSILON: float = 1e-7
    GRAD_CLIP_NORM: Optional[float] = 1.0

    # Loss weights
    LAMBDA_POINT: float = 1.0
    LAMBDA_TREND: float = 0.5
    LAMBDA_DIRECTION: float = 0.3
    LAMBDA_VARIANCE: float = 0.2

    # Data parameters
    DATA_PATH: str = "./data/price_data.csv"
    TRAIN_SPLIT: float = 0.8
    BATCH_SIZE: int = 32

    # Training parameters
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10

    # Paths
    MODELS_DIR: str = "./models"
    LOGS_DIR: str = "./logs"
    PLUGINS_DIR: str = "./plugins"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure
        flat_config = cls._flatten_dict(config_dict)

        return cls(**flat_config)

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}".upper() if parent_key else k.upper()
            if isinstance(v, dict):
                items.extend(Config._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self._to_nested_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _to_nested_dict(self) -> Dict[str, Any]:
        """Convert flat config to nested dictionary."""
        # Implementation of nested conversion
        ...
```

---

## Best Practices & Guidelines

### 1. Component Design Guidelines

**DO:**
- Keep components focused and single-purpose
- Use type hints for all function signatures
- Write comprehensive docstrings
- Handle edge cases gracefully
- Return consistent types
- Log important events and warnings

**DON'T:**
- Create components with side effects
- Use global state
- Hard-code file paths or constants
- Assume specific data formats without validation
- Silently fail or swallow exceptions

### 2. Registry Usage Guidelines

**DO:**
- Use descriptive, consistent naming conventions
- Add relevant tags for discoverability
- Provide clear descriptions
- Register components immediately after definition
- Use type hints in decorator parameters

**DON'T:**
- Register lambda functions (hard to debug)
- Use generic names like "model1", "model2"
- Register without documentation
- Override registrations unintentionally

### 3. Testing Guidelines

**DO:**
- Test each registry independently
- Test component retrieval and instantiation
- Test error cases (not found, invalid params, etc.)
- Test filtering and listing functionality
- Write integration tests for workflows

**DON'T:**
- Test implementation details
- Create tests with external dependencies
- Hard-code test data paths
- Skip edge cases

### 4. Documentation Guidelines

**DO:**
- Document all public APIs
- Provide usage examples
- Explain parameters and return values
- Document expected behavior and exceptions
- Keep documentation up-to-date

**DON'T:**
- Document internal implementation details
- Use jargon without explanation
- Assume prior knowledge

---

## Appendix

### A. Glossary

- **Registry:** A central repository that manages component registration and retrieval
- **Component:** A reusable code element (function, class) registered with a registry
- **Decorator:** Python syntax `@name` used to register components
- **Tag:** Categorization label for filtering components
- **Auto-Discovery:** Automatic detection and registration of components
- **Plugin:** External module that extends functionality via registries
- **Drop-in:** Component that can be used without code changes (config-driven)

### B. References

- **Existing Implementation:** `losses.py` (lines 1-477)
- **Model Architecture:** `model.py` PricePredictor class (lines 1478-1617)
- **Training Loop:** `model.py` train_and_evaluate function (lines 970-1261)
- **Data Processing:** `model.py` DataProcessor class (lines 145-381)

### C. Future Enhancements

1. **Hyperparameter Tuning Registry**
   - Register search strategies (grid, random, Bayesian)
   - Integrate with Optuna, Ray Tune, etc.

2. **Ensemble Methods Registry**
   - Register ensemble strategies (bagging, stacking, etc.)
   - Combine multiple models from Models registry

3. **Deployment Strategies Registry**
   - Register deployment targets (TF Lite, ONNX, TensorRT)
   - Streamline model export and serving

4. **Monitoring & Alerting Registry**
   - Register monitoring backends (Prometheus, Grafana)
   - Alert on performance degradation

5. **Auto-ML Integration**
   - Use registries to define search spaces
   - Automated architecture search

### D. Contact & Support

- **Documentation:** [Link to docs]
- **Issue Tracker:** [Link to issues]
- **Discussions:** [Link to forum/discussions]
- **Contributing:** See CONTRIBUTING.md

---

## Document Control

| Version | Date       | Author | Changes                          |
|---------|------------|--------|----------------------------------|
| 1.0     | 2026-01-14 | Claude | Initial specification            |

---

**End of Specifications Document**
