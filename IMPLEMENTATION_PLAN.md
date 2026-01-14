# Registry-Based Architecture - Implementation Plan
## Neural Trade Project

**Status:** Planning Phase
**Start Date:** 2026-01-14
**Target Completion:** 12 weeks
**Current Branch:** `claude/registry-modular-structure-1hpZb`

---

## Overview

This document provides a detailed, actionable implementation plan for transitioning the neural_trade codebase to a registry-based modular architecture. The plan is divided into 5 major phases, with each phase broken down into specific tasks.

---

## Phase Structure

```
PHASE 1: Foundation & Infrastructure    [Weeks 1-2]  ████████░░░░░░░░░░░░ 20%
PHASE 2: Core Registries                [Weeks 3-6]  ████████████░░░░░░░░ 40%
PHASE 3: Data & Visualization           [Weeks 7-8]  ████████████████░░░░ 60%
PHASE 4: Advanced Features              [Weeks 9-10] ████████████████████ 80%
PHASE 5: Refinement & Documentation     [Weeks 11-12] ████████████████████ 100%
```

---

## PHASE 1: Foundation & Infrastructure
**Duration:** 2 weeks
**Goal:** Set up core infrastructure without breaking existing functionality

### Milestone 1.1: Core Registry Framework
**Duration:** 3 days

#### Tasks

**1.1.1: Create directory structure** (2 hours)
```bash
mkdir -p core registries tests/registries utils models data visualization
```

- [ ] Create `core/` directory for base components
- [ ] Create `registries/` directory for all registries
- [ ] Create `models/` directory for model architectures
- [ ] Create `data/` directory for data processing
- [ ] Create `visualization/` directory for plotting
- [ ] Create `tests/registries/` for registry tests
- [ ] Verify structure matches specification

**1.1.2: Implement BaseRegistry class** (1 day)
File: `core/registry.py`

- [ ] Define `RegistryEntry` dataclass with metadata fields
- [ ] Implement `BaseRegistry` abstract class
  - [ ] `register()` decorator with full parameter support
  - [ ] `get()` method with component retrieval
  - [ ] `list()` and `list_names()` methods
  - [ ] `filter_by_tag()` method
  - [ ] `has()`, `remove()`, `clear()` methods
  - [ ] `get_metadata()` method
  - [ ] Abstract `validate_component()` method
- [ ] Implement `RegistryMixin` helper class
- [ ] Add comprehensive docstrings
- [ ] Validate against specification

**1.1.3: Create custom exceptions** (1 hour)
File: `core/exceptions.py`

- [ ] Define `RegistryError` base exception
- [ ] Define `ComponentNotFoundError`
- [ ] Define `ComponentValidationError`
- [ ] Define `DuplicateRegistrationError`
- [ ] Define `DependencyError`
- [ ] Add docstrings with usage examples

**1.1.4: Create core __init__.py** (30 minutes)
File: `core/__init__.py`

- [ ] Export `BaseRegistry`
- [ ] Export `RegistryEntry`
- [ ] Export `RegistryMixin`
- [ ] Export all exceptions
- [ ] Add module docstring

**1.1.5: Write base registry tests** (1 day)
File: `tests/test_registry_base.py`

- [ ] Test component registration
- [ ] Test duplicate registration warning
- [ ] Test override functionality
- [ ] Test component retrieval
- [ ] Test component not found error
- [ ] Test listing and filtering
- [ ] Test metadata storage and retrieval
- [ ] Test tag filtering
- [ ] Achieve 100% coverage on BaseRegistry
- [ ] Run tests and verify all pass

**Deliverables:**
- [x] Core directory structure
- [ ] `core/registry.py` with BaseRegistry (fully tested)
- [ ] `core/exceptions.py` with custom exceptions
- [ ] `tests/test_registry_base.py` with 100% coverage
- [ ] All tests passing

---

### Milestone 1.2: Migrate Losses Registry
**Duration:** 4 days

#### Tasks

**1.2.1: Refactor existing losses.py** (1 day)
File: `registries/losses.py`

- [ ] Move `losses.py` to `registries/losses.py`
- [ ] Update `Losses` class to inherit from `BaseRegistry`
- [ ] Remove duplicate functionality (now in BaseRegistry)
- [ ] Keep all existing loss functions
- [ ] Update decorators to use new BaseRegistry API
- [ ] Add metadata (tags, descriptions, versions) to all losses
- [ ] Update imports within the file
- [ ] Verify all loss functions still work

**1.2.2: Update Losses class implementation** (4 hours)
File: `registries/losses.py`

- [ ] Implement `validate_component()` method
  - Validate loss function signature
  - Check for required parameters
- [ ] Add tags to existing losses:
  - `focal_loss`: ["classification", "imbalanced"]
  - `dice_loss`: ["classification", "overlap"]
  - `combined_direction_loss`: ["classification", "direction"]
  - `point_huber`: ["regression", "robust"]
  - `local_trend_loss`: ["trend", "local"]
  - `extended_trend_loss`: ["trend", "extended"]
  - `custom_loss`: ["composite", "default"]
- [ ] Add descriptions from docstrings
- [ ] Ensure backward compatibility

**1.2.3: Create backward compatibility layer** (2 hours)
File: `losses.py` (root level)

- [ ] Create import shim in root directory
- [ ] Import and re-export everything from `registries/losses`
- [ ] Add deprecation warning (optional)
- [ ] Verify existing code still works

```python
# losses.py (root level)
"""
Backward compatibility shim.
Import from registries.losses for new code.
"""
from registries.losses import (
    Losses,
    focal_loss,
    dice_loss,
    combined_direction_loss,
    compute_dynamic_alpha,
    point_huber,
    local_trend_loss,
    extended_trend_loss,
    custom_loss,
    # ... all exports
)

__all__ = [
    "Losses",
    "focal_loss",
    # ... all exports
]
```

**1.2.4: Update tests** (3 hours)
File: `tests/test_losses.py`

- [ ] Move to `tests/registries/test_losses.py`
- [ ] Update import paths
- [ ] Add tests for new BaseRegistry methods
- [ ] Test metadata retrieval
- [ ] Test filtering by tags
- [ ] Ensure all existing tests still pass
- [ ] Add integration test with BaseRegistry

**1.2.5: Update model.py imports** (1 hour)
File: `model.py`

- [ ] Update import from `losses` to `registries.losses`
- [ ] Verify delegation pattern still works
- [ ] Run full training test to ensure compatibility
- [ ] Check that loss computation is unchanged

**1.2.6: Verify backward compatibility** (2 hours)

- [ ] Run all existing tests
- [ ] Test all three notebooks (diagnostics, inference, trade)
- [ ] Verify no breaking changes
- [ ] Check import times (should be similar)
- [ ] Document any edge cases

**Deliverables:**
- [ ] `registries/losses.py` inheriting from BaseRegistry
- [ ] `losses.py` backward compatibility shim
- [ ] `tests/registries/test_losses.py` with updated tests
- [ ] All tests passing (old and new)
- [ ] No breaking changes to existing code
- [ ] Updated imports in `model.py`

---

### Milestone 1.3: Registry Auto-Discovery System
**Duration:** 2 days

#### Tasks

**1.3.1: Implement registries __init__.py** (4 hours)
File: `registries/__init__.py`

- [ ] Create auto-discovery function
- [ ] Implement module scanning
- [ ] Import all registry modules
- [ ] Export all registry classes
- [ ] Add `print_registry_summary()` function
- [ ] Add environment variable control for summary
- [ ] Handle import errors gracefully

**1.3.2: Create plugin loader** (1 day)
File: `core/plugin_loader.py`

- [ ] Implement `load_plugins()` function
- [ ] Add plugin directory scanning
- [ ] Implement safe module loading
- [ ] Add error handling for failed plugins
- [ ] Log loaded plugins
- [ ] Document plugin structure
- [ ] Create example plugin template

**1.3.3: Create plugins directory structure** (1 hour)

- [ ] Create `plugins/` directory
- [ ] Create `plugins/README.md` with guidelines
- [ ] Create `plugins/example_plugin.py` template
- [ ] Add `.gitignore` for local plugins

**1.3.4: Test auto-discovery** (3 hours)
File: `tests/test_auto_discovery.py`

- [ ] Test registries module import
- [ ] Test that Losses is available
- [ ] Test `print_registry_summary()` function
- [ ] Test plugin loading
- [ ] Test plugin error handling
- [ ] Test plugin component registration

**Deliverables:**
- [ ] `registries/__init__.py` with auto-discovery
- [ ] `core/plugin_loader.py` for external plugins
- [ ] `plugins/` directory with documentation
- [ ] `tests/test_auto_discovery.py`
- [ ] All auto-discovery tests passing

---

### Milestone 1.4: Enhanced Configuration System
**Duration:** 2 days

#### Tasks

**1.4.1: Update Config dataclass** (4 hours)
File: `core/config.py`

- [ ] Move Config from `model.py` to `core/config.py`
- [ ] Add registry component selection fields:
  - `MODEL_NAME: str = "gru_attention"`
  - `OPTIMIZER_NAME: str = "adam"`
  - `LOSS_NAME: str = "custom_loss"`
  - `METRICS: List[str] = [...]`
  - `CALLBACKS: List[str] = [...]`
  - `DATA_LOADER: str = "csv"`
  - `VISUALIZATION: str = "plotly_interactive"`
- [ ] Preserve all existing configuration fields
- [ ] Add `from_yaml()` class method
- [ ] Add `to_yaml()` method
- [ ] Add `_flatten_dict()` helper
- [ ] Add comprehensive docstrings

**1.4.2: Create example config.yaml** (2 hours)
File: `config.yaml`

- [ ] Create comprehensive YAML configuration
- [ ] Include all component selections
- [ ] Include model architecture parameters
- [ ] Include optimizer parameters
- [ ] Include loss weights
- [ ] Include training parameters
- [ ] Add comments explaining each section
- [ ] Validate YAML syntax

**1.4.3: Implement YAML loading/saving** (3 hours)
File: `core/config.py`

- [ ] Install `pyyaml` dependency
- [ ] Implement nested dict flattening
- [ ] Implement nested dict creation
- [ ] Add validation for loaded configs
- [ ] Handle missing fields with defaults
- [ ] Test with example `config.yaml`

**1.4.4: Create config backward compatibility** (1 hour)
File: `model.py`

- [ ] Keep Config class definition (deprecated)
- [ ] Import Config from `core.config`
- [ ] Add deprecation notice in docstring
- [ ] Verify existing code works

**1.4.5: Test configuration system** (2 hours)
File: `tests/test_config.py`

- [ ] Test Config dataclass creation
- [ ] Test `from_yaml()` loading
- [ ] Test `to_yaml()` saving
- [ ] Test round-trip (save → load)
- [ ] Test default values
- [ ] Test invalid YAML handling
- [ ] Test missing fields

**Deliverables:**
- [ ] `core/config.py` with enhanced Config class
- [ ] `config.yaml` example configuration
- [ ] YAML loading/saving functionality
- [ ] `tests/test_config.py` with full coverage
- [ ] Backward compatibility maintained

---

### Phase 1 Checklist

**Infrastructure:**
- [ ] Directory structure created
- [ ] BaseRegistry implemented and tested
- [ ] Custom exceptions defined
- [ ] Core module exports configured

**Losses Migration:**
- [ ] Losses registry refactored to use BaseRegistry
- [ ] Backward compatibility maintained
- [ ] All tests updated and passing
- [ ] Imports updated in main code

**Auto-Discovery:**
- [ ] Registry auto-discovery working
- [ ] Plugin system implemented
- [ ] Example plugin created
- [ ] Documentation written

**Configuration:**
- [ ] Config class enhanced with registry fields
- [ ] YAML support implemented
- [ ] Example config.yaml created
- [ ] Tests passing

**Validation:**
- [ ] All existing tests pass
- [ ] No breaking changes
- [ ] Notebooks still work
- [ ] Import time acceptable (<3 seconds)
- [ ] Code coverage ≥95%

**Documentation:**
- [ ] README updated with new structure
- [ ] Migration notes documented
- [ ] Plugin guidelines written
- [ ] Configuration examples provided

---

## PHASE 2: Core Registries
**Duration:** 4 weeks (staggered)
**Goal:** Implement essential component registries

### Milestone 2.1: Models Registry
**Duration:** 1.5 weeks

#### Tasks

**2.1.1: Extract model architecture** (1 day)
File: `models/gru_attention.py`

- [ ] Create `models/base.py` with base interfaces
- [ ] Extract `PricePredictor.build_model()` to standalone function
- [ ] Create `build_gru_attention_model(config: Config) -> tf.keras.Model`
- [ ] Preserve all functionality exactly
- [ ] Extract `LearnableIndicators` class
- [ ] Extract `PositionalEncodingLayer` class
- [ ] Ensure model builds identically
- [ ] Add comprehensive docstrings

**2.1.2: Create Models registry** (1 day)
File: `registries/models.py`

- [ ] Create `Models` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature has `config` parameter
  - Check return type is `tf.keras.Model`
  - Validate model has inputs and outputs
- [ ] Register `gru_attention` model with metadata:
  - Description: "Bi-GRU with multi-head attention and transformer blocks"
  - Tags: ["rnn", "attention", "transformer", "default"]
  - Version: "2.0.0"
- [ ] Add model validation helper
- [ ] Test model retrieval and building

**2.1.3: Implement alternative models** (2 days)
Files: `models/lstm_transformer.py`, `models/conv_net.py`

- [ ] Design and implement `build_lstm_transformer_model()`
  - LSTM encoder instead of GRU
  - Pure transformer decoder
  - Same input/output interface
  - Register with tags: ["rnn", "transformer"]

- [ ] Design and implement `build_conv1d_model()`
  - CNN-based architecture
  - Faster inference
  - Same input/output interface
  - Register with tags: ["cnn", "fast"]

- [ ] Ensure all models have same interface
- [ ] Test that all models build successfully
- [ ] Document architecture differences

**2.1.4: Integrate with training code** (1 day)
File: `model.py`

- [ ] Update `train_and_evaluate()` to use Models registry
- [ ] Replace direct `PricePredictor` usage with:
  ```python
  model_builder = Models.get(config.MODEL_NAME)
  model = model_builder(config)
  ```
- [ ] Update `CustomTrainModel` wrapper to work with any model
- [ ] Test training with all registered models
- [ ] Verify backward compatibility

**2.1.5: Test Models registry** (1 day)
File: `tests/registries/test_models.py`

- [ ] Test model registration
- [ ] Test model retrieval
- [ ] Test model building with config
- [ ] Test model validation
- [ ] Test input/output shapes
- [ ] Test all registered models build successfully
- [ ] Test filtering by tags
- [ ] Integration test: train 1 epoch with each model

**2.1.6: Update configuration** (2 hours)

- [ ] Add `MODEL_NAME` to Config defaults
- [ ] Update `config.yaml` with model selection
- [ ] Document available models
- [ ] Test config-driven model selection

**Deliverables:**
- [ ] `models/base.py` with base interfaces
- [ ] `models/gru_attention.py` with extracted default model
- [ ] `models/lstm_transformer.py` with alternative model
- [ ] `models/conv_net.py` with CNN model
- [ ] `registries/models.py` with Models registry
- [ ] Integration with training code
- [ ] `tests/registries/test_models.py` with full coverage
- [ ] Updated configuration
- [ ] Documentation

---

### Milestone 2.2: Optimizers Registry
**Duration:** 1 week

#### Tasks

**2.2.1: Create Optimizers registry** (3 hours)
File: `registries/optimizers.py`

- [ ] Create `Optimizers` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature has `config` parameter
  - Check return type is `tf.keras.optimizers.Optimizer`
- [ ] Add comprehensive docstrings

**2.2.2: Register standard optimizers** (1 day)

- [ ] Implement and register `build_adam()`:
  - Tags: ["adaptive", "default"]
  - Support learning rate, betas, epsilon
  - Support gradient clipping

- [ ] Implement and register `build_adamw()`:
  - Tags: ["adaptive", "weight_decay"]
  - Support weight decay
  - Support gradient clipping

- [ ] Implement and register `build_sgd_momentum()`:
  - Tags: ["sgd", "momentum"]
  - Support momentum
  - Support Nesterov acceleration

- [ ] Implement and register `build_rmsprop()`:
  - Tags: ["adaptive"]
  - Support decay, momentum

- [ ] Add Config fields for optimizer-specific parameters

**2.2.3: Integrate with training code** (4 hours)
File: `model.py`

- [ ] Update `train_and_evaluate()` to use Optimizers registry
- [ ] Replace hardcoded optimizer with:
  ```python
  optimizer = Optimizers.get(config.OPTIMIZER_NAME, config=config)
  ```
- [ ] Test training with all optimizers
- [ ] Verify convergence behavior

**2.2.4: Add learning rate schedules** (1 day)
File: `registries/optimizers.py`

- [ ] Implement LR schedule builders:
  - `build_adam_with_cosine_decay()`
  - `build_adam_with_exponential_decay()`
  - `build_adam_with_warmup()`
- [ ] Register with appropriate tags
- [ ] Add schedule parameters to Config
- [ ] Test schedule application

**2.2.5: Test Optimizers registry** (4 hours)
File: `tests/registries/test_optimizers.py`

- [ ] Test optimizer registration
- [ ] Test optimizer retrieval
- [ ] Test optimizer building with config
- [ ] Test all optimizers instantiate correctly
- [ ] Test gradient clipping
- [ ] Test LR schedules
- [ ] Integration test: train 10 steps with each optimizer

**2.2.6: Update configuration** (1 hour)

- [ ] Add `OPTIMIZER_NAME` to Config defaults
- [ ] Update `config.yaml` with optimizer selection
- [ ] Document optimizer options
- [ ] Add optimizer-specific parameters

**Deliverables:**
- [ ] `registries/optimizers.py` with Optimizers registry
- [ ] At least 4 optimizers registered
- [ ] LR schedule support
- [ ] Integration with training code
- [ ] `tests/registries/test_optimizers.py` with full coverage
- [ ] Updated configuration
- [ ] Documentation

---

### Milestone 2.3: Metrics Registry
**Duration:** 1.5 weeks

#### Tasks

**2.3.1: Create Metrics registry** (3 hours)
File: `registries/metrics.py`

- [ ] Create `Metrics` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature has `y_true`, `y_pred` parameters
  - Check return type is numeric
- [ ] Add helper for batch metric computation

**2.3.2: Migrate existing metrics** (1 day)
File: `registries/metrics.py`

- [ ] Import functions from `metrics_utils.py`
- [ ] Register `safe_mape` with tags: ["regression", "percentage"]
- [ ] Register `smape` with tags: ["regression", "percentage"]
- [ ] Register `wape` with tags: ["regression", "percentage", "weighted"]
- [ ] Add any other functions from `metrics_utils.py`
- [ ] Preserve all functionality

**2.3.3: Extract metrics from model.py** (1 day)
File: `registries/metrics.py`

- [ ] Extract metric computation from `_compute_all_horizon_metrics()`:
  - `mse` - Tags: ["regression", "loss"]
  - `rmse` - Tags: ["regression"]
  - `mae` - Tags: ["regression"]
  - `direction_accuracy` - Tags: ["classification", "direction"]
  - `direction_f1` - Tags: ["classification", "direction"]
  - `mcc` - Tags: ["classification", "correlation"]
  - `sensitivity` - Tags: ["classification"]
  - `specificity` - Tags: ["classification"]
  - `calibration_error` - Tags: ["calibration"]

- [ ] Create standalone functions for each metric
- [ ] Register with appropriate metadata
- [ ] Ensure each handles edge cases (NaN, division by zero)

**2.3.4: Implement additional metrics** (1 day)

- [ ] Implement and register `r2_score`:
  - Tags: ["regression", "correlation"]

- [ ] Implement and register `max_error`:
  - Tags: ["regression", "worst_case"]

- [ ] Implement and register `explained_variance`:
  - Tags: ["regression", "variance"]

- [ ] Implement and register `sharpe_ratio`:
  - Tags: ["trading", "performance"]

- [ ] Implement and register `max_drawdown`:
  - Tags: ["trading", "risk"]

**2.3.5: Integrate with evaluation code** (1 day)
File: `model.py`

- [ ] Update `_compute_all_horizon_metrics()` to use registry
- [ ] Make metric selection configurable via Config
- [ ] Compute only requested metrics
- [ ] Return dict with metric names and values
- [ ] Test that evaluation produces same results

**2.3.6: Create metric groups** (4 hours)
File: `registries/metrics.py`

- [ ] Define metric groups:
  - `REGRESSION_METRICS = ["mse", "rmse", "mae", "safe_mape"]`
  - `DIRECTION_METRICS = ["direction_accuracy", "direction_f1", "mcc"]`
  - `TRADING_METRICS = ["sharpe_ratio", "max_drawdown"]`
  - `ALL_METRICS = [...]`

- [ ] Add helper function to compute metric groups
- [ ] Update Config to support metric groups
- [ ] Test group computation

**2.3.7: Test Metrics registry** (1 day)
File: `tests/registries/test_metrics.py`

- [ ] Test metric registration
- [ ] Test metric retrieval
- [ ] Test metric computation with dummy data
- [ ] Test all metrics handle edge cases:
  - Empty arrays
  - Arrays with zeros
  - Arrays with NaN
  - Perfect predictions
  - Worst predictions
- [ ] Test metric groups
- [ ] Integration test: compute all metrics on validation data

**2.3.8: Update configuration and docs** (2 hours)

- [ ] Add `METRICS` list to Config
- [ ] Update `config.yaml` with metric selection
- [ ] Document all available metrics
- [ ] Provide metric selection examples
- [ ] Create metrics comparison table

**Deliverables:**
- [ ] `registries/metrics.py` with Metrics registry
- [ ] At least 15 metrics registered
- [ ] Metric groups defined
- [ ] Integration with evaluation code
- [ ] `tests/registries/test_metrics.py` with full coverage
- [ ] Updated configuration
- [ ] Comprehensive documentation

---

### Phase 2 Checklist

**Models Registry:**
- [ ] At least 3 model architectures registered
- [ ] All models have same input/output interface
- [ ] Models integrate with training pipeline
- [ ] Config-driven model selection works
- [ ] Tests passing with ≥95% coverage

**Optimizers Registry:**
- [ ] At least 4 optimizers registered
- [ ] LR schedules supported
- [ ] Gradient clipping works
- [ ] Config-driven optimizer selection works
- [ ] Tests passing with ≥95% coverage

**Metrics Registry:**
- [ ] At least 15 metrics registered
- [ ] Metric groups defined
- [ ] Integration with evaluation code
- [ ] Config-driven metric selection works
- [ ] Tests passing with ≥95% coverage

**Integration:**
- [ ] All registries work together
- [ ] End-to-end training works
- [ ] Config-driven component selection functional
- [ ] No breaking changes
- [ ] Performance acceptable (no significant slowdown)

**Documentation:**
- [ ] All new components documented
- [ ] Usage examples provided
- [ ] Configuration guide updated
- [ ] Architecture diagrams created

---

## PHASE 3: Data & Visualization Registries
**Duration:** 2 weeks
**Goal:** Modularize data pipeline and visualization

### Milestone 3.1: Data Loaders Registry
**Duration:** 1 week

#### Tasks

**3.1.1: Extract DataProcessor** (1 day)
File: `data/processor.py`

- [ ] Move `DataProcessor` class from `model.py` to `data/processor.py`
- [ ] Keep all existing functionality
- [ ] Update imports in `model.py`
- [ ] Test that data loading still works
- [ ] Preserve backward compatibility

**3.1.2: Create DataLoaders registry** (4 hours)
File: `registries/data_loaders.py`

- [ ] Create `DataLoaders` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature has `config` parameter
  - Check return type is `pd.DataFrame`
  - Validate DataFrame has required columns
  - Validate timestamp index
- [ ] Add data validation helpers

**3.1.3: Implement CSV loader** (3 hours)
File: `registries/data_loaders.py`

- [ ] Extract data loading logic from `DataProcessor`
- [ ] Create `load_from_csv(config, file_path=None) -> pd.DataFrame`
- [ ] Register with tags: ["file", "default"]
- [ ] Support configurable file path
- [ ] Add CSV format validation
- [ ] Handle missing files gracefully
- [ ] Test with existing data files

**3.1.4: Implement database loaders** (1 day)
File: `registries/data_loaders.py`

- [ ] Implement `load_from_postgres()`:
  - Tags: ["database", "sql", "postgres"]
  - Use SQLAlchemy
  - Support custom queries
  - Connection string from config

- [ ] Implement `load_from_mysql()`:
  - Tags: ["database", "sql", "mysql"]
  - Similar interface to postgres

- [ ] Add database dependencies to requirements
- [ ] Test with mock databases

**3.1.5: Implement API loaders** (1 day)
File: `registries/data_loaders.py`

- [ ] Implement `load_from_binance()`:
  - Tags: ["api", "exchange", "crypto"]
  - Use ccxt library
  - Support symbol and timeframe params

- [ ] Implement `load_from_alpha_vantage()`:
  - Tags: ["api", "stocks"]
  - Support API key from config

- [ ] Implement `load_from_yahoo_finance()`:
  - Tags: ["api", "stocks", "free"]

- [ ] Add rate limiting
- [ ] Handle API errors
- [ ] Cache API responses

**3.1.6: Integrate with DataProcessor** (4 hours)
File: `data/processor.py`

- [ ] Update `DataProcessor.load_and_prepare_data()` to use registry
- [ ] Add `data_loader` parameter (defaults to config value)
- [ ] Support passing custom loaders
- [ ] Test with all loaders
- [ ] Verify preprocessing still works

**3.1.7: Test DataLoaders registry** (4 hours)
File: `tests/registries/test_data_loaders.py`

- [ ] Test loader registration
- [ ] Test CSV loader with real files
- [ ] Test database loaders with mocks
- [ ] Test API loaders with mocks
- [ ] Test data validation
- [ ] Test error handling (missing files, API errors, etc.)
- [ ] Integration test: load → process → train

**3.1.8: Update configuration** (1 hour)

- [ ] Add `DATA_LOADER` to Config
- [ ] Add loader-specific parameters
- [ ] Update `config.yaml` with examples
- [ ] Document each loader's requirements

**Deliverables:**
- [ ] `data/processor.py` with extracted DataProcessor
- [ ] `registries/data_loaders.py` with DataLoaders registry
- [ ] At least 5 data loaders implemented
- [ ] Integration with DataProcessor
- [ ] `tests/registries/test_data_loaders.py` with full coverage
- [ ] Updated configuration and documentation

---

### Milestone 3.2: Preprocessors Registry
**Duration:** 4 days

#### Tasks

**3.2.1: Create Preprocessors registry** (2 hours)
File: `registries/preprocessors.py`

- [ ] Create `Preprocessors` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature has `df` and `config` parameters
  - Check return type is `pd.DataFrame`
  - Validate shape preservation (or document changes)
- [ ] Add pipeline composition helpers

**3.2.2: Extract scaling preprocessors** (4 hours)
File: `registries/preprocessors.py`

- [ ] Implement `apply_standard_scaler()`:
  - Tags: ["scaling", "default"]
  - Extract from DataProcessor
  - Support fit/transform split

- [ ] Implement `apply_minmax_scaler()`:
  - Tags: ["scaling", "bounded"]

- [ ] Implement `apply_robust_scaler()`:
  - Tags: ["scaling", "outlier_robust"]

- [ ] Register all scalers
- [ ] Test each scaler independently

**3.2.3: Implement transformation preprocessors** (4 hours)

- [ ] Implement `compute_log_returns()`:
  - Tags: ["transform", "finance"]

- [ ] Implement `compute_differences()`:
  - Tags: ["transform", "stationary"]

- [ ] Implement `remove_outliers()`:
  - Tags: ["cleaning", "outliers"]
  - Multiple methods (IQR, z-score, isolation forest)

- [ ] Implement `handle_missing_values()`:
  - Tags: ["cleaning", "imputation"]
  - Multiple strategies (ffill, bfill, interpolate, mean)

**3.2.4: Implement feature engineering preprocessors** (1 day)

- [ ] Implement `add_technical_indicators()`:
  - Tags: ["features", "technical"]
  - Optional TA-Lib integration

- [ ] Implement `add_time_features()`:
  - Tags: ["features", "temporal"]
  - Hour, day of week, month, etc.

- [ ] Implement `add_lag_features()`:
  - Tags: ["features", "lag"]

- [ ] Implement `add_rolling_stats()`:
  - Tags: ["features", "rolling"]

**3.2.5: Create preprocessing pipelines** (4 hours)
File: `registries/preprocessors.py`

- [ ] Implement `PreprocessingPipeline` class:
  - Compose multiple preprocessors
  - Execute in order
  - Support fit/transform split
  - Save/load pipeline state

- [ ] Create predefined pipelines:
  - `default_pipeline` = [scale, clean]
  - `full_pipeline` = [clean, features, scale]
  - `minimal_pipeline` = [scale only]

- [ ] Register pipelines
- [ ] Test pipeline execution

**3.2.6: Integrate with DataProcessor** (3 hours)
File: `data/processor.py`

- [ ] Update `DataProcessor` to use Preprocessors registry
- [ ] Make preprocessing configurable
- [ ] Support custom preprocessing pipelines
- [ ] Test with all preprocessors
- [ ] Ensure results match previous implementation

**3.2.7: Test Preprocessors registry** (4 hours)
File: `tests/registries/test_preprocessors.py`

- [ ] Test preprocessor registration
- [ ] Test each preprocessor independently
- [ ] Test preprocessing pipelines
- [ ] Test fit/transform split
- [ ] Test with various data types
- [ ] Test error handling
- [ ] Integration test: full preprocessing pipeline

**Deliverables:**
- [ ] `registries/preprocessors.py` with Preprocessors registry
- [ ] At least 10 preprocessors registered
- [ ] Preprocessing pipeline system
- [ ] Integration with DataProcessor
- [ ] `tests/registries/test_preprocessors.py` with full coverage
- [ ] Documentation

---

### Milestone 3.3: Visualizations Registry
**Duration:** 3 days

#### Tasks

**3.3.1: Extract visualization code** (3 hours)
File: `visualization/plotly_backend.py`

- [ ] Extract `make_interactive_plot_callback()` from `model.py`
- [ ] Move to `visualization/plotly_backend.py`
- [ ] Keep all functionality intact
- [ ] Update imports in `model.py`
- [ ] Test callback still works

**3.3.2: Create Visualizations registry** (2 hours)
File: `registries/visualizations.py`

- [ ] Create `Visualizations` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check signature accepts data dict and config
  - Return type can be figure, callback, or None
- [ ] Add visualization helpers

**3.3.3: Implement Plotly visualizations** (4 hours)
File: `visualization/plotly_backend.py`

- [ ] Create `create_plotly_training_callback()`:
  - Tags: ["plotly", "interactive", "training", "default"]
  - Current interactive plot functionality

- [ ] Create `create_plotly_predictions_plot()`:
  - Tags: ["plotly", "interactive", "predictions"]
  - Plot predictions vs actuals

- [ ] Create `create_plotly_dashboard()`:
  - Tags: ["plotly", "interactive", "dashboard"]
  - Multi-panel dashboard

- [ ] Register all Plotly visualizations

**3.3.4: Implement Matplotlib visualizations** (1 day)
File: `visualization/matplotlib_backend.py`

- [ ] Create `create_matplotlib_training_plot()`:
  - Tags: ["matplotlib", "static", "training", "publication"]
  - Static publication-quality plots

- [ ] Create `create_matplotlib_predictions_plot()`:
  - Tags: ["matplotlib", "static", "predictions"]

- [ ] Create `create_matplotlib_report()`:
  - Tags: ["matplotlib", "static", "report"]
  - Complete PDF-ready report

- [ ] Register all Matplotlib visualizations

**3.3.5: Implement logging visualizations** (4 hours)

- [ ] Create `log_to_tensorboard()`:
  - Tags: ["tensorboard", "logging"]
  - TensorBoard integration

- [ ] Create `log_to_wandb()`:
  - Tags: ["wandb", "cloud", "logging"]
  - Weights & Biases integration

- [ ] Create `log_to_mlflow()`:
  - Tags: ["mlflow", "logging"]
  - MLflow integration

- [ ] Register all logging visualizations

**3.3.6: Integrate with callbacks** (3 hours)
File: `model.py`

- [ ] Update callback creation to use Visualizations registry
- [ ] Make visualization backend configurable
- [ ] Support multiple simultaneous visualizations
- [ ] Test all visualization backends

**3.3.7: Test Visualizations registry** (4 hours)
File: `tests/registries/test_visualizations.py`

- [ ] Test visualization registration
- [ ] Test each visualization with dummy data
- [ ] Test callback creation
- [ ] Test multiple backends simultaneously
- [ ] Test with actual training data
- [ ] Test error handling

**Deliverables:**
- [ ] `visualization/plotly_backend.py` with Plotly visualizations
- [ ] `visualization/matplotlib_backend.py` with Matplotlib visualizations
- [ ] `registries/visualizations.py` with Visualizations registry
- [ ] At least 8 visualizations registered
- [ ] Integration with training callbacks
- [ ] `tests/registries/test_visualizations.py` with full coverage
- [ ] Documentation

---

### Phase 3 Checklist

**Data Loaders:**
- [ ] At least 5 data loaders implemented
- [ ] CSV, database, and API loaders working
- [ ] Data validation implemented
- [ ] Integration with DataProcessor complete
- [ ] Tests passing

**Preprocessors:**
- [ ] At least 10 preprocessors implemented
- [ ] Preprocessing pipeline system working
- [ ] Fit/transform split supported
- [ ] Integration with DataProcessor complete
- [ ] Tests passing

**Visualizations:**
- [ ] At least 8 visualizations implemented
- [ ] Multiple backends supported (Plotly, Matplotlib)
- [ ] Logging integrations working (TensorBoard, W&B)
- [ ] Integration with training callbacks complete
- [ ] Tests passing

**Integration:**
- [ ] Complete data pipeline configurable
- [ ] Visualization backend selectable
- [ ] End-to-end training works with all options
- [ ] Performance acceptable

**Documentation:**
- [ ] All loaders documented
- [ ] All preprocessors documented
- [ ] All visualizations documented
- [ ] Usage examples provided

---

## PHASE 4: Advanced Features & Polish
**Duration:** 2 weeks
**Goal:** Implement callbacks/layers registries and plugin system

### Milestone 4.1: Callbacks Registry
**Duration:** 4 days

#### Tasks

**4.1.1: Create Callbacks registry** (2 hours)
File: `registries/callbacks.py`

- [ ] Create `Callbacks` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check return type is Callback or list of Callbacks
  - Validate Callback interface
- [ ] Add callback composition helpers

**4.1.2: Extract and register custom callbacks** (1 day)
File: `registries/callbacks.py`

- [ ] Extract `TqdmCallback` from `model.py`
- [ ] Register as `tqdm_progress`:
  - Tags: ["progress", "console", "default"]

- [ ] Extract `ParamsLogger` from `model.py`
- [ ] Register as `params_logger`:
  - Tags: ["logging", "mlflow"]

- [ ] Extract interactive plot callback
- [ ] Register as `interactive_plot`:
  - Tags: ["visualization", "plotly", "default"]

**4.1.3: Implement callback builders** (1 day)

- [ ] Create `build_early_stopping()`:
  - Tags: ["regularization", "training"]
  - Configurable monitor and patience

- [ ] Create `build_model_checkpoint()`:
  - Tags: ["persistence", "training"]
  - Configurable save path and monitor

- [ ] Create `build_reduce_lr_on_plateau()`:
  - Tags: ["optimization", "training"]
  - Configurable patience and factor

- [ ] Create `build_csv_logger()`:
  - Tags: ["logging", "training"]

**4.1.4: Implement multi-monitor callbacks** (4 hours)

- [ ] Create `build_multi_monitor_early_stopping()`:
  - Tags: ["regularization", "ensemble"]
  - Stop on multiple conditions
  - Currently used in training

- [ ] Create `build_metric_threshold_callback()`:
  - Tags: ["regularization", "threshold"]
  - Stop when metric reaches threshold

**4.1.5: Integrate with training** (3 hours)
File: `model.py`

- [ ] Update callback creation to use Callbacks registry
- [ ] Support callback list from config
- [ ] Handle callbacks that return lists
- [ ] Test all callbacks work correctly

**4.1.6: Test Callbacks registry** (4 hours)
File: `tests/registries/test_callbacks.py`

- [ ] Test callback registration
- [ ] Test callback retrieval
- [ ] Test each callback independently
- [ ] Test callback with mock training
- [ ] Integration test: full training with all callbacks

**Deliverables:**
- [ ] `registries/callbacks.py` with Callbacks registry
- [ ] At least 8 callbacks registered
- [ ] Custom callbacks extracted from model.py
- [ ] Integration with training
- [ ] Tests passing
- [ ] Documentation

---

### Milestone 4.2: Layers Registry
**Duration:** 3 days

#### Tasks

**4.2.1: Create Layers registry** (2 hours)
File: `registries/layers.py`

- [ ] Create `Layers` class inheriting from `BaseRegistry`
- [ ] Implement `validate_component()`:
  - Check return type is Layer class or instance
  - Validate layer interface
- [ ] Add layer builder helpers

**4.2.2: Extract and register existing layers** (4 hours)

- [ ] Extract `LearnableIndicators` from `model.py`
- [ ] Move to `models/layers/learnable_indicators.py`
- [ ] Register as `learnable_indicators`:
  - Tags: ["indicators", "meta_learning", "default"]

- [ ] Extract `PositionalEncodingLayer` from `model.py`
- [ ] Move to `models/layers/positional_encoding.py`
- [ ] Register as `positional_encoding`:
  - Tags: ["attention", "transformer", "default"]

**4.2.3: Implement additional layers** (1 day)

- [ ] Implement `SqueezeExcitationLayer`:
  - Tags: ["attention", "channel"]
  - Channel-wise attention

- [ ] Implement `WaveNetCausalLayer`:
  - Tags: ["cnn", "causal", "time_series"]
  - Dilated causal convolution

- [ ] Implement `MultiScaleConvLayer`:
  - Tags: ["cnn", "multi_scale"]
  - Parallel convolutions at different scales

- [ ] Register all new layers

**4.2.4: Make layers swappable in models** (4 hours)

- [ ] Update model builders to use Layers registry
- [ ] Add layer selection to Config
- [ ] Support custom layer configurations
- [ ] Test models with different layer combinations

**4.2.5: Test Layers registry** (3 hours)
File: `tests/registries/test_layers.py`

- [ ] Test layer registration
- [ ] Test layer retrieval
- [ ] Test layer instantiation
- [ ] Test layer in model building
- [ ] Test layer serialization
- [ ] Test custom layer configurations

**Deliverables:**
- [ ] `registries/layers.py` with Layers registry
- [ ] Existing layers extracted and registered
- [ ] At least 5 layers registered
- [ ] Integration with model builders
- [ ] Tests passing
- [ ] Documentation

---

### Milestone 4.3: Plugin System Enhancement
**Duration:** 3 days

#### Tasks

**4.3.1: Enhance plugin loader** (4 hours)
File: `core/plugin_loader.py`

- [ ] Add plugin dependency checking
- [ ] Implement plugin versioning
- [ ] Add plugin conflict detection
- [ ] Implement plugin priority/ordering
- [ ] Add plugin enable/disable functionality
- [ ] Improve error messages

**4.3.2: Create plugin templates** (4 hours)
Directory: `plugins/templates/`

- [ ] Create `model_plugin_template.py`:
  - Shows how to register custom models
  - Includes documentation

- [ ] Create `loss_plugin_template.py`:
  - Shows how to register custom losses

- [ ] Create `metric_plugin_template.py`:
  - Shows how to register custom metrics

- [ ] Create comprehensive plugin development guide

**4.3.3: Implement example plugins** (1 day)
Directory: `plugins/examples/`

- [ ] Create `wavenet_plugin.py`:
  - Implements WaveNet model architecture
  - Shows full plugin structure

- [ ] Create `trading_metrics_plugin.py`:
  - Implements trading-specific metrics
  - Sharpe ratio, Sortino ratio, Calmar ratio

- [ ] Create `custom_callbacks_plugin.py`:
  - Implements specialized callbacks
  - Telegram notifications, email alerts

**4.3.4: Add plugin CLI** (4 hours)
File: `scripts/plugin_cli.py`

- [ ] Implement `list` command:
  - Show all available plugins
  - Show loaded/unloaded status

- [ ] Implement `info` command:
  - Show plugin details
  - Show registered components

- [ ] Implement `enable/disable` commands:
  - Enable/disable plugins

- [ ] Implement `validate` command:
  - Check plugin validity

**4.3.5: Test plugin system** (4 hours)
File: `tests/test_plugins.py`

- [ ] Test plugin loading
- [ ] Test plugin dependency checking
- [ ] Test plugin conflict detection
- [ ] Test plugin enable/disable
- [ ] Test example plugins
- [ ] Test plugin error handling

**Deliverables:**
- [ ] Enhanced plugin loader
- [ ] Plugin templates created
- [ ] Example plugins implemented
- [ ] Plugin CLI tool
- [ ] Comprehensive plugin documentation
- [ ] Tests passing

---

### Phase 4 Checklist

**Callbacks Registry:**
- [ ] At least 8 callbacks registered
- [ ] Custom callbacks extracted
- [ ] Integration complete
- [ ] Tests passing

**Layers Registry:**
- [ ] At least 5 layers registered
- [ ] Layers swappable in models
- [ ] Integration complete
- [ ] Tests passing

**Plugin System:**
- [ ] Enhanced plugin loader
- [ ] Plugin templates available
- [ ] Example plugins working
- [ ] Plugin CLI functional
- [ ] Documentation complete

**Integration:**
- [ ] All registries working together
- [ ] Plugins can extend all registries
- [ ] Configuration fully supports all features
- [ ] End-to-end tests passing

---

## PHASE 5: Refinement & Documentation
**Duration:** 2 weeks
**Goal:** Polish, optimize, and document the complete system

### Milestone 5.1: Code Refactoring
**Duration:** 1 week

#### Tasks

**5.1.1: Refactor model.py** (2 days)

- [ ] Extract remaining components to appropriate modules
- [ ] Reduce model.py to <1000 lines (currently 3010 lines)
- [ ] Move `DataProcessor` to `data/processor.py` (already planned)
- [ ] Move `CustomTrainModel` to `training/custom_model.py`
- [ ] Move training logic to `training/trainer.py`
- [ ] Keep only main entry points in model.py
- [ ] Update all imports
- [ ] Test that everything still works

**5.1.2: Organize model architecture code** (1 day)

- [ ] Create `models/` module structure:
  ```
  models/
  ├── __init__.py
  ├── base.py              # Base classes
  ├── gru_attention.py     # Default model
  ├── lstm_transformer.py  # Alternative model
  ├── conv_net.py          # CNN model
  └── layers/
      ├── __init__.py
      ├── learnable_indicators.py
      ├── positional_encoding.py
      └── attention.py
  ```
- [ ] Ensure clean imports
- [ ] Update registries to use new locations
- [ ] Test all models still build correctly

**5.1.3: Organize data pipeline code** (1 day)

- [ ] Create `data/` module structure:
  ```
  data/
  ├── __init__.py
  ├── processor.py         # Main DataProcessor
  ├── loaders/
  │   ├── __init__.py
  │   ├── csv_loader.py
  │   ├── database_loader.py
  │   └── api_loader.py
  └── preprocessors/
      ├── __init__.py
      ├── scalers.py
      ├── cleaners.py
      └── features.py
  ```
- [ ] Move code to appropriate modules
- [ ] Update registry implementations
- [ ] Test data pipeline

**5.1.4: Create training module** (1 day)
Directory: `training/`

- [ ] Create `training/__init__.py`
- [ ] Create `training/trainer.py`:
  - Main `train_and_evaluate()` function
  - Training orchestration logic

- [ ] Create `training/custom_model.py`:
  - `CustomTrainModel` class
  - Custom training step logic

- [ ] Create `training/callbacks.py`:
  - Callback creation helpers

- [ ] Update imports throughout codebase

**5.1.5: Remove backward compatibility shims** (1 day)

- [ ] Remove root-level `losses.py` import shim
- [ ] Update all import paths to use `registries.*`
- [ ] Remove deprecated Config location
- [ ] Update notebooks with new imports
- [ ] Test that all code uses new structure

**Deliverables:**
- [ ] `model.py` reduced to <1000 lines
- [ ] Code organized into logical modules
- [ ] Clean module structure
- [ ] All imports updated
- [ ] All tests passing

---

### Milestone 5.2: Performance Optimization
**Duration:** 3 days

#### Tasks

**5.2.1: Profile import times** (4 hours)

- [ ] Profile current import time
- [ ] Identify slow imports
- [ ] Implement lazy loading where appropriate
- [ ] Cache registry lookups
- [ ] Measure improvement
- [ ] Target: <2 seconds total import time

**5.2.2: Optimize registry lookups** (4 hours)

- [ ] Profile registry.get() calls
- [ ] Add result caching for repeated lookups
- [ ] Optimize validation checks
- [ ] Benchmark performance impact
- [ ] Ensure no regression in training speed

**5.2.3: Optimize auto-discovery** (3 hours)

- [ ] Make auto-discovery optional (env var)
- [ ] Implement selective registration
- [ ] Add registration caching
- [ ] Test different scenarios
- [ ] Document performance characteristics

**5.2.4: Memory profiling** (4 hours)

- [ ] Profile memory usage
- [ ] Identify memory leaks
- [ ] Optimize large data structures
- [ ] Test with long training runs
- [ ] Document memory requirements

**5.2.5: Create performance benchmarks** (1 day)
File: `scripts/benchmark.py`

- [ ] Benchmark import time
- [ ] Benchmark registry lookup time
- [ ] Benchmark model building time
- [ ] Benchmark training overhead
- [ ] Create performance baseline
- [ ] Set up regression testing

**Deliverables:**
- [ ] Import time <2 seconds
- [ ] No measurable training overhead
- [ ] Memory usage optimized
- [ ] Performance benchmarks created
- [ ] Regression tests in place

---

### Milestone 5.3: Testing & Quality Assurance
**Duration:** 2 days

#### Tasks

**5.3.1: Achieve test coverage goals** (1 day)

- [ ] Run coverage report
- [ ] Identify uncovered code
- [ ] Write tests for uncovered areas
- [ ] Achieve ≥95% coverage on core modules
- [ ] Achieve ≥90% coverage overall
- [ ] Document coverage metrics

**5.3.2: Integration testing** (4 hours)
File: `tests/test_end_to_end.py`

- [ ] Test complete training pipeline
- [ ] Test all component combinations
- [ ] Test config-driven workflows
- [ ] Test plugin loading and usage
- [ ] Test error scenarios
- [ ] Test with real data

**5.3.3: Notebook testing** (4 hours)

- [ ] Update all notebooks with new imports
- [ ] Test diagnostics.ipynb
- [ ] Test inference.ipynb
- [ ] Test trade.ipynb
- [ ] Verify all notebooks run end-to-end
- [ ] Add notebook CI testing

**5.3.4: Create test utilities** (3 hours)
File: `tests/utils.py`

- [ ] Create mock data generators
- [ ] Create fixture factories
- [ ] Create assertion helpers
- [ ] Create registry test helpers
- [ ] Document test utilities

**Deliverables:**
- [ ] ≥95% coverage on core modules
- [ ] ≥90% coverage overall
- [ ] Comprehensive integration tests
- [ ] All notebooks tested and working
- [ ] Test utilities documented

---

### Milestone 5.4: Documentation
**Duration:** 3 days

#### Tasks

**5.4.1: Update README** (4 hours)
File: `README.md`

- [ ] Add overview of registry system
- [ ] Add quick start guide
- [ ] Add installation instructions
- [ ] Add usage examples
- [ ] Add architecture diagram
- [ ] Add links to detailed docs

**5.4.2: Create user guide** (1 day)
File: `docs/USER_GUIDE.md`

- [ ] Configuration guide
- [ ] Component selection guide
- [ ] Custom component creation guide
- [ ] Plugin development guide
- [ ] Troubleshooting section
- [ ] FAQ section

**5.4.3: Create API documentation** (1 day)
Directory: `docs/api/`

- [ ] Document BaseRegistry API
- [ ] Document each registry (Models, Optimizers, etc.)
- [ ] Document Config class
- [ ] Document plugin system
- [ ] Generate API docs with Sphinx/MkDocs

**5.4.4: Create examples** (4 hours)
Directory: `examples/`

- [ ] Create `basic_training.py`:
  - Simple training example

- [ ] Create `custom_model.py`:
  - How to add custom model

- [ ] Create `custom_loss.py`:
  - How to add custom loss

- [ ] Create `plugin_example/`:
  - Complete plugin example

- [ ] Create `config_examples/`:
  - Various configuration examples

**5.4.5: Create migration guide** (4 hours)
File: `docs/MIGRATION.md`

- [ ] Document changes from old structure
- [ ] Provide step-by-step migration guide
- [ ] List breaking changes
- [ ] Provide code migration examples
- [ ] Add troubleshooting tips

**5.4.6: Create architecture documentation** (4 hours)
File: `docs/ARCHITECTURE.md`

- [ ] Document overall architecture
- [ ] Explain registry pattern
- [ ] Show component relationships
- [ ] Include diagrams
- [ ] Document design decisions

**Deliverables:**
- [ ] README updated
- [ ] Comprehensive user guide
- [ ] Complete API documentation
- [ ] Working examples
- [ ] Migration guide
- [ ] Architecture documentation

---

### Milestone 5.5: Final Polish
**Duration:** 2 days

#### Tasks

**5.5.1: Code quality improvements** (4 hours)

- [ ] Run linter (flake8, pylint)
- [ ] Fix all linting issues
- [ ] Run type checker (mypy)
- [ ] Add missing type hints
- [ ] Format code (black, isort)
- [ ] Add docstrings where missing

**5.5.2: CI/CD setup** (4 hours)
File: `.github/workflows/tests.yml`

- [ ] Set up GitHub Actions
- [ ] Run tests on push
- [ ] Run linting
- [ ] Check code coverage
- [ ] Test on multiple Python versions
- [ ] Generate coverage reports

**5.5.3: Create release checklist** (2 hours)
File: `docs/RELEASE_CHECKLIST.md`

- [ ] List all pre-release tasks
- [ ] Version bumping procedure
- [ ] Testing requirements
- [ ] Documentation requirements
- [ ] Release announcement template

**5.5.4: Prepare v2.0 release** (4 hours)

- [ ] Update version numbers
- [ ] Create CHANGELOG.md
- [ ] Tag release
- [ ] Create release notes
- [ ] Archive old structure (optional branch)

**5.5.5: Final validation** (4 hours)

- [ ] Run full test suite
- [ ] Test all examples
- [ ] Test all notebooks
- [ ] Verify documentation
- [ ] Test on fresh environment
- [ ] Sign off on release

**Deliverables:**
- [ ] Code quality excellent
- [ ] CI/CD pipeline working
- [ ] Release process documented
- [ ] v2.0 ready for release
- [ ] All quality gates passed

---

### Phase 5 Checklist

**Code Organization:**
- [ ] model.py <1000 lines
- [ ] Code in logical modules
- [ ] Clean imports throughout
- [ ] No backward compatibility shims
- [ ] All tests passing

**Performance:**
- [ ] Import time <2 seconds
- [ ] No training overhead
- [ ] Memory optimized
- [ ] Benchmarks created

**Testing:**
- [ ] ≥95% core coverage
- [ ] ≥90% overall coverage
- [ ] Integration tests comprehensive
- [ ] Notebooks tested

**Documentation:**
- [ ] README complete
- [ ] User guide complete
- [ ] API docs complete
- [ ] Examples working
- [ ] Migration guide available
- [ ] Architecture documented

**Release:**
- [ ] Code quality excellent
- [ ] CI/CD working
- [ ] Release process ready
- [ ] All quality gates passed
- [ ] v2.0 tagged and released

---

## Summary & Timeline

### Phase Timeline

| Phase | Duration | Completion |
|-------|----------|------------|
| Phase 1: Foundation | 2 weeks | ░░░░░░░░░░░░░░░░░░░░ 0% |
| Phase 2: Core Registries | 4 weeks | ░░░░░░░░░░░░░░░░░░░░ 0% |
| Phase 3: Data & Visualization | 2 weeks | ░░░░░░░░░░░░░░░░░░░░ 0% |
| Phase 4: Advanced Features | 2 weeks | ░░░░░░░░░░░░░░░░░░░░ 0% |
| Phase 5: Refinement | 2 weeks | ░░░░░░░░░░░░░░░░░░░░ 0% |
| **Total** | **12 weeks** | ░░░░░░░░░░░░░░░░░░░░ 0% |

### Critical Path

1. **Week 1-2:** Foundation (blocking all other work)
2. **Week 3-6:** Core registries (Models → Optimizers → Metrics)
3. **Week 7-8:** Data & visualization (parallel with Phase 4)
4. **Week 9-10:** Advanced features (parallel with Phase 3)
5. **Week 11-12:** Final polish and release

### Success Metrics

**Code Quality:**
- [ ] Lines of code reduced by 30%
- [ ] Test coverage ≥95% (core), ≥90% (overall)
- [ ] Import time <2 seconds
- [ ] Zero performance regression

**Functionality:**
- [ ] All 9 registries implemented and tested
- [ ] 50+ components registered
- [ ] Plugin system functional
- [ ] Config-driven workflow complete

**Documentation:**
- [ ] 5+ documentation files
- [ ] 10+ working examples
- [ ] API docs complete
- [ ] Migration guide available

**Adoption:**
- [ ] Zero breaking changes without migration path
- [ ] All notebooks updated
- [ ] Backward compatibility maintained through Phase 4

---

## Risk Management

### High Priority Risks

**Risk:** Breaking existing functionality during refactoring
**Mitigation:**
- Maintain backward compatibility until Phase 5
- Comprehensive test coverage
- Incremental migration with validation at each step

**Risk:** Performance degradation from registry overhead
**Mitigation:**
- Benchmark at each phase
- Implement caching and lazy loading
- Profile and optimize hot paths

**Risk:** Timeline slippage
**Mitigation:**
- Phase 1 is well-scoped and foundational
- Phases 2-4 can be partially parallelized
- Phase 5 is buffer time

**Risk:** Documentation falling behind implementation
**Mitigation:**
- Document as you implement
- Code examples in each milestone
- Dedicated documentation phase (Phase 5)

### Medium Priority Risks

**Risk:** Plugin system security concerns
**Mitigation:**
- Plugin validation
- Sandboxed execution (future)
- Clear security guidelines

**Risk:** Configuration complexity
**Mitigation:**
- Sensible defaults
- Configuration templates
- Validation and error messages

---

## Next Steps

### Immediate Actions (Week 1)

1. **Create directory structure** (Day 1)
   ```bash
   mkdir -p core registries tests/registries models data visualization training
   touch core/{__init__.py,registry.py,exceptions.py,config.py}
   touch registries/__init__.py
   ```

2. **Implement BaseRegistry** (Days 1-2)
   - Write `core/registry.py`
   - Write `core/exceptions.py`
   - Write `tests/test_registry_base.py`
   - Run tests

3. **Migrate Losses** (Days 3-5)
   - Move `losses.py` to `registries/losses.py`
   - Update to use BaseRegistry
   - Create backward compatibility shim
   - Update and run tests

4. **Implement Auto-Discovery** (Days 6-7)
   - Write `registries/__init__.py`
   - Write `core/plugin_loader.py`
   - Test auto-discovery
   - Create example plugin

5. **Enhance Configuration** (Days 8-10)
   - Move Config to `core/config.py`
   - Add YAML support
   - Create `config.yaml`
   - Test configuration loading

### Weekly Status Updates

Track progress by updating this document:
- Mark completed tasks with [x]
- Update percentage bars
- Note any blockers or changes
- Adjust timeline as needed

---

## Appendix

### A. File Organization Reference

**Current Structure (Before):**
```
neural_trade/
├── model.py                 (3010 lines - monolithic)
├── losses.py                (477 lines)
├── metrics_utils.py         (124 lines)
└── tests/
    └── test_losses.py       (82 lines)
```

**Target Structure (After):**
```
neural_trade/
├── core/
│   ├── registry.py          # Base registry
│   ├── config.py            # Configuration
│   ├── exceptions.py        # Custom exceptions
│   └── plugin_loader.py     # Plugin system
├── registries/
│   ├── __init__.py          # Auto-discovery
│   ├── losses.py            # ✓ Already migrated
│   ├── models.py            # Model architectures
│   ├── optimizers.py        # Optimizers
│   ├── metrics.py           # Metrics
│   ├── callbacks.py         # Callbacks
│   ├── data_loaders.py      # Data loaders
│   ├── visualizations.py    # Visualizations
│   ├── layers.py            # Custom layers
│   └── preprocessors.py     # Preprocessors
├── models/
│   ├── base.py              # Base classes
│   ├── gru_attention.py     # Default model
│   ├── lstm_transformer.py  # Alternative
│   ├── conv_net.py          # CNN model
│   └── layers/              # Custom layers
├── data/
│   ├── processor.py         # DataProcessor
│   ├── loaders/             # Data loaders
│   └── preprocessors/       # Preprocessors
├── training/
│   ├── trainer.py           # Training logic
│   └── custom_model.py      # CustomTrainModel
├── visualization/
│   ├── plotly_backend.py    # Plotly plots
│   └── matplotlib_backend.py # Matplotlib plots
├── utils/
│   └── metrics_utils.py     # Utilities
├── plugins/
│   ├── templates/           # Plugin templates
│   └── examples/            # Example plugins
├── tests/
│   ├── registries/          # Registry tests
│   ├── test_integration.py  # E2E tests
│   └── ...
├── docs/
│   ├── USER_GUIDE.md
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── MIGRATION.md
├── model.py                 # Entry point (<1000 lines)
├── config.yaml              # Configuration
└── README.md                # Updated docs
```

### B. Dependencies

**Required:**
- tensorflow ≥2.10
- pandas
- numpy
- pyyaml (new)

**Optional:**
- ccxt (API loaders)
- sqlalchemy (database loaders)
- plotly (visualization)
- matplotlib (visualization)
- tensorboard (logging)
- wandb (logging)
- mlflow (logging)

### C. Compatibility Matrix

| Python | TensorFlow | Status |
|--------|-----------|--------|
| 3.8    | 2.10-2.13 | ✓ Supported |
| 3.9    | 2.10-2.15 | ✓ Supported |
| 3.10   | 2.10-2.15 | ✓ Supported |
| 3.11   | 2.14-2.15 | ✓ Supported |

---

**Document Version:** 1.0
**Last Updated:** 2026-01-14
**Status:** Ready for Implementation
**Branch:** `claude/registry-modular-structure-1hpZb`

---

**End of Implementation Plan**
