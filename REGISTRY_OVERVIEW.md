# Registry-Based Architecture - Executive Overview
## Neural Trade Project

**Quick Reference Guide**

---

## What is This?

A **registry-based modular architecture** that transforms the neural_trade codebase from a monolithic structure into a flexible, extensible system where components can be swapped like building blocks.

### Current Problem

```python
# Current: Everything hardcoded in model.py (3010 lines!)
model = PricePredictor.build_model(config)  # Only one architecture
optimizer = tf.keras.optimizers.Adam(...)   # Hardcoded optimizer
loss = custom_loss(...)                      # Hardcoded loss
# Want different model? Edit 3000 lines of code! 😰
```

### Solution: Registry Pattern

```python
# Future: Component selection via config
model = Models.get(config.MODEL_NAME, config=config)      # 10+ architectures
optimizer = Optimizers.get(config.OPTIMIZER_NAME, config)  # 5+ optimizers
loss = Losses.get(config.LOSS_NAME)                        # 10+ losses
# Want different model? Change config! 🎉
```

---

## Core Concept: Registries

A **registry** is a central catalog that manages components of a specific type.

### How It Works

```python
# 1. Register a component (one-time, on import)
@Models.register(name="gru_attention", tags=["rnn", "attention"])
def build_gru_attention_model(config):
    # ... build model ...
    return model

# 2. Use the component anywhere (config-driven)
model = Models.get("gru_attention", config=config)

# 3. List available components
Models.list_names()  # ['gru_attention', 'lstm_transformer', 'conv_net', ...]

# 4. Filter by tags
Models.filter_by_tag("rnn")  # ['gru_attention', 'lstm_transformer']
```

### Benefits

✅ **Modularity:** Each component is independent
✅ **Extensibility:** Add new components without editing existing code
✅ **Discoverability:** List all available options
✅ **Configuration:** Select components via config file
✅ **Testability:** Test each component in isolation
✅ **Plugins:** Third-party extensions without core changes

---

## The 9 Registries

### 1. **Models Registry** 🏗️
**What:** Model architectures (GRU-Attention, LSTM-Transformer, CNN, etc.)

```python
@Models.register(name="gru_attention", tags=["rnn", "attention", "default"])
def build_gru_attention_model(config: Config) -> tf.keras.Model:
    # Build and return model
    pass

# Usage
model = Models.get(config.MODEL_NAME, config=config)
```

**Current Status:** ✅ **Already exists** (losses.py pattern)
**Impact:** 🔥 **HIGH** - Enables architecture experimentation

---

### 2. **Optimizers Registry** ⚡
**What:** Optimizer configurations (Adam, AdamW, SGD, etc.)

```python
@Optimizers.register(name="adam", tags=["adaptive", "default"])
def build_adam(config: Config) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(lr=config.LR, ...)

# Usage
optimizer = Optimizers.get(config.OPTIMIZER_NAME, config=config)
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **HIGH** - Quick wins for optimization tuning

---

### 3. **Losses Registry** 🎯
**What:** Loss functions (focal, dice, Huber, custom composite, etc.)

```python
@Losses.register(name="custom_loss", tags=["composite", "default"])
def custom_loss(y_true, y_pred):
    # Compute loss
    pass

# Usage
loss_fn = Losses.get(config.LOSS_NAME)
```

**Current Status:** ✅ **DONE** - Already implemented in losses.py
**Impact:** 🔥 **HIGH** - Template for other registries

---

### 4. **Metrics Registry** 📊
**What:** Evaluation metrics (MSE, MAPE, direction accuracy, MCC, etc.)

```python
@Metrics.register(name="safe_mape", tags=["regression", "percentage"])
def safe_mape(y_true, y_pred):
    # Compute metric
    pass

# Usage
metrics = {name: Metrics.get(name) for name in config.METRICS}
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **MEDIUM** - Better evaluation flexibility

---

### 5. **Callbacks Registry** 📞
**What:** Training callbacks (TQDM, early stopping, plotting, logging, etc.)

```python
@Callbacks.register(name="tqdm_progress", tags=["progress", "console"])
def build_tqdm_callback(config: Config):
    return TqdmCallback(...)

# Usage
callbacks = [Callbacks.get(name, config) for name in config.CALLBACKS]
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **MEDIUM** - Better training customization

---

### 6. **Data Loaders Registry** 📥
**What:** Data loading strategies (CSV, database, API, etc.)

```python
@DataLoaders.register(name="csv", tags=["file", "default"])
def load_from_csv(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH, ...)

# Usage
df = DataLoaders.get(config.DATA_LOADER, config=config)
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **MEDIUM** - Multi-source data pipelines

---

### 7. **Visualizations Registry** 📈
**What:** Visualization backends (Plotly, Matplotlib, TensorBoard, W&B)

```python
@Visualizations.register(name="plotly_interactive", tags=["plotly", "default"])
def create_plotly_plot(data, config):
    # Create and return plot
    pass

# Usage
viz = Visualizations.get(config.VISUALIZATION, data=results, config=config)
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **LOW** - Better visualization options

---

### 8. **Layers Registry** 🧱
**What:** Custom Keras layers (LearnableIndicators, PositionalEncoding, etc.)

```python
@Layers.register(name="learnable_indicators", tags=["indicators", "default"])
def build_learnable_indicators(**kwargs):
    return LearnableIndicators(**kwargs)

# Usage
layer = Layers.get(config.INDICATOR_LAYER, **layer_config)
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **LOW** - Component customization

---

### 9. **Preprocessors Registry** 🔧
**What:** Data preprocessing steps (scaling, feature engineering, etc.)

```python
@Preprocessors.register(name="standard_scaler", tags=["scaling", "default"])
def apply_standard_scaler(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Scale and return
    pass

# Usage
for prep in config.PREPROCESSORS:
    df = Preprocessors.get(prep, df=df, config=config)
```

**Current Status:** 🟡 **Needs implementation**
**Impact:** 🔥 **MEDIUM** - Flexible data pipeline

---

## Configuration-Driven Workflow

### Before (Hardcoded)

```python
# model.py - Line 1111
optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LR)

# model.py - Line 1478
model = PricePredictor.build_model(config)

# Want to try SGD? Edit code, commit, push, pray 🙏
```

### After (Config-Driven)

```yaml
# config.yaml
model:
  name: gru_attention        # or lstm_transformer, conv_net, ...

optimizer:
  name: adam                  # or adamw, sgd_momentum, rmsprop, ...
  learning_rate: 0.001

loss:
  name: custom_loss           # or focal_loss, huber_loss, ...

metrics:
  - mse
  - safe_mape
  - direction_accuracy
  - mcc

callbacks:
  - tqdm_progress
  - early_stopping_ensemble
  - interactive_plot

data:
  loader: csv                 # or postgres, binance_api, ...
  path: ./data/price_data.csv

visualization:
  backend: plotly_interactive # or matplotlib_static, tensorboard, ...
```

```python
# training code (simplified)
def train_model(config_path: str):
    # Load config
    config = Config.from_yaml(config_path)

    # Get all components from registries
    data = DataLoaders.get(config.DATA_LOADER, config=config)
    model = Models.get(config.MODEL_NAME, config=config)
    optimizer = Optimizers.get(config.OPTIMIZER_NAME, config=config)
    loss = Losses.get(config.LOSS_NAME)

    # Compile
    model.compile(optimizer=optimizer, loss=loss)

    # Get callbacks
    callbacks = [Callbacks.get(name, config) for name in config.CALLBACKS]

    # Train
    model.fit(data, callbacks=callbacks, ...)

    # Visualize
    viz = Visualizations.get(config.VISUALIZATION)
    viz(results, config)
```

**Result:** Change `config.yaml`, no code changes needed! 🎉

---

## Plugin System

**Problem:** Want to add custom components without forking the repo?

**Solution:** External plugins!

### Plugin Example

```python
# plugins/my_custom_model.py

"""
External plugin that adds a custom WaveNet model.
Just drop this file in plugins/ and it's automatically available!
"""

from registries import Models
from core.config import Config
import tensorflow as tf

@Models.register(
    name="wavenet_predictor",
    tags=["cnn", "causal", "plugin"],
    author="Your Name"
)
def build_wavenet_model(config: Config) -> tf.keras.Model:
    """My awesome WaveNet architecture."""
    # Build WaveNet model
    inputs = tf.keras.Input(shape=(config.LOOKBACK, config.NUM_FEATURES))
    # ... implementation ...
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# That's it! Now available as:
# model = Models.get("wavenet_predictor", config=config)
```

**Usage:**

1. Drop plugin file in `plugins/` directory
2. Import registries (auto-discovery runs)
3. Use your component: `config.MODEL_NAME = "wavenet_predictor"`

**Plugin Discovery:**

```bash
# List all plugins
python -m scripts.plugin_cli list

# Show plugin details
python -m scripts.plugin_cli info my_custom_model

# Validate plugin
python -m scripts.plugin_cli validate my_custom_model
```

---

## Implementation Phases

### 📊 **Phase 1: Foundation** (Weeks 1-2) - 20% Complete
✅ Base registry infrastructure
✅ Migrate existing losses.py
✅ Auto-discovery system
✅ Enhanced configuration

**Deliverables:**
- `core/registry.py` - Base registry class
- `registries/losses.py` - Migrated (already exists!)
- `core/config.py` - YAML configuration support
- `core/plugin_loader.py` - Plugin system

---

### 🏗️ **Phase 2: Core Registries** (Weeks 3-6) - 60% Complete
🔲 Models registry (3 architectures)
🔲 Optimizers registry (5 optimizers)
🔲 Metrics registry (15 metrics)

**Deliverables:**
- `registries/models.py` - Model architectures
- `registries/optimizers.py` - Optimizers
- `registries/metrics.py` - Evaluation metrics
- Full config-driven training pipeline

---

### 📦 **Phase 3: Data & Visualization** (Weeks 7-8) - 80% Complete
🔲 Data loaders registry (5 loaders)
🔲 Preprocessors registry (10 preprocessors)
🔲 Visualizations registry (8 visualizations)

**Deliverables:**
- `registries/data_loaders.py` - Data loading
- `registries/preprocessors.py` - Preprocessing
- `registries/visualizations.py` - Visualization backends

---

### 🚀 **Phase 4: Advanced Features** (Weeks 9-10) - 90% Complete
🔲 Callbacks registry
🔲 Layers registry
🔲 Enhanced plugin system

**Deliverables:**
- `registries/callbacks.py` - Training callbacks
- `registries/layers.py` - Custom layers
- Plugin templates and examples
- Plugin CLI tool

---

### 🎨 **Phase 5: Polish** (Weeks 11-12) - 100% Complete
🔲 Code refactoring (model.py < 1000 lines)
🔲 Performance optimization
🔲 Testing (95%+ coverage)
🔲 Documentation
🔲 v2.0 Release

**Deliverables:**
- Refactored codebase
- Comprehensive documentation
- Performance benchmarks
- Release notes

---

## Benefits Summary

### For Developers 👨‍💻

✅ **Less Code:** Add components without editing 3000 lines
✅ **Clear Structure:** Know exactly where code belongs
✅ **Easy Testing:** Test components in isolation
✅ **Plugin Support:** Extend without forking

### For Researchers 🔬

✅ **Fast Experimentation:** Try different models via config
✅ **Reproducibility:** Config files = exact experiment setup
✅ **Comparison:** Run 10 configs, compare results
✅ **Custom Components:** Add specialized losses/metrics

### For Production 🚀

✅ **Maintainability:** Modular code is easier to maintain
✅ **Flexibility:** Swap components without code changes
✅ **Extensibility:** Add features without breaking existing code
✅ **Documentation:** Clear component catalog

---

## Quick Start (After Implementation)

### 1. Installation

```bash
pip install -e .
```

### 2. Basic Training

```python
from core.config import Config
from training.trainer import train_model

# Load config
config = Config.from_yaml("config.yaml")

# Train
results = train_model(config)
```

### 3. List Available Components

```python
from registries import Models, Optimizers, Losses, Metrics

print(f"Models: {Models.list_names()}")
print(f"Optimizers: {Optimizers.list_names()}")
print(f"Losses: {Losses.list_names()}")
print(f"Metrics: {Metrics.list_names()}")
```

### 4. Create Custom Component

```python
from registries import Models

@Models.register(name="my_model", tags=["custom"])
def build_my_model(config):
    # Build your model
    return model

# Now use it!
config.MODEL_NAME = "my_model"
model = Models.get(config.MODEL_NAME, config=config)
```

### 5. Create Plugin

```bash
# Create plugin file
cat > plugins/my_plugin.py << 'EOF'
from registries import Models

@Models.register(name="my_plugin_model")
def build_my_plugin_model(config):
    # Your implementation
    pass
EOF

# It's automatically available!
python -c "from registries import Models; print(Models.list_names())"
```

---

## File Structure

### Before 🔴

```
neural_trade/
├── model.py           (3010 lines 😱)
├── losses.py          (477 lines)
├── metrics_utils.py   (124 lines)
└── tests/
```

### After 🟢

```
neural_trade/
├── core/              # Base infrastructure
├── registries/        # 9 registries
├── models/            # Model architectures
├── data/              # Data pipeline
├── training/          # Training logic
├── visualization/     # Plots
├── plugins/           # External plugins
├── tests/             # Comprehensive tests
├── docs/              # Documentation
├── examples/          # Usage examples
├── model.py           (<1000 lines 🎉)
└── config.yaml        # Configuration
```

**Result:** 3010 lines → ~8 modular files of 200-500 lines each

---

## Comparison

| Feature | Before | After |
|---------|--------|-------|
| **Model Selection** | Edit code | Change config |
| **Add Model** | Edit 3010-line file | Create new file, register |
| **Add Metric** | Edit inline function | Register new metric |
| **Add Data Source** | Rewrite DataProcessor | Register new loader |
| **Plugins** | Fork repo | Drop file in plugins/ |
| **Experiment Tracking** | Manual code changes | Save config files |
| **Documentation** | Scattered comments | Auto-generated catalog |
| **Testing** | Test 3000 lines | Test 200-line module |

---

## Registry CLI Tools

### List Components

```bash
# List all models
python -m scripts.registry_cli list models

# Output:
# MODELS (3 registered)
#   • gru_attention           [rnn, attention, default]
#     Bi-GRU with multi-head attention and transformer blocks
#
#   • lstm_transformer        [rnn, transformer]
#     LSTM-based alternative with pure transformer decoder
#
#   • conv_net                [cnn, fast]
#     Lightweight CNN-based architecture for fast inference
```

### Show Component Info

```bash
# Show detailed info
python -m scripts.registry_cli info models gru_attention

# Output:
# GRU_ATTENTION
#   name            : gru_attention
#   description     : Bi-GRU with multi-head attention and transformer blocks
#   tags            : ['rnn', 'attention', 'default']
#   version         : 2.0.0
#   author          : Neural Trade Team
#   dependencies    : []
```

### Search by Tag

```bash
# Find all RNN-based models
python -m scripts.registry_cli search --tag rnn

# Output:
# Components tagged with 'rnn':
#   Models:
#     • gru_attention
#     • lstm_transformer
```

### Registry Summary

```bash
# Show all registries
python -c "from registries import print_registry_summary; print_registry_summary()"

# Output:
# ============================================================
# NEURAL TRADE - REGISTRY SUMMARY
# ============================================================
# Models                 3 registered
# Optimizers            5 registered
# Losses               10 registered
# Metrics              15 registered
# Callbacks             8 registered
# Data Loaders          5 registered
# Visualizations        8 registered
# Layers                5 registered
# Preprocessors        10 registered
# ============================================================
```

---

## FAQ

### Q: Do I need to rewrite all my code?
**A:** No! We maintain backward compatibility until Phase 5. Existing code continues to work.

### Q: Will this slow down training?
**A:** No. Registry lookup is one-time at initialization. Training loop unchanged.

### Q: Can I still use the old way?
**A:** Yes, during migration. After v2.0 release, registry-based is recommended.

### Q: How do I migrate my custom code?
**A:** See `docs/MIGRATION.md` for step-by-step guide.

### Q: Can I combine multiple registries?
**A:** Yes! That's the point. Mix and match any registered components.

### Q: What if I need a component not in the registry?
**A:** Create a plugin! No need to modify core code.

### Q: Is this over-engineering?
**A:** No. The codebase is already 3000+ lines. Registries add ~500 lines of infrastructure to enable infinite extensibility.

---

## Example Configurations

### Conservative Setup (Default)

```yaml
# config/conservative.yaml
model:
  name: gru_attention
optimizer:
  name: adam
  learning_rate: 0.001
loss:
  name: custom_loss
```

### Experimental Setup

```yaml
# config/experimental.yaml
model:
  name: wavenet_predictor   # Plugin model!
optimizer:
  name: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
loss:
  name: focal_loss          # Try different loss
metrics:
  - mse
  - sharpe_ratio           # Trading-specific metric
  - max_drawdown
```

### Fast Inference Setup

```yaml
# config/fast_inference.yaml
model:
  name: conv_net            # Lightweight model
optimizer:
  name: sgd_momentum
visualization:
  backend: matplotlib_static  # Static plots only
```

### Multi-Source Data

```yaml
# config/multi_source.yaml
data:
  loader: binance_api       # Live data from Binance
  symbol: BTCUSDT
  timeframe: 1h
preprocessors:
  - handle_missing_values
  - remove_outliers
  - add_technical_indicators
  - standard_scaler
```

---

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CONFIG.YAML                          │
│  model: gru_attention                                        │
│  optimizer: adam                                             │
│  loss: custom_loss                                           │
│  metrics: [mse, safe_mape, direction_accuracy]              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      REGISTRIES                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Models  │  │Optimizer│  │ Losses  │  │ Metrics │       │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │       │
│  │ │GRU  │ │  │ │Adam │ │  │ │Focal│ │  │ │MSE  │ │       │
│  │ │LSTM │ │  │ │AdamW│ │  │ │Dice │ │  │ │MAPE │ │       │
│  │ │CNN  │ │  │ │SGD  │ │  │ │Huber│ │  │ │MCC  │ │       │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       ▲              ▲            ▲            ▲            │
│       │              │            │            │            │
│  ┌────┴───────┬──────┴───────┬────┴───────┬────┴──────┐   │
│  │  @register │  @register   │ @register  │ @register │   │
│  └────────────┴──────────────┴────────────┴───────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                              │
│  1. Load data    → DataLoaders.get(config.DATA_LOADER)     │
│  2. Preprocess   → Preprocessors.get(...)                  │
│  3. Build model  → Models.get(config.MODEL_NAME)           │
│  4. Get optimizer→ Optimizers.get(config.OPTIMIZER_NAME)   │
│  5. Get loss     → Losses.get(config.LOSS_NAME)            │
│  6. Get callbacks→ Callbacks.get(...)                      │
│  7. Train        → model.fit(...)                          │
│  8. Evaluate     → Metrics.get(...)                        │
│  9. Visualize    → Visualizations.get(...)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

### 👉 Start Here

1. **Read the specs:** `REGISTRY_SPECIFICATIONS.md` (comprehensive design)
2. **Check the plan:** `IMPLEMENTATION_PLAN.md` (detailed tasks)
3. **Start Phase 1:** Follow Week 1 tasks in implementation plan

### 👉 For Contributors

1. Review architecture in `REGISTRY_SPECIFICATIONS.md`
2. Pick a registry to implement (Phase 2)
3. Follow the patterns from `losses.py`
4. Write tests
5. Submit PR

### 👉 For Users

1. Wait for Phase 2 completion (Models/Optimizers/Metrics)
2. Test with provided examples
3. Create custom components
4. Share feedback

---

## Contact & Resources

- **Specifications:** `REGISTRY_SPECIFICATIONS.md`
- **Implementation Plan:** `IMPLEMENTATION_PLAN.md`
- **Current Branch:** `claude/registry-modular-structure-1hpZb`
- **Status:** Planning Phase ✅ → Implementation Phase 🏗️

---

## Summary

**What we're building:**
- 9 registries for all major components
- Config-driven component selection
- Plugin system for extensions
- Modular, testable architecture

**Why it matters:**
- 3010-line file → Modular components
- Hardcoded → Config-driven
- Monolithic → Extensible
- Difficult → Easy experimentation

**When it's done:**
- Week 12: Full implementation
- All registries functional
- Plugin system working
- Comprehensive docs
- v2.0 released

**How to help:**
- Review specifications
- Implement registries
- Write plugins
- Provide feedback

---

**Let's build something amazing! 🚀**

---

*Document Version: 1.0*
*Last Updated: 2026-01-14*
*Status: Ready for Implementation*
