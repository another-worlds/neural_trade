# Plugins

A plugin is a Python module that registers extra components into one of the nine registries
(Models, Optimizers, Metrics, Callbacks, DataLoaders, Visualizations, Layers, Preprocessors,
Losses). Nothing here is imported automatically. A program loads plugins explicitly:

```python
from neural_trade.registries import load_all
load_all(config, plugins_dir="plugins")          # or set Config.PLUGINS_DIR
```

or from the command line with `neural-trade registry list --plugins plugins`.

- `templates/` shows the shape of a model, loss and metric plugin. Files there are never loaded.
- `examples/echo_metric.py` is a working plugin: after loading, `Metrics.has("n_samples")` is true.

Every registry is strict: a component with the wrong signature, or a duplicate name, raises at
load time instead of failing later during training.
