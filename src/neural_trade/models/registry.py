"""The Models registry class (registry 1 of 9): architecture builders.

Moved here from ``neural_trade.registries.models`` (NT-027): ``training/trainer.py``,
``training/custom_model.py`` and ``training/artifacts.py`` dispatch through it
(``Models.build(...)``), so the registry now lives in the same package as its client and
``models/`` no longer needs to import ``neural_trade.registries``. ``neural_trade.registries
.models`` re-exports :class:`Models` and :func:`ensure_predictive_outputs` for their old callers.
(The ``PricePredictor`` facade that used to sit in front of ``Models.build`` was removed in
NT-028: it had no caller besides the two tests that checked it exist.)

A component is ``builder(config) -> tf.keras.Model``. ``Models.build(name, config)`` checks
the result has the 3N + 1 outputs of ``core.horizons.predictive_outputs_type(N)`` (N horizons
x price/direction/variance + vacuum overflow; the 10 of
:class:`~neural_trade.core.outputs.PredictiveOutputs` at the reference N = 3), each of shape
``[None, 1]`` and named after its field, so a new architecture cannot silently break the
training objective or the serving path.

Deferred (named in docs/archive/REGISTRY_SPECIFICATIONS.md, the pre-remediation registry design,
no implementation exists yet): ``lstm_transformer``, ``conv1d_attention``.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Optional, Tuple

import tensorflow as tf

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.horizons import REFERENCE_N_HORIZONS, predictive_output_fields
from neural_trade.core.horizons import n_horizons as _n_horizons
from neural_trade.core.registry import BaseRegistry


def ensure_predictive_outputs(model: tf.keras.Model, name: str = "model", *,
                              n_horizons: Optional[int] = None) -> tf.keras.Model:
    """Raise unless ``model`` returns the 3N + 1 PredictiveOutputs heads, in order, each ``[None, 1]``.

    ``n_horizons`` defaults to the reference three (10 outputs). Besides the count and the shapes,
    every output must carry its field's name (NT-042): the producing layer is called
    ``price_h0`` / ``direction_h0`` / ..., or that name plus a suffix such as ``_clip``
    (``price_h0_clip``), so swapped or misnamed heads are refused instead of silently trained on
    the wrong target.
    """
    fields = predictive_output_fields(REFERENCE_N_HORIZONS if n_horizons is None else int(n_horizons))
    outputs = list(model.outputs)
    expected = len(fields)
    if len(outputs) != expected:
        raise ComponentValidationError(
            f"{name}: expected {expected} outputs {fields}, got {len(outputs)}")
    bad = [(f, tuple(o.shape)) for f, o in zip(fields, outputs)
           if tuple(o.shape)[1:] != (1,)]
    if bad:
        raise ComponentValidationError(f"{name}: every output must be [None, 1]; got {bad}")
    names = list(getattr(model, "output_names", None) or [])
    if len(names) == expected:
        wrong = [(f, n) for f, n in zip(fields, names) if not (n == f or n.startswith(f + "_"))]
        if wrong:
            raise ComponentValidationError(
                f"{name}: output names do not match the PredictiveOutputs fields (expected, got): {wrong}")
    return model


class Models(BaseRegistry):
    registry = {}
    strict = True
    default = "gru_attention"
    discovery_modules: ClassVar[Tuple[str, ...]] = (
        "neural_trade.models.gru_attention",
        "neural_trade.models.gru_small",
        "neural_trade.models.linear_indicators",
    )

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and bool(params) and params[0] == "config"

    @classmethod
    def build(cls, name, config, /, *args, **kwargs) -> tf.keras.Model:
        resolved = cls.resolve(name)
        model = super().build(resolved, config, *args, **kwargs)
        if not isinstance(model, tf.keras.Model):
            raise ComponentValidationError(f"Models: '{resolved}' did not return a tf.keras.Model")
        return ensure_predictive_outputs(model, resolved, n_horizons=_n_horizons(config))


from neural_trade.models.gru_attention import build_gru_attention  # noqa: E402
from neural_trade.models.gru_small import build_gru_small  # noqa: E402
from neural_trade.models.linear_indicators import build_linear_indicators  # noqa: E402

Models.register(name="gru_attention", version="3.0.0",
                tags=["rnn", "attention", "transformer", "multi_horizon", "default"],
                description="Learnable indicators + Bi-GRU + attention + energy-gated convs, one tower per horizon")(
    build_gru_attention)
Models.register(name="gru_small", version="1.0.0",
                tags=["rnn", "multi_horizon", "capacity_study"],
                description="Learnable indicators + one GRU(32), same towers/heads as gru_attention (NT-104)")(
    build_gru_small)
Models.register(name="linear_indicators", version="1.0.0",
                tags=["linear", "multi_horizon", "capacity_study"],
                description="Learnable indicators, pooled, linear heads (NT-104)")(
    build_linear_indicators)
Models._initialized = True
