"""BaseRegistry hardening (plan section B3): per-subclass storage, strict mode, live
exceptions, defaults, dependency checks, build(), explicit discovery."""
from __future__ import annotations

import sys
import types

import pytest

from neural_trade.core.exceptions import (
    ComponentNotFoundError,
    ComponentValidationError,
    DependencyError,
    DuplicateRegistrationError,
    RegistryError,
    RegistryNotInitializedError,
)
from neural_trade.core.registry import BaseRegistry


def _make(strict=False, default=None, discovery=()):
    class R(BaseRegistry):  # note: no `registry = {}` - __init_subclass__ must add one
        @classmethod
        def validate_component(cls, component):
            return callable(component)

    R.strict, R.default, R.discovery_modules = strict, default, tuple(discovery)
    return R


def test_each_subclass_gets_its_own_registry_dict():
    a, b = _make(), _make()
    a.register(name="only_in_a")(lambda: 1)
    assert a.has("only_in_a") and not b.has("only_in_a")
    assert a.registry is not b.registry is not BaseRegistry.registry


def test_not_found_is_both_a_registry_error_and_a_key_error():
    r = _make()
    with pytest.raises(ComponentNotFoundError, match="not found") as exc:
        r.get("missing")
    assert isinstance(exc.value, KeyError) and isinstance(exc.value, RegistryError)
    assert not str(exc.value).startswith("'"), "message should not carry KeyError's quotes"
    with pytest.raises(KeyError):
        r.get_metadata("missing")


def test_strict_registry_raises_on_invalid_and_duplicate():
    r = _make(strict=True)
    with pytest.raises(ComponentValidationError):
        r.register(name="not_callable")(42)
    r.register(name="f")(lambda: 1)
    with pytest.raises(DuplicateRegistrationError):
        r.register(name="f")(lambda: 2)
    r.register(name="f", override=True)(lambda: 3)
    assert r.build("f") == 3


def test_non_strict_registry_warns_and_keeps_the_first():
    r = _make(strict=False)
    r.register(name="f")(lambda: 1)
    with pytest.warns(UserWarning, match="already registered"):
        r.register(name="f")(lambda: 2)
    assert r.build("f") == 1


def test_default_resolve_build_and_get_default():
    r = _make(default="d")
    r.register(name="d")(lambda x=0: x + 1)
    assert r.resolve(None) == "d" and r.resolve("x") == "x"
    assert r.build(None, 4) == 5
    assert r.get_default()(1) == 2
    with pytest.raises(ComponentNotFoundError):
        _make().resolve(None)


def test_missing_dependency_raises_on_get_and_build():
    r = _make()
    r.register(name="needs_nothing_real", dependencies=["definitely_not_installed_pkg>=1.0"])(lambda: 1)
    with pytest.raises(DependencyError, match="definitely_not_installed_pkg"):
        r.get("needs_nothing_real")
    with pytest.raises(DependencyError):
        r.build("needs_nothing_real")
    r.register(name="ok", dependencies=["numpy", "scikit-learn"])(lambda: 2)
    assert r.build("ok") == 2


def test_auto_discover_imports_declared_modules_and_fails_loudly_when_strict():
    r = _make(strict=True)
    mod = types.ModuleType("_nt_fake_discovery")

    def _register():
        r.register(name="from_discovery")(lambda: "found")

    mod.__dict__["_register"] = _register
    sys.modules["_nt_fake_discovery"] = mod
    _register()
    r.discovery_modules = ("_nt_fake_discovery",)
    r.auto_discover()
    assert r._initialized and r.build("from_discovery") == "found"

    r.discovery_modules = ("_nt_this_module_does_not_exist",)
    with pytest.raises(RegistryNotInitializedError):
        r.auto_discover()
    del sys.modules["_nt_fake_discovery"]


def test_legacy_import_path_is_the_same_object():
    import core.registry as legacy
    import neural_trade.core.registry as new
    assert legacy is new and legacy.BaseRegistry is BaseRegistry
