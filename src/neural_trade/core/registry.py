"""Base registry shared by the nine component registries.

A registry maps a name to a component (a builder function, a Layer class, a metric,
...) plus metadata. Components register themselves with a decorator::

    class Models(BaseRegistry):
        strict = True
        default = "gru_attention"

        @classmethod
        def validate_component(cls, component):
            return callable(component)

    @Models.register(name="gru_attention", tags=["rnn", "attention", "default"])
    def build_gru_attention(config): ...

    model = Models.build(None, config)            # default component, called
    builder = Models.get("gru_attention")         # the component itself

Hardening over the original design (remediation plan section B3):

* Each subclass gets its own ``registry`` dict automatically (``__init_subclass__``);
  forgetting to redeclare it used to share one dict across every registry.
* ``strict`` registries raise :class:`ComponentValidationError` and
  :class:`DuplicateRegistrationError`; non-strict ones warn and register anyway (the
  original behaviour, still used by ad-hoc/test registries).
* Unknown names raise :class:`ComponentNotFoundError` (a ``KeyError``).
* Declared ``dependencies`` are checked on ``get``/``build`` and raise
  :class:`DependencyError` when not importable.
* ``default``/``get_default``/``resolve`` and ``build`` (always calls the component).
* ``discovery_modules`` + ``auto_discover()``: explicit imports, never a directory glob
  (a glob once imported a stale 3,600-line copy of the model).
"""
from __future__ import annotations

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar

from .exceptions import (
    ComponentNotFoundError,
    ComponentValidationError,
    DependencyError,
    DuplicateRegistrationError,
    RegistryNotInitializedError,
)

T = TypeVar("T")

# Distribution names whose import name differs.
_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "tensorflow-cpu": "tensorflow",
    "ipython": "IPython",
}


def _import_name(requirement: str) -> str:
    name = requirement
    for sep in ("[", "<", ">", "=", "!", "~", ";", " "):
        name = name.split(sep, 1)[0]
    name = name.strip()
    return _IMPORT_NAME.get(name.lower(), name.replace("-", "_"))


@dataclass
class RegistryEntry(Generic[T]):
    """Metadata for one registered component."""

    name: str
    component: T
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    registered_at: Optional[str] = None

    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.now().isoformat()

    def missing_dependencies(self) -> List[str]:
        return [d for d in self.dependencies if importlib.util.find_spec(_import_name(d)) is None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
            "dependencies": self.dependencies,
            "registered_at": self.registered_at,
        }


class BaseRegistry:
    """Abstract base for all registries (subclasses implement ``validate_component``)."""

    registry: ClassVar[Dict[str, RegistryEntry]] = {}
    _initialized: ClassVar[bool] = False
    #: raise instead of warning on validation failures and duplicate names
    strict: ClassVar[bool] = False
    #: component used when ``resolve``/``build`` get ``name=None``
    default: ClassVar[Optional[str]] = None
    #: modules imported by ``auto_discover`` (their decorators register components)
    discovery_modules: ClassVar[Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "registry" not in cls.__dict__:
            cls.registry = {}
        cls._initialized = False

    # ------------------------------------------------------------------ registration
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
    ) -> Callable[[T], T]:
        """Decorator registering ``component`` under ``name`` (default: its ``__name__``)."""

        def decorator(component):
            component_name = name or component.__name__

            if component_name in cls.registry and not override:
                msg = (f"{cls.__name__}: '{component_name}' already registered. "
                       f"Use override=True to replace.")
                if cls.strict:
                    raise DuplicateRegistrationError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                return component

            component_description = description
            if component_description is None and getattr(component, "__doc__", None):
                component_description = component.__doc__.strip().split("\n")[0].strip()

            entry = RegistryEntry(
                name=component_name,
                component=component,
                description=component_description,
                tags=list(tags or []),
                version=version,
                author=author,
                dependencies=list(dependencies or []),
            )

            try:
                valid = bool(cls.validate_component(component))
                problem = None if valid else "failed validation"
            except Exception as exc:  # a validator must never take the registry down
                valid, problem = False, f"error validating: {exc}"
            if not valid:
                if cls.strict:
                    raise ComponentValidationError(
                        f"{cls.__name__}: component '{component_name}' {problem}")
                warnings.warn(
                    f"{cls.__name__}: Component '{component_name}' {problem}. Registering anyway.",
                    UserWarning,
                    stacklevel=2,
                )

            cls.registry[component_name] = entry
            try:
                component._registry_name = component_name
                component._registry_entry = entry
            except (AttributeError, TypeError):
                pass  # builtins and some callables refuse attributes; metadata lives in the entry
            return component

        return decorator

    # ------------------------------------------------------------------ lookup
    @classmethod
    def entry(cls, name: str) -> RegistryEntry:
        if name not in cls.registry:
            available = ", ".join(cls.list_names())
            raise ComponentNotFoundError(f"{cls.__name__}: '{name}' not found. Available: [{available}]")
        return cls.registry[name]

    @classmethod
    def _checked_component(cls, name: str) -> Any:
        entry = cls.entry(name)
        missing = entry.missing_dependencies()
        if missing:
            raise DependencyError(
                f"{cls.__name__}: '{name}' needs {missing}, which cannot be imported. "
                f"Install them (e.g. pip install {' '.join(missing)}).")
        return entry.component

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Return the component; if keyword arguments are given and it is callable, call it.

        The call-if-kwargs behaviour is legacy; new code should use :meth:`build`.
        """
        component = cls._checked_component(name)
        if kwargs and callable(component):
            return component(**kwargs)
        return component

    @classmethod
    def resolve(cls, name: Optional[str]) -> str:
        """``name`` itself, or the registry default when ``name`` is None/empty."""
        if name:
            return name
        if cls.default is None:
            raise ComponentNotFoundError(f"{cls.__name__}: no name given and no default set")
        return cls.default

    @classmethod
    def get_default(cls) -> Any:
        return cls.get(cls.resolve(None))

    @classmethod
    def build(cls, name: Optional[str], /, *args, **kwargs) -> Any:
        """Resolve ``name`` (None -> default) and call the component with the arguments."""
        component = cls._checked_component(cls.resolve(name))
        if not callable(component):
            raise ComponentValidationError(f"{cls.__name__}: '{name}' is not callable")
        return component(*args, **kwargs)

    # ------------------------------------------------------------------ listing
    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in cls.registry.values()]

    @classmethod
    def list_names(cls) -> List[str]:
        return sorted(cls.registry.keys())

    @classmethod
    def filter_by_tag(cls, tag: str) -> List[str]:
        return [n for n, e in cls.registry.items() if tag in e.tags]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.registry

    @classmethod
    def count(cls) -> int:
        return len(cls.registry)

    @classmethod
    def search(cls, query: str) -> List[str]:
        """Names whose name, description or tags contain ``query`` (case-insensitive)."""
        q = query.lower()
        out = []
        for n, e in cls.registry.items():
            if q in n.lower() or (e.description and q in e.description.lower()) \
                    or any(q in t.lower() for t in e.tags):
                out.append(n)
        return out

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        return cls.entry(name).to_dict()

    # ------------------------------------------------------------------ maintenance
    @classmethod
    def remove(cls, name: str) -> None:
        cls.registry.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls.registry.clear()
        cls._initialized = False

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        """Return True when ``component`` satisfies this registry's contract."""
        raise NotImplementedError(f"{cls.__name__} must implement validate_component")

    @classmethod
    def auto_discover(cls) -> None:
        """Import every module in ``discovery_modules`` so their decorators run."""
        for mod in cls.discovery_modules:
            try:
                importlib.import_module(mod)
            except ImportError as exc:
                msg = f"{cls.__name__}: discovery module '{mod}' failed to import: {exc}"
                if cls.strict:
                    raise RegistryNotInitializedError(msg) from exc
                warnings.warn(msg, UserWarning, stacklevel=2)
        cls._initialized = True


class RegistryMixin:
    """Convenience accessors for classes that look components up in registries."""

    @staticmethod
    def get_from_registry(registry: type, name: str, **kwargs):
        return registry.get(name, **kwargs)

    @staticmethod
    def list_from_registry(registry: type) -> List[str]:
        return registry.list_names()

    @staticmethod
    def has_in_registry(registry: type, name: str) -> bool:
        return registry.has(name)
