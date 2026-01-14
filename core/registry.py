"""
Base registry implementation for all component types.
Provides consistent API and functionality across all registries.
"""

from typing import Dict, Callable, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import warnings
from datetime import datetime

T = TypeVar('T')


@dataclass
class RegistryEntry(Generic[T]):
    """
    Metadata for a registered component.

    Attributes:
        name: Unique component name
        component: The actual component (function, class, etc.)
        description: Human-readable description
        tags: Categorization tags for filtering
        version: Semantic version string
        author: Component author
        dependencies: Required dependencies
        registered_at: Registration timestamp
    """
    name: str
    component: T
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    registered_at: Optional[str] = None

    def __post_init__(self):
        """Set registration timestamp if not provided."""
        if self.registered_at is None:
            self.registered_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
            "dependencies": self.dependencies,
            "registered_at": self.registered_at,
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

    Example:
        class Models(BaseRegistry):
            registry = {}

            @classmethod
            def validate_component(cls, component):
                return callable(component)

        @Models.register(name="my_model", tags=["custom"])
        def build_model(config):
            return model

        model_builder = Models.get("my_model")
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

        Returns:
            Decorator function

        Example:
            @Models.register(name="gru_attention", tags=["rnn", "attention"])
            def build_gru_attention_model(config):
                return model
        """
        def decorator(component):
            component_name = name or component.__name__

            # Check for duplicates
            if component_name in cls.registry and not override:
                warnings.warn(
                    f"{cls.__name__}: '{component_name}' already registered. "
                    f"Use override=True to replace.",
                    UserWarning,
                    stacklevel=2
                )
                return component

            # Extract description from docstring if not provided
            component_description = description
            if component_description is None and component.__doc__:
                # Get first line of docstring
                doc_lines = component.__doc__.strip().split('\n')
                component_description = doc_lines[0].strip()

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

            # Validate component
            try:
                if not cls.validate_component(component):
                    warnings.warn(
                        f"{cls.__name__}: Component '{component_name}' failed validation. "
                        f"Registering anyway.",
                        UserWarning,
                        stacklevel=2
                    )
            except Exception as e:
                warnings.warn(
                    f"{cls.__name__}: Error validating '{component_name}': {e}. "
                    f"Registering anyway.",
                    UserWarning,
                    stacklevel=2
                )

            # Register component
            cls.registry[component_name] = entry

            # Add metadata to component (if possible)
            try:
                component._registry_name = component_name
                component._registry_entry = entry
            except (AttributeError, TypeError):
                # Some objects (like integers, strings) don't support attribute assignment
                # This is fine - metadata is stored in the registry anyway
                pass

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

        Example:
            model_builder = Models.get("gru_attention")
            model = model_builder(config)

            # Or with kwargs
            model = Models.get("gru_attention", config=config)
        """
        if name not in cls.registry:
            available = ", ".join(cls.list_names())
            raise KeyError(
                f"{cls.__name__}: '{name}' not found. "
                f"Available: [{available}]"
            )

        entry = cls.registry[name]
        component = entry.component

        # If kwargs provided and component is callable, call it
        if kwargs and callable(component):
            return component(**kwargs)

        return component

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """
        List all registered components with metadata.

        Returns:
            List of dictionaries containing component metadata

        Example:
            components = Models.list()
            for comp in components:
                print(f"{comp['name']}: {comp['description']}")
        """
        return [entry.to_dict() for entry in cls.registry.values()]

    @classmethod
    def list_names(cls) -> List[str]:
        """
        List all registered component names.

        Returns:
            Sorted list of component names

        Example:
            names = Models.list_names()
            print(f"Available models: {', '.join(names)}")
        """
        return sorted(cls.registry.keys())

    @classmethod
    def filter_by_tag(cls, tag: str) -> List[str]:
        """
        Filter components by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of component names matching the tag

        Example:
            rnn_models = Models.filter_by_tag("rnn")
        """
        return [
            name for name, entry in cls.registry.items()
            if tag in entry.tags
        ]

    @classmethod
    def has(cls, name: str) -> bool:
        """
        Check if component is registered.

        Args:
            name: Component name

        Returns:
            True if component exists, False otherwise

        Example:
            if Models.has("my_model"):
                model = Models.get("my_model")
        """
        return name in cls.registry

    @classmethod
    def remove(cls, name: str) -> None:
        """
        Remove a registered component (primarily for testing).

        Args:
            name: Component name to remove

        Example:
            Models.remove("test_model")
        """
        if name in cls.registry:
            del cls.registry[name]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered components (primarily for testing).

        Example:
            Models.clear()
        """
        cls.registry.clear()
        cls._initialized = False

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered component.

        Args:
            name: Component name

        Returns:
            Dictionary containing component metadata

        Raises:
            KeyError: If component not found

        Example:
            metadata = Models.get_metadata("gru_attention")
            print(f"Version: {metadata['version']}")
            print(f"Tags: {metadata['tags']}")
        """
        if name not in cls.registry:
            raise KeyError(f"{cls.__name__}: '{name}' not found")
        return cls.registry[name].to_dict()

    @classmethod
    @abstractmethod
    def validate_component(cls, component: Any) -> bool:
        """
        Validate that a component meets registry requirements.

        To be implemented by subclasses.

        Args:
            component: Component to validate

        Returns:
            True if valid, False otherwise

        Example:
            class Models(BaseRegistry):
                @classmethod
                def validate_component(cls, component):
                    # Check that component is callable
                    return callable(component)
        """
        pass

    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover and register components from module.

        To be implemented by subclasses if auto-discovery is needed.
        Default implementation does nothing.

        Example:
            class Models(BaseRegistry):
                @classmethod
                def auto_discover(cls):
                    # Import all modules in models/ directory
                    # Components will register themselves via decorators
                    import models.gru_attention
                    import models.lstm_transformer
        """
        pass

    @classmethod
    def count(cls) -> int:
        """
        Get count of registered components.

        Returns:
            Number of registered components

        Example:
            print(f"Total models: {Models.count()}")
        """
        return len(cls.registry)

    @classmethod
    def search(cls, query: str) -> List[str]:
        """
        Search component names and descriptions for query string.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching component names

        Example:
            results = Models.search("attention")
            # Returns: ["gru_attention", "lstm_attention", ...]
        """
        query_lower = query.lower()
        results = []

        for name, entry in cls.registry.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue

            # Search in description
            if entry.description and query_lower in entry.description.lower():
                results.append(name)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in entry.tags):
                results.append(name)

        return results


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
        """
        Get component from specified registry.

        Args:
            registry: Registry class to query
            name: Component name
            **kwargs: Additional arguments for component

        Returns:
            Retrieved component

        Example:
            loss_fn = self.get_from_registry(Losses, "focal_loss")
        """
        return registry.get(name, **kwargs)

    @staticmethod
    def list_from_registry(registry: BaseRegistry) -> List[str]:
        """
        List all components in specified registry.

        Args:
            registry: Registry class to query

        Returns:
            List of component names

        Example:
            models = self.list_from_registry(Models)
        """
        return registry.list_names()

    @staticmethod
    def has_in_registry(registry: BaseRegistry, name: str) -> bool:
        """
        Check if component exists in registry.

        Args:
            registry: Registry class to query
            name: Component name

        Returns:
            True if component exists

        Example:
            if self.has_in_registry(Models, "my_model"):
                ...
        """
        return registry.has(name)
