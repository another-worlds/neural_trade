"""
Core module for neural_trade registry-based architecture.

This module provides the base infrastructure for the registry system:
- BaseRegistry: Abstract base class for all registries
- RegistryEntry: Metadata container for registered components
- RegistryMixin: Helper mixin for classes that use registries
- Custom exceptions for registry errors

Example:
    from core import BaseRegistry, RegistryEntry
    from core.exceptions import ComponentNotFoundError

    class MyRegistry(BaseRegistry):
        registry = {}

        @classmethod
        def validate_component(cls, component):
            return callable(component)
"""

# Import base registry classes
from .registry import (
    BaseRegistry,
    RegistryEntry,
    RegistryMixin,
)

# Import exceptions
from .exceptions import (
    RegistryError,
    ComponentNotFoundError,
    ComponentValidationError,
    DuplicateRegistrationError,
    DependencyError,
    RegistryNotInitializedError,
    InvalidConfigurationError,
)

# Define public API
__all__ = [
    # Registry classes
    "BaseRegistry",
    "RegistryEntry",
    "RegistryMixin",
    # Exceptions
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentValidationError",
    "DuplicateRegistrationError",
    "DependencyError",
    "RegistryNotInitializedError",
    "InvalidConfigurationError",
]

# Version information
__version__ = "2.0.0-alpha"
__author__ = "Neural Trade Team"
