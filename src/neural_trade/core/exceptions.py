"""Exceptions for the registry system and configuration.

Every exception here derives from :class:`RegistryError`. The two that replace a
built-in failure mode also derive from that built-in, so existing callers that
catch ``KeyError`` or ``ValueError`` keep working:

* :class:`ComponentNotFoundError` is a ``KeyError`` (raised by ``get``/``build``).
* :class:`InvalidConfigurationError` is a ``ValueError`` (raised by ``Config.validate``).
"""


class RegistryError(Exception):
    """Base class for every registry and configuration error."""


class ComponentNotFoundError(RegistryError, KeyError):
    """A requested component name is not registered.

    Also a ``KeyError``, so ``pytest.raises(KeyError)`` and ``except KeyError`` still match.
    """

    # KeyError.__str__ wraps the message in quotes; keep the plain message.
    __str__ = Exception.__str__


class ComponentValidationError(RegistryError):
    """A component failed ``validate_component`` while the registry was strict."""


class DuplicateRegistrationError(RegistryError):
    """A name was registered twice without ``override=True`` while the registry was strict."""


class DependencyError(RegistryError):
    """A component's declared dependencies cannot be imported."""


class RegistryNotInitializedError(RegistryError):
    """A declared discovery module could not be imported, so the registry is incomplete."""


class InvalidConfigurationError(RegistryError, ValueError):
    """A configuration value or combination of values is invalid.

    Also a ``ValueError``, so existing ``pytest.raises(ValueError)`` checks still match.
    """
