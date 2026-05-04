"""
Custom exceptions for the registry system.

These exceptions provide clear error messages and help distinguish
registry-related errors from other application errors.
"""


class RegistryError(Exception):
    """
    Base exception for all registry-related errors.

    All custom registry exceptions inherit from this class,
    making it easy to catch any registry-related error.

    Example:
        try:
            model = Models.get("nonexistent")
        except RegistryError as e:
            print(f"Registry error: {e}")
    """
    pass


class ComponentNotFoundError(RegistryError):
    """
    Raised when a requested component is not found in the registry.

    This is typically raised by Registry.get() when the specified
    component name doesn't exist.

    Example:
        try:
            loss_fn = Losses.get("undefined_loss")
        except ComponentNotFoundError as e:
            print(f"Loss function not found: {e}")
            # Fallback to default
            loss_fn = Losses.get("mse")
    """
    pass


class ComponentValidationError(RegistryError):
    """
    Raised when a component fails validation during registration.

    This occurs when a component doesn't meet the requirements
    specified by the registry's validate_component() method.

    Example:
        @Models.register(name="invalid_model")
        def not_a_model_builder():
            # Missing required 'config' parameter
            return None

        # Would raise ComponentValidationError if strict validation enabled
    """
    pass


class DuplicateRegistrationError(RegistryError):
    """
    Raised when attempting to register a component with a name that
    already exists, without using override=True.

    Example:
        @Models.register(name="my_model")
        def model_v1(config):
            return model

        @Models.register(name="my_model")  # Without override=True
        def model_v2(config):
            return model

        # Would raise DuplicateRegistrationError if strict mode enabled
        # Currently shows warning instead
    """
    pass


class DependencyError(RegistryError):
    """
    Raised when a component's dependencies are not met.

    This can occur when a component specifies required dependencies
    that are not available in the environment.

    Example:
        @Models.register(
            name="advanced_model",
            dependencies=["tensorflow>=2.10", "transformers"]
        )
        def build_advanced_model(config):
            return model

        # Would raise DependencyError if dependencies not installed
    """
    pass


class RegistryNotInitializedError(RegistryError):
    """
    Raised when attempting to use a registry that hasn't been initialized.

    This can occur if auto-discovery hasn't been run or if the
    registry module hasn't been imported.

    Example:
        # Without importing registries
        try:
            model = Models.get("some_model")
        except RegistryNotInitializedError:
            # Import registries to trigger auto-discovery
            import registries
            model = Models.get("some_model")
    """
    pass


class InvalidConfigurationError(RegistryError):
    """
    Raised when configuration is invalid or inconsistent.

    This can occur when loading configuration from YAML or when
    validating configuration values.

    Example:
        try:
            config = Config.from_yaml("invalid_config.yaml")
        except InvalidConfigurationError as e:
            print(f"Configuration error: {e}")
            # Use default configuration
            config = Config()
    """
    pass
