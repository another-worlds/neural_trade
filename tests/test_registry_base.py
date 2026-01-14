"""
Unit tests for base registry functionality.

Tests the core BaseRegistry class and RegistryEntry to ensure
all functionality works correctly before implementing specific registries.
"""

import pytest
import warnings
from core.registry import BaseRegistry, RegistryEntry, RegistryMixin
from core.exceptions import RegistryError


class TestRegistry(BaseRegistry):
    """Test registry implementation for testing purposes."""
    registry = {}

    @classmethod
    def validate_component(cls, component):
        """Simple validation: component must be callable."""
        return callable(component)


# Fixtures

@pytest.fixture(autouse=True)
def clear_test_registry():
    """Clear test registry before and after each test."""
    TestRegistry.clear()
    yield
    TestRegistry.clear()


# Test RegistryEntry

def test_registry_entry_creation():
    """Test creating a registry entry with all fields."""
    def dummy_func():
        pass

    entry = RegistryEntry(
        name="test_func",
        component=dummy_func,
        description="A test function",
        tags=["test", "dummy"],
        version="1.0.0",
        author="Test Author",
        dependencies=["dep1", "dep2"]
    )

    assert entry.name == "test_func"
    assert entry.component == dummy_func
    assert entry.description == "A test function"
    assert entry.tags == ["test", "dummy"]
    assert entry.version == "1.0.0"
    assert entry.author == "Test Author"
    assert entry.dependencies == ["dep1", "dep2"]
    assert entry.registered_at is not None


def test_registry_entry_to_dict():
    """Test converting registry entry to dictionary."""
    def dummy_func():
        pass

    entry = RegistryEntry(
        name="test_func",
        component=dummy_func,
        description="A test function",
        tags=["test"],
        version="1.0.0"
    )

    entry_dict = entry.to_dict()

    assert entry_dict["name"] == "test_func"
    assert entry_dict["description"] == "A test function"
    assert entry_dict["tags"] == ["test"]
    assert entry_dict["version"] == "1.0.0"
    assert "registered_at" in entry_dict


# Test Component Registration

def test_register_component():
    """Test basic component registration."""
    @TestRegistry.register(name="test_func")
    def test_function():
        return 42

    assert TestRegistry.has("test_func")
    assert TestRegistry.get("test_func")() == 42


def test_register_without_name():
    """Test registration using function name."""
    @TestRegistry.register()
    def my_function():
        return "hello"

    assert TestRegistry.has("my_function")
    assert TestRegistry.get("my_function")() == "hello"


def test_register_with_metadata():
    """Test registration with full metadata."""
    @TestRegistry.register(
        name="advanced_func",
        description="An advanced function",
        tags=["advanced", "special"],
        version="2.0.0",
        author="Test Author"
    )
    def advanced_function():
        return "advanced"

    metadata = TestRegistry.get_metadata("advanced_func")
    assert metadata["description"] == "An advanced function"
    assert "advanced" in metadata["tags"]
    assert "special" in metadata["tags"]
    assert metadata["version"] == "2.0.0"
    assert metadata["author"] == "Test Author"


def test_register_with_docstring():
    """Test that docstring is used as description if not provided."""
    @TestRegistry.register(name="documented_func")
    def documented_function():
        """This is a documented function."""
        return "doc"

    metadata = TestRegistry.get_metadata("documented_func")
    assert metadata["description"] == "This is a documented function."


def test_register_adds_metadata_to_component():
    """Test that registration adds metadata attributes to component."""
    @TestRegistry.register(name="meta_func")
    def meta_function():
        return True

    assert hasattr(meta_function, '_registry_name')
    assert hasattr(meta_function, '_registry_entry')
    assert meta_function._registry_name == "meta_func"


# Test Duplicate Registration

def test_duplicate_registration_warning():
    """Test warning on duplicate registration without override."""
    @TestRegistry.register(name="duplicate")
    def func1():
        return 1

    with pytest.warns(UserWarning, match="already registered"):
        @TestRegistry.register(name="duplicate")
        def func2():
            return 2

    # First registration should be kept
    assert TestRegistry.get("duplicate")() == 1


def test_override_registration():
    """Test overriding registration with override=True."""
    @TestRegistry.register(name="override_test")
    def func1():
        return 1

    @TestRegistry.register(name="override_test", override=True)
    def func2():
        return 2

    # Second registration should replace first
    assert TestRegistry.get("override_test")() == 2


# Test Component Retrieval

def test_get_component():
    """Test retrieving a registered component."""
    @TestRegistry.register(name="get_test")
    def test_func():
        return "success"

    component = TestRegistry.get("get_test")
    assert callable(component)
    assert component() == "success"


def test_get_with_kwargs():
    """Test retrieving and calling component with kwargs."""
    @TestRegistry.register(name="kwargs_test")
    def test_func(x, y=10):
        return x + y

    # Get without kwargs - returns function
    func = TestRegistry.get("kwargs_test")
    assert callable(func)
    assert func(5) == 15

    # Get with kwargs - returns result
    result = TestRegistry.get("kwargs_test", x=3, y=7)
    assert result == 10


def test_get_nonexistent_component():
    """Test error when component not found."""
    with pytest.raises(KeyError, match="not found"):
        TestRegistry.get("nonexistent")


# Test Listing and Filtering

def test_list_names():
    """Test listing registered component names."""
    @TestRegistry.register(name="func1")
    def f1():
        pass

    @TestRegistry.register(name="func2")
    def f2():
        pass

    names = TestRegistry.list_names()
    assert "func1" in names
    assert "func2" in names
    assert len(names) == 2


def test_list_with_metadata():
    """Test listing components with full metadata."""
    @TestRegistry.register(name="func1", tags=["tag1"])
    def f1():
        pass

    @TestRegistry.register(name="func2", tags=["tag2"])
    def f2():
        pass

    components = TestRegistry.list()
    assert len(components) == 2
    assert any(c["name"] == "func1" and "tag1" in c["tags"] for c in components)
    assert any(c["name"] == "func2" and "tag2" in c["tags"] for c in components)


def test_filter_by_tag():
    """Test filtering components by tag."""
    @TestRegistry.register(name="func1", tags=["tag_a"])
    def f1():
        pass

    @TestRegistry.register(name="func2", tags=["tag_b"])
    def f2():
        pass

    @TestRegistry.register(name="func3", tags=["tag_a", "tag_b"])
    def f3():
        pass

    tag_a_components = TestRegistry.filter_by_tag("tag_a")
    assert "func1" in tag_a_components
    assert "func3" in tag_a_components
    assert "func2" not in tag_a_components

    tag_b_components = TestRegistry.filter_by_tag("tag_b")
    assert "func2" in tag_b_components
    assert "func3" in tag_b_components
    assert "func1" not in tag_b_components


def test_search():
    """Test searching components."""
    @TestRegistry.register(name="attention_model", description="Model with attention")
    def f1():
        pass

    @TestRegistry.register(name="basic_model", tags=["attention"])
    def f2():
        pass

    @TestRegistry.register(name="simple_func")
    def f3():
        pass

    # Search in name
    results = TestRegistry.search("attention")
    assert "attention_model" in results

    # Search in description
    assert "attention_model" in results

    # Search in tags
    assert "basic_model" in results

    # Not found
    assert "simple_func" not in results


# Test Utility Methods

def test_has():
    """Test checking if component exists."""
    @TestRegistry.register(name="exists")
    def func():
        pass

    assert TestRegistry.has("exists") is True
    assert TestRegistry.has("nonexistent") is False


def test_count():
    """Test counting registered components."""
    assert TestRegistry.count() == 0

    @TestRegistry.register(name="func1")
    def f1():
        pass

    assert TestRegistry.count() == 1

    @TestRegistry.register(name="func2")
    def f2():
        pass

    assert TestRegistry.count() == 2


def test_remove():
    """Test removing a component."""
    @TestRegistry.register(name="to_remove")
    def func():
        pass

    assert TestRegistry.has("to_remove")

    TestRegistry.remove("to_remove")

    assert not TestRegistry.has("to_remove")


def test_clear():
    """Test clearing all components."""
    @TestRegistry.register(name="func1")
    def f1():
        pass

    @TestRegistry.register(name="func2")
    def f2():
        pass

    assert TestRegistry.count() == 2

    TestRegistry.clear()

    assert TestRegistry.count() == 0


def test_get_metadata():
    """Test getting component metadata."""
    @TestRegistry.register(
        name="meta_test",
        description="Test metadata",
        tags=["test"],
        version="1.2.3"
    )
    def func():
        pass

    metadata = TestRegistry.get_metadata("meta_test")

    assert metadata["name"] == "meta_test"
    assert metadata["description"] == "Test metadata"
    assert metadata["tags"] == ["test"]
    assert metadata["version"] == "1.2.3"


def test_get_metadata_nonexistent():
    """Test error when getting metadata for nonexistent component."""
    with pytest.raises(KeyError, match="not found"):
        TestRegistry.get_metadata("nonexistent")


# Test RegistryMixin

def test_registry_mixin_get_from_registry():
    """Test RegistryMixin get_from_registry method."""
    @TestRegistry.register(name="mixin_test")
    def func():
        return "mixin_success"

    mixin = RegistryMixin()
    result = mixin.get_from_registry(TestRegistry, "mixin_test")

    assert callable(result)
    assert result() == "mixin_success"


def test_registry_mixin_list_from_registry():
    """Test RegistryMixin list_from_registry method."""
    @TestRegistry.register(name="func1")
    def f1():
        pass

    @TestRegistry.register(name="func2")
    def f2():
        pass

    mixin = RegistryMixin()
    names = mixin.list_from_registry(TestRegistry)

    assert "func1" in names
    assert "func2" in names


def test_registry_mixin_has_in_registry():
    """Test RegistryMixin has_in_registry method."""
    @TestRegistry.register(name="exists")
    def func():
        pass

    mixin = RegistryMixin()

    assert mixin.has_in_registry(TestRegistry, "exists") is True
    assert mixin.has_in_registry(TestRegistry, "nonexistent") is False


# Test Validation

def test_validation_called_on_registration():
    """Test that validate_component is called during registration."""
    # Register a callable - should pass validation
    @TestRegistry.register(name="valid")
    def valid_func():
        pass

    assert TestRegistry.has("valid")

    # Register a non-callable - should show warning but still register
    with pytest.warns(UserWarning, match="failed validation"):
        TestRegistry.register(name="invalid")(42)


# Test Edge Cases

def test_empty_registry():
    """Test operations on empty registry."""
    assert TestRegistry.count() == 0
    assert TestRegistry.list_names() == []
    assert TestRegistry.list() == []
    assert TestRegistry.filter_by_tag("any") == []
    assert TestRegistry.search("any") == []


def test_register_class():
    """Test registering a class."""
    @TestRegistry.register(name="test_class")
    class TestClass:
        def __init__(self, value):
            self.value = value

        def get_value(self):
            return self.value

    assert TestRegistry.has("test_class")

    # Get class and instantiate
    cls = TestRegistry.get("test_class")
    obj = cls(42)
    assert obj.get_value() == 42


def test_register_lambda():
    """Test registering a lambda function."""
    TestRegistry.register(name="test_lambda")(lambda x: x * 2)

    assert TestRegistry.has("test_lambda")
    func = TestRegistry.get("test_lambda")
    assert func(5) == 10


def test_multiple_registries_independent():
    """Test that multiple registry classes maintain independent registries."""
    class Registry1(BaseRegistry):
        registry = {}

        @classmethod
        def validate_component(cls, component):
            return callable(component)

    class Registry2(BaseRegistry):
        registry = {}

        @classmethod
        def validate_component(cls, component):
            return callable(component)

    @Registry1.register(name="func1")
    def f1():
        pass

    @Registry2.register(name="func2")
    def f2():
        pass

    assert Registry1.has("func1")
    assert not Registry1.has("func2")
    assert Registry2.has("func2")
    assert not Registry2.has("func1")

    # Cleanup
    Registry1.clear()
    Registry2.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
