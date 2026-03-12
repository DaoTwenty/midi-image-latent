"""Tests for ComponentRegistry in midi_vae/registry.py.

Verifies registration, retrieval, duplicate handling, error handling,
and listing of registered components.
"""

from __future__ import annotations

import pytest

from midi_vae.registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Helpers / base classes for test stubs
# ---------------------------------------------------------------------------


class DummyBase:
    """Minimal base class for test stubs."""

    def __init__(self, x: int = 0) -> None:
        self.x = x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before each test to avoid cross-test contamination."""
    # Save existing entries, clear, yield, then restore.
    original = {k: dict(v) for k, v in ComponentRegistry._registry.items()}
    ComponentRegistry.clear()
    yield
    # Restore state after each test so other tests using real components still work.
    ComponentRegistry._registry.clear()
    ComponentRegistry._registry.update(original)


# ---------------------------------------------------------------------------
# Register tests
# ---------------------------------------------------------------------------


class TestRegister:
    """Tests for ComponentRegistry.register decorator."""

    def test_register_creates_entry(self) -> None:
        """Registering a class creates an entry in the registry."""
        @ComponentRegistry.register("widget", "foo")
        class FooWidget(DummyBase):
            pass

        assert ComponentRegistry.get("widget", "foo") is FooWidget

    def test_register_returns_class_unchanged(self) -> None:
        """The decorator returns the original class."""
        @ComponentRegistry.register("widget", "bar")
        class BarWidget(DummyBase):
            pass

        assert BarWidget.__name__ == "BarWidget"

    def test_register_multiple_names_same_type(self) -> None:
        """Multiple names can be registered under the same type."""
        @ComponentRegistry.register("widget", "alpha")
        class AlphaWidget(DummyBase):
            pass

        @ComponentRegistry.register("widget", "beta")
        class BetaWidget(DummyBase):
            pass

        assert ComponentRegistry.get("widget", "alpha") is AlphaWidget
        assert ComponentRegistry.get("widget", "beta") is BetaWidget

    def test_register_multiple_types(self) -> None:
        """Components can be registered under different types."""
        @ComponentRegistry.register("type_a", "x")
        class X(DummyBase):
            pass

        @ComponentRegistry.register("type_b", "x")
        class Y(DummyBase):
            pass

        assert ComponentRegistry.get("type_a", "x") is X
        assert ComponentRegistry.get("type_b", "x") is Y

    def test_register_duplicate_raises(self) -> None:
        """Registering the same (type, name) twice raises ValueError."""
        @ComponentRegistry.register("widget", "dup")
        class First(DummyBase):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @ComponentRegistry.register("widget", "dup")
            class Second(DummyBase):
                pass


# ---------------------------------------------------------------------------
# Get tests
# ---------------------------------------------------------------------------


class TestGet:
    """Tests for ComponentRegistry.get method."""

    def test_get_returns_registered_class(self) -> None:
        """get() returns the class that was registered."""
        @ComponentRegistry.register("gadget", "myname")
        class MyGadget(DummyBase):
            pass

        result = ComponentRegistry.get("gadget", "myname")
        assert result is MyGadget

    def test_get_unknown_type_raises(self) -> None:
        """get() raises KeyError for unknown component type."""
        with pytest.raises(KeyError, match="no_such_type"):
            ComponentRegistry.get("no_such_type", "anything")

    def test_get_unknown_name_raises(self) -> None:
        """get() raises KeyError for unknown name within a known type."""
        @ComponentRegistry.register("known_type", "known_name")
        class KnownClass(DummyBase):
            pass

        with pytest.raises(KeyError, match="no_such_name"):
            ComponentRegistry.get("known_type", "no_such_name")

    def test_get_error_lists_available_names(self) -> None:
        """KeyError message includes available names for helpful debugging."""
        @ComponentRegistry.register("hint_type", "alpha")
        class Alpha(DummyBase):
            pass

        @ComponentRegistry.register("hint_type", "beta")
        class Beta(DummyBase):
            pass

        with pytest.raises(KeyError) as exc_info:
            ComponentRegistry.get("hint_type", "gamma")
        # The error should mention available options
        assert "alpha" in str(exc_info.value) or "beta" in str(exc_info.value)


# ---------------------------------------------------------------------------
# List / create tests
# ---------------------------------------------------------------------------


class TestListComponents:
    """Tests for ComponentRegistry.list_components and create."""

    def test_list_components_all(self) -> None:
        """list_components() with no args returns all registered types."""
        @ComponentRegistry.register("t1", "n1")
        class C1(DummyBase):
            pass

        @ComponentRegistry.register("t2", "n2")
        class C2(DummyBase):
            pass

        result = ComponentRegistry.list_components()
        assert "t1" in result
        assert "t2" in result
        assert "n1" in result["t1"]
        assert "n2" in result["t2"]

    def test_list_components_single_type(self) -> None:
        """list_components(type) returns only that type's entries."""
        @ComponentRegistry.register("specific_type", "item_a")
        class A(DummyBase):
            pass

        @ComponentRegistry.register("specific_type", "item_b")
        class B(DummyBase):
            pass

        result = ComponentRegistry.list_components("specific_type")
        assert "specific_type" in result
        names = result["specific_type"]
        assert "item_a" in names
        assert "item_b" in names

    def test_list_components_unknown_type_returns_empty(self) -> None:
        """list_components() for an unknown type returns empty list."""
        result = ComponentRegistry.list_components("nonexistent_type")
        assert result == {"nonexistent_type": []}

    def test_create_instantiates_class(self) -> None:
        """create() returns an instance of the registered class."""
        @ComponentRegistry.register("factory", "myfactory")
        class MyFactory(DummyBase):
            pass

        instance = ComponentRegistry.create("factory", "myfactory", x=42)
        assert isinstance(instance, MyFactory)
        assert instance.x == 42

    def test_create_passes_args(self) -> None:
        """create() correctly forwards positional and keyword arguments."""
        @ComponentRegistry.register("factory", "argfactory")
        class ArgFactory:
            def __init__(self, a: int, b: str = "default") -> None:
                self.a = a
                self.b = b

        instance = ComponentRegistry.create("factory", "argfactory", 10, b="hello")
        assert instance.a == 10
        assert instance.b == "hello"


# ---------------------------------------------------------------------------
# Clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for ComponentRegistry.clear method."""

    def test_clear_removes_all_registrations(self) -> None:
        """clear() removes all registered components."""
        @ComponentRegistry.register("temp_type", "temp_name")
        class TempClass(DummyBase):
            pass

        ComponentRegistry.clear()
        assert ComponentRegistry.list_components() == {}

    def test_can_reregister_after_clear(self) -> None:
        """After clear(), the same (type, name) can be registered again."""
        @ComponentRegistry.register("reuse_type", "reuse_name")
        class First(DummyBase):
            pass

        ComponentRegistry.clear()

        @ComponentRegistry.register("reuse_type", "reuse_name")
        class Second(DummyBase):
            pass

        assert ComponentRegistry.get("reuse_type", "reuse_name") is Second
