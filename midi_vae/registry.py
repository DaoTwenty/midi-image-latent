"""Component registry for dynamically registering and retrieving implementations.

Usage:
    @ComponentRegistry.register('channel_strategy', 'velocity_only')
    class VelocityOnlyStrategy(ChannelStrategy):
        ...

    # Later:
    cls = ComponentRegistry.get('channel_strategy', 'velocity_only')
    instance = cls(config)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """Central registry for all swappable components.

    Components are registered by (component_type, name) pairs.
    New implementations just need the @register decorator — zero pipeline changes.
    """

    _registry: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, component_type: str, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a component class.

        Args:
            component_type: Category of component (e.g., 'channel_strategy', 'vae', 'metric').
            name: Unique name within the component type.

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If (component_type, name) is already registered.
        """

        def decorator(klass: type[T]) -> type[T]:
            if component_type not in cls._registry:
                cls._registry[component_type] = {}
            if name in cls._registry[component_type]:
                raise ValueError(
                    f"Component '{name}' already registered for type '{component_type}': "
                    f"{cls._registry[component_type][name]}"
                )
            cls._registry[component_type][name] = klass
            logger.debug(f"Registered {component_type}/{name} -> {klass.__name__}")
            return klass

        return decorator

    @classmethod
    def get(cls, component_type: str, name: str) -> type:
        """Retrieve a registered component class.

        Args:
            component_type: Category of component.
            name: Registered name.

        Returns:
            The registered class.

        Raises:
            KeyError: If the component is not registered.
        """
        if component_type not in cls._registry:
            raise KeyError(
                f"No components registered for type '{component_type}'. "
                f"Available types: {list(cls._registry.keys())}"
            )
        if name not in cls._registry[component_type]:
            raise KeyError(
                f"Component '{name}' not found for type '{component_type}'. "
                f"Available: {list(cls._registry[component_type].keys())}"
            )
        return cls._registry[component_type][name]

    @classmethod
    def list_components(cls, component_type: str | None = None) -> dict[str, list[str]]:
        """List all registered components.

        Args:
            component_type: If given, list only this type. Otherwise list all.

        Returns:
            Dict mapping component_type -> list of registered names.
        """
        if component_type is not None:
            if component_type not in cls._registry:
                return {component_type: []}
            return {component_type: list(cls._registry[component_type].keys())}
        return {ct: list(names.keys()) for ct, names in cls._registry.items()}

    @classmethod
    def create(cls, component_type: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create an instance of a registered component.

        Args:
            component_type: Category of component.
            name: Registered name.
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.

        Returns:
            An instance of the registered class.
        """
        klass = cls.get(component_type, name)
        return klass(*args, **kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Primarily for testing."""
        cls._registry.clear()
