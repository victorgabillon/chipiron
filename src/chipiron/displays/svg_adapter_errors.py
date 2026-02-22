"""Error classes for SVG adapter construction and payload validation."""

from typing import Any


class SvgAdapterError(Exception):
    """Base class for SVG adapter related errors."""


class InvalidSvgAdapterPayloadTypeError(TypeError, SvgAdapterError):
    """Raised when an adapter receives a payload of unexpected type."""

    def __init__(
        self,
        *,
        adapter_name: str,
        expected_type: type[Any],
        actual_value: Any,
    ) -> None:
        """Initialize the error with details about the type mismatch."""
        super().__init__(
            f"{adapter_name} expected {expected_type.__name__}, got "
            f"{type(actual_value).__name__} ({actual_value!r})"
        )


class UnregisteredSvgAdapterError(ValueError, SvgAdapterError):
    """Raised when no SVG adapter exists for a game kind."""

    def __init__(self, *, game_kind: Any) -> None:
        """Initialize the error with details about the unregistered game kind."""
        super().__init__(f"No SvgGameAdapter registered for game_kind={game_kind!r}")
