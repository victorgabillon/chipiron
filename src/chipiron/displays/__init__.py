"""Public exports for the displays package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gui import MainWindow

__all__ = ["MainWindow"]


def __getattr__(name: str) -> Any:
    """Load GUI exports lazily so display helpers can be imported independently."""
    if name == "MainWindow":
        from .gui import MainWindow  # pylint: disable=import-outside-toplevel

        return MainWindow
    raise AttributeError(name)
