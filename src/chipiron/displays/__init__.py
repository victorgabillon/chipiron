"""Public exports for the displays package."""

from __future__ import annotations

from typing import Any

__all__ = ["MainWindow"]


def __getattr__(name: str) -> Any:
    """Load GUI exports lazily so display helpers can be imported independently."""
    if name == "MainWindow":
        from .gui import MainWindow

        return MainWindow
    raise AttributeError(name)
