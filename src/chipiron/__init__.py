"""Top-level Chipiron package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["game", "player", "set_seeds", "tool"]


def __getattr__(name: str) -> Any:
    """Load legacy top-level aliases without importing the whole app eagerly."""
    value: Any
    if name == "game":
        value = import_module(".games", __name__)
    elif name == "player":
        value = import_module(".players", __name__)
    elif name == "tool":
        value = import_module(".utils", __name__)
    elif name == "set_seeds":
        from .utils.my_random import set_seeds as value
    else:
        raise AttributeError(name)

    globals()[name] = value
    return value
