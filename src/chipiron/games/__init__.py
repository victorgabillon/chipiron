"""Module for managing games."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["GamePlayingStatus", "MatchManager", "ObservableGamePlayingStatus"]

_EXPORT_MODULES = {
    "GamePlayingStatus": ".domain.game.game_playing_status",
    "MatchManager": ".domain.match.match_manager",
    "ObservableGamePlayingStatus": ".domain.game.observable_game_playing_status",
}


def __getattr__(name: str) -> Any:
    """Resolve public game exports without importing the full runtime eagerly."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
