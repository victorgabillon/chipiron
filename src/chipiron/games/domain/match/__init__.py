"""Module to manage matches."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MatchConfigTag",
    "MatchSettingsArgs",
    "SoloMatchSchedule",
    "TwoRoleMatchSchedule",
    "create_match_manager",
]

_EXPORT_MODULES = {
    "MatchConfigTag": ".match_tag",
    "MatchSettingsArgs": ".match_settings_args",
    "SoloMatchSchedule": ".match_role_schedule",
    "TwoRoleMatchSchedule": ".match_role_schedule",
    "create_match_manager": ".match_factories",
}


def __getattr__(name: str) -> Any:
    """Resolve public match exports without importing factories eagerly."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
