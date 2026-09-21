"""Init file for scripts module."""

from __future__ import annotations

__all__ = [
    "IScript",
    "Script",
    "ScriptType",
]


def __getattr__(name: str) -> object:
    """Load public script symbols lazily to avoid parser imports for enum access."""
    if name == "IScript":
        from .iscript import IScript

        return IScript
    if name == "Script":
        from .script import Script

        return Script
    if name == "ScriptType":
        from .script_type import ScriptType

        return ScriptType
    raise AttributeError(name)
