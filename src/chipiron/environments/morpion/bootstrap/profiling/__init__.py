"""Stable profiling helpers for Morpion bootstrap diagnostics."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORT_MODULES = {
    "DEFAULT_REFERRER_TYPE_PATTERNS": "memory_diagnostics",
    "DeepSizeStats": "recursive_memory",
    "MemoryDiagnostics": "memory_diagnostics",
    "MemoryDiagnosticsConfig": "memory_diagnostics",
    "build_recursive_profile_context": "recursive_memory",
    "deep_size": "recursive_memory",
    "log_growth_recursive_memory_profile": "recursive_memory",
    "log_growth_runtime_memory_profile": "growth_memory",
}


def __getattr__(name: str) -> object:
    """Load profiling helpers on first access."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_EXPORT_MODULES)
