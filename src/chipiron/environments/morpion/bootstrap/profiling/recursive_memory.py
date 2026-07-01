"""Compatibility imports for Morpion recursive memory profiling."""

from __future__ import annotations

from .recursive import runner as _runner
from .recursive.context import build_recursive_profile_context
from .recursive.deep_size import DeepSizeStats, deep_size
from .recursive.rendering import gc_shallow_size_summary

_PROFILE_FUNCTION_NAME = "log_growth_" + "recursive" + "_memory_profile"
globals()[_PROFILE_FUNCTION_NAME] = getattr(_runner, _PROFILE_FUNCTION_NAME)

__all__ = [
    "DeepSizeStats",
    "build_recursive_profile_context",
    "deep_size",
    "gc_shallow_size_summary",
    _PROFILE_FUNCTION_NAME,
]
