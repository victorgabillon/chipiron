"""Compatibility imports for Morpion recursive memory profiling."""

from __future__ import annotations

from .recursive.context import build_recursive_profile_context
from .recursive.deep_size import DeepSizeStats, deep_size
from .recursive.rendering import gc_shallow_size_summary
from .recursive.runner import log_growth_recursive_memory_profile

__all__ = [
    "DeepSizeStats",
    "build_recursive_profile_context",
    "deep_size",
    "gc_shallow_size_summary",
    "log_growth_recursive_memory_profile",
]
