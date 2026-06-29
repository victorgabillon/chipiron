"""Memory-diagnostics glue for Morpion bootstrap cycles."""

from __future__ import annotations

import gc
from typing import Protocol

from .profiling.memory_diagnostics import MemoryDiagnostics, MemoryDiagnosticsConfig


class MemoryDiagnosticsArgs(Protocol):
    """Small argument surface needed to configure memory diagnostics."""

    @property
    def memory_diagnostics(self) -> bool:
        """Whether memory diagnostics are enabled."""
        ...

    @property
    def memory_diagnostics_gc_growth(self) -> bool:
        """Whether to log GC growth diagnostics."""
        ...

    @property
    def memory_diagnostics_tracemalloc(self) -> bool:
        """Whether to enable tracemalloc diagnostics."""
        ...

    @property
    def memory_diagnostics_torch_tensors(self) -> bool:
        """Whether to report live torch tensor diagnostics."""
        ...

    @property
    def memory_diagnostics_referrers(self) -> bool:
        """Whether to inspect selected referrer graphs."""
        ...

    @property
    def memory_diagnostics_referrer_type_patterns(self) -> tuple[str, ...]:
        """Return type-name patterns selected for referrer diagnostics."""
        ...

    @property
    def memory_diagnostics_referrer_max_objects_per_type(self) -> int:
        """Return the per-type object cap for referrer diagnostics."""
        ...

    @property
    def memory_diagnostics_referrer_max_depth(self) -> int:
        """Return the maximum referrer traversal depth."""
        ...

    @property
    def memory_diagnostics_top_n(self) -> int:
        """Return the number of top memory records to log."""
        ...


def memory_diagnostics_config_from_args(
    args: MemoryDiagnosticsArgs,
) -> MemoryDiagnosticsConfig:
    """Build memory diagnostics config from bootstrap args."""
    return MemoryDiagnosticsConfig(
        enabled=args.memory_diagnostics,
        gc_growth=args.memory_diagnostics_gc_growth,
        tracemalloc=args.memory_diagnostics_tracemalloc,
        torch_tensors=args.memory_diagnostics_torch_tensors,
        referrers=args.memory_diagnostics_referrers,
        referrer_type_patterns=args.memory_diagnostics_referrer_type_patterns,
        referrer_max_objects_per_type=(
            args.memory_diagnostics_referrer_max_objects_per_type
        ),
        referrer_max_depth=args.memory_diagnostics_referrer_max_depth,
        referrer_top_n=args.memory_diagnostics_top_n,
        top_n=args.memory_diagnostics_top_n,
    )


def log_after_cycle_gc(
    memory: MemoryDiagnostics,
    *,
    tag: str = "after_cycle_gc",
) -> None:
    """Collect garbage and log one post-cycle memory checkpoint."""
    gc.collect()
    memory.log(tag)


__all__ = ["log_after_cycle_gc", "memory_diagnostics_config_from_args"]
