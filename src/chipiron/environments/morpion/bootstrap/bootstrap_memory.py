"""Memory-diagnostics glue for Morpion bootstrap cycles."""

from __future__ import annotations

import gc
from typing import Protocol

from .memory_diagnostics import MemoryDiagnostics, MemoryDiagnosticsConfig


class MemoryDiagnosticsArgs(Protocol):
    """Small argument surface needed to configure memory diagnostics."""

    @property
    def memory_diagnostics(self) -> bool:
        ...

    @property
    def memory_diagnostics_gc_growth(self) -> bool:
        ...

    @property
    def memory_diagnostics_tracemalloc(self) -> bool:
        ...

    @property
    def memory_diagnostics_torch_tensors(self) -> bool:
        ...

    @property
    def memory_diagnostics_referrers(self) -> bool:
        ...

    @property
    def memory_diagnostics_referrer_type_patterns(self) -> tuple[str, ...]:
        ...

    @property
    def memory_diagnostics_referrer_max_objects_per_type(self) -> int:
        ...

    @property
    def memory_diagnostics_referrer_max_depth(self) -> int:
        ...

    @property
    def memory_diagnostics_top_n(self) -> int:
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
