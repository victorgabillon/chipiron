"""Search-runner protocol for Morpion bootstrap workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .control import MorpionBootstrapEffectiveRuntimeConfig


class MorpionSearchRunner(Protocol):
    """Thin search-runner boundary shared by bootstrap workflows."""

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Load existing search state or initialize a fresh one."""
        ...

    def grow(self, max_growth_steps: int) -> None:
        """Grow the underlying search state by a bounded number of steps."""
        ...

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Persist a training-grade tree snapshot to ``output_path``."""
        ...

    def current_tree_size(self) -> int:
        """Return the current size of the search tree."""
        ...


__all__ = ["MorpionSearchRunner"]