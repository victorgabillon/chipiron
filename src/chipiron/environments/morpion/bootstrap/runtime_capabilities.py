"""Typed operations at Morpion's existing live-runtime boundaries.

These structural capabilities describe operations, not ownership. Growth retains
the live-tree mutation and commit authority. Projection repair consumes none of
them, and legacy callers opt into only the operations they actually use.
"""

from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeGuard,
    cast,
    get_protocol_members,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Any

    from anemone.tree_exploration import ReevaluationReport

    from .control import MorpionBootstrapEffectiveRuntimeConfig
    from .pipeline_artifacts import MorpionReevaluationPatch


@runtime_checkable
class SearchStateRuntime(Protocol):
    """Restore and inspect the private search state used during activation."""

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Restore using the existing checkpoint and evaluator rules."""
        ...

    def current_tree_size(self) -> int:
        """Return the live tree size for the ordinary commit metadata."""
        ...


@runtime_checkable
class CheckpointWriter(Protocol):
    """Persist the owner's current runtime without committing its publication."""

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Write the existing checkpoint format."""
        ...


@runtime_checkable
class RuntimeDiscard(Protocol):
    """Release private live state at restore and failure boundaries."""

    def discard_uncommitted_runtime(self) -> None:
        """Discard RAM without changing the committed authority."""
        ...


@runtime_checkable
class ReevaluationPatchRuntime(Protocol):
    """Apply a previously validated patch under Growth ownership."""

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int | None:
        """Return the applied count, retaining the legacy None convention."""
        ...


@runtime_checkable
class ShardedTrainingExportRuntime(Protocol):
    """Retain the public additive training-export adapter."""

    def export_sharded_training_tree_snapshot(
        self, output_dir: str | Path, *, generation: int
    ) -> str | Path:
        """Export training rows in the selected generation."""
        ...


@runtime_checkable
class ScientificTrainingExportRuntime(Protocol):
    """Export scientific v2 values from a quiescent runtime."""

    def export_scientific_training_snapshot(
        self, output_dir: str | Path, *, generation: int, producer: Mapping[str, str]
    ) -> str | Path:
        """Export using the unchanged scientific identity and target contract."""
        ...


@runtime_checkable
class ClosedScientificExportRuntime(Protocol):
    """Export the closed scientific snapshot paired with a checkpoint."""

    def export_closed_scientific_snapshot(
        self,
        destination: str | Path,
        *,
        generation: int,
        producer: Mapping[str, str],
        committed_source: Path | None = None,
        _validation_result: dict[str, Any] | None = None,
        previous_producer: Mapping[str, str] | None = None,
        encoding: str | None = None,
    ) -> str | Path:
        """Reuse committed validation evidence without changing persisted schemas."""
        ...


@runtime_checkable
class BoundedGrowthRuntime(SearchStateRuntime, Protocol):
    """Grow with both the historical step bound and the opt-in node bound."""

    def grow(self, max_growth_steps: int) -> None:
        """Grow by at most the configured number of steps."""
        ...

    def set_unique_node_budget(self, maximum: int) -> None:
        """Enforce the configured maximum unique live nodes."""
        ...


@runtime_checkable
class EvaluatorRefreshRuntime(Protocol):
    """Validate certificates and explicitly refresh an accepted evaluator."""

    def validate_model_refresh_certificates(self) -> None:
        """Reject unsupported or contradictory proofs before refreshing."""
        ...

    def refresh_model_estimates(self, model_bundle_path: Path) -> ReevaluationReport:
        """Refresh estimates using the existing bounded evaluator implementation."""
        ...


@runtime_checkable
class ContinualGrowthRuntime(  # pylint: disable=too-many-ancestors
    BoundedGrowthRuntime,
    CheckpointWriter,
    RuntimeDiscard,
    ReevaluationPatchRuntime,
    ClosedScientificExportRuntime,
    Protocol,
):
    """Compose operation contracts, with no shared implementation inheritance."""


@runtime_checkable
class ModelActivationRuntime(  # pylint: disable=too-many-ancestors
    SearchStateRuntime,
    CheckpointWriter,
    RuntimeDiscard,
    ReevaluationPatchRuntime,
    ClosedScientificExportRuntime,
    EvaluatorRefreshRuntime,
    Protocol,
):
    """Private runtime used by the existing atomic model activation transaction."""


def supports_runtime_capability[T](
    runtime: object, capability: Callable[..., T]
) -> TypeGuard[T]:
    """Narrow external objects while rejecting missing or noncallable methods.

    Preserve legacy dynamically delegated hooks: runtime protocol isinstance
    checks use static attribute lookup and would reject those adapters. Check
    every required callable here; static contracts check signatures and the
    existing consumers still validate external return values.
    """
    return all(
        callable(getattr(runtime, member, None))
        for member in get_protocol_members(cast("type[T]", capability))
    )
