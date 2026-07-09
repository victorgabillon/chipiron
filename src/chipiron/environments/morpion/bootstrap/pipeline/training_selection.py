"""Training-selection guards shared by Morpion pipeline entrypoints."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.pipeline.training_recovery import (
    DEFAULT_STALE_GRACE_SECONDS,
    inspect_training_state_for_recovery,
)
from chipiron.environments.morpion.bootstrap.pipeline_claims import (
    load_active_pipeline_stage_claim,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.pipeline.cursors import (
        MorpionTrainingLowerBound,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
    )


def training_completed_cursor_generation(
    lower_bound: MorpionTrainingLowerBound,
) -> int:
    """Return the completed cursor generation as a concrete skip boundary."""
    return (
        0
        if lower_bound.cursor_completed_generation is None
        else lower_bound.cursor_completed_generation
    )


def manifest_has_auto_recovery(
    manifest: MorpionPipelineGenerationManifest,
) -> bool:
    """Return whether one manifest records stale-training auto-recovery."""
    return isinstance(manifest.metadata.get("auto_recovery"), MappingABC)


def started_cursor_generation_is_safely_reclaimable(
    paths: MorpionBootstrapPaths,
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
    lower_bound: MorpionTrainingLowerBound,
    *,
    now_unix_s: float | None,
    stale_grace_seconds: int = DEFAULT_STALE_GRACE_SECONDS,
) -> bool:
    """Return whether the started cursor generation should bypass the skip bound."""
    generation = lower_bound.cursor_started_generation
    if generation is None or lower_bound.generation != generation:
        return False
    completed_generation = training_completed_cursor_generation(lower_bound)
    if completed_generation >= generation:
        return False
    manifest = manifests.get(generation)
    if manifest is None:
        return False
    if manifest.dataset_status != "done":
        return False
    if manifest.training_status != "not_started":
        return False
    state = inspect_training_state_for_recovery(
        paths.pipeline_generation_dir_for_generation(generation),
        now_utc=(
            datetime.now(UTC)
            if now_unix_s is None
            else datetime.fromtimestamp(now_unix_s, tz=UTC)
        ),
        stale_grace_seconds=stale_grace_seconds,
    )
    if state.evaluator_results_count not in (0, None):
        return False
    if state.status_file_status not in (None, "not_started", "training"):
        return False
    claim = load_active_pipeline_stage_claim(
        paths.pipeline_training_claim_path_for_generation(generation),
        now_unix_s=now_unix_s,
    )
    return claim is None
