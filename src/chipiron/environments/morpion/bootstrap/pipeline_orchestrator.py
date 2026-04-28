"""Sequential file-driven orchestrator for the Morpion artifact pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .bootstrap_paths import MorpionBootstrapPaths
from .pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
)
from .pipeline_stages import (
    _require_artifact_pipeline_mode,
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .run_state import MorpionBootstrapRunState
    from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)

_GENERATION_DIR_RE = re.compile(r"^generation_(\d{6})$")


def _negative_max_growth_cycles_error() -> ValueError:
    """Build the stable invalid growth-cycle count error."""
    return ValueError("max_growth_cycles must be >= 0")


@dataclass(frozen=True, slots=True)
class MorpionPipelineOrchestratorResult:
    """Summary of one sequential artifact-pipeline orchestration pass."""

    growth_run_state: MorpionBootstrapRunState | None
    dataset_generations: tuple[int, ...]
    training_generations: tuple[int, ...]


def list_pipeline_manifest_generations(paths: MorpionBootstrapPaths) -> tuple[int, ...]:
    """Return sorted generations that have a valid generation dir and manifest."""
    if not paths.pipeline_dir.is_dir():
        return ()

    generations: list[int] = []
    for child in paths.pipeline_dir.iterdir():
        if not child.is_dir():
            continue
        match = _GENERATION_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        manifest_path = child / "manifest.json"
        if manifest_path.is_file():
            generations.append(int(match.group(1)))
    return tuple(sorted(generations))


def load_available_pipeline_manifests(
    paths: MorpionBootstrapPaths,
) -> dict[int, MorpionPipelineGenerationManifest]:
    """Load all valid persisted manifests under the pipeline directory."""
    return {
        generation: load_pipeline_manifest(
            paths.pipeline_manifest_path_for_generation(generation)
        )
        for generation in list_pipeline_manifest_generations(paths)
    }


def dataset_stage_is_pending(manifest: MorpionPipelineGenerationManifest) -> bool:
    """Return whether one manifest is ready for dataset extraction."""
    return (
        manifest.tree_snapshot_path is not None
        and manifest.dataset_status in {"not_started", "failed"}
    )


def training_stage_is_pending(manifest: MorpionPipelineGenerationManifest) -> bool:
    """Return whether one manifest is ready for model training."""
    return (
        manifest.rows_path is not None
        and manifest.dataset_status == "done"
        and manifest.training_status in {"not_started", "failed"}
    )


def run_morpion_artifact_pipeline_once(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_growth_cycles: int = 1,
) -> MorpionPipelineOrchestratorResult:
    """Run growth, dataset, and training stages sequentially via artifacts."""
    _require_artifact_pipeline_mode(args)
    if max_growth_cycles < 0:
        raise _negative_max_growth_cycles_error()
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    existing_generations = list_pipeline_manifest_generations(paths)
    LOGGER.info(
        "[pipeline] orchestrator_start max_growth_cycles=%s existing_generations=%s",
        max_growth_cycles,
        existing_generations,
    )

    growth_run_state: MorpionBootstrapRunState | None = None
    if max_growth_cycles > 0:
        growth_run_state = run_pipeline_growth_stage(
            args,
            runner,
            max_cycles=max_growth_cycles,
        )
    else:
        LOGGER.info("[pipeline] orchestrator_growth_skipped reason=max_growth_cycles_zero")

    manifests = load_available_pipeline_manifests(paths)

    dataset_generations: list[int] = []
    for generation in sorted(manifests):
        manifest = manifests[generation]
        if not dataset_stage_is_pending(manifest):
            continue
        LOGGER.info("[pipeline] orchestrator_dataset_dispatch generation=%s", generation)
        run_pipeline_dataset_stage(args, generation=generation)
        dataset_generations.append(generation)

    manifests = load_available_pipeline_manifests(paths)

    training_generations: list[int] = []
    for generation in sorted(manifests):
        manifest = manifests[generation]
        if not training_stage_is_pending(manifest):
            continue
        LOGGER.info("[pipeline] orchestrator_training_dispatch generation=%s", generation)
        run_pipeline_training_stage(args, generation=generation)
        training_generations.append(generation)

    LOGGER.info(
        "[pipeline] orchestrator_done dataset_generations=%s training_generations=%s",
        tuple(dataset_generations),
        tuple(training_generations),
    )
    return MorpionPipelineOrchestratorResult(
        growth_run_state=growth_run_state,
        dataset_generations=tuple(dataset_generations),
        training_generations=tuple(training_generations),
    )


__all__ = [
    "MorpionPipelineOrchestratorResult",
    "dataset_stage_is_pending",
    "list_pipeline_manifest_generations",
    "load_available_pipeline_manifests",
    "run_morpion_artifact_pipeline_once",
    "training_stage_is_pending",
]
