"""Sequential file-driven orchestrator for the Morpion artifact pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .bootstrap_paths import MorpionBootstrapPaths
from .pipeline_artifacts import (
    MissingMorpionPipelineArtifactError,
    MorpionPipelineGenerationManifest,
    MorpionPipelineStageClaim,
    MorpionPipelineStageName,
    load_pipeline_stage_claim,
    load_pipeline_manifest,
)
from .pipeline_claims import load_active_pipeline_stage_claim, pipeline_stage_claim_is_expired
from .pipeline_stages import (
    _require_artifact_pipeline_mode,
    run_pipeline_dataset_stage,
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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


@dataclass(frozen=True, slots=True)
class MorpionPipelineWorkerResult:
    """Summary of one autonomous one-shot artifact-pipeline worker pass."""

    stage: MorpionPipelineStageName
    generation: int | None
    ran_stage: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class _LatestDatasetSummary:
    generation: int | None
    manifest_path: str | None
    created_at_utc: str | None
    rows: int | str | None
    rows_path: str | None


@dataclass(frozen=True, slots=True)
class _DatasetSelectionDiagnostics:
    latest_tree_generation: int | None
    pending_generations: tuple[int, ...]
    claimable_generations: tuple[int, ...]
    selected_generation: int | None
    selected_manifest: MorpionPipelineGenerationManifest | None


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


def _list_pipeline_generation_directories(paths: MorpionBootstrapPaths) -> tuple[int, ...]:
    """Return sorted generation directories whether or not a manifest exists."""
    if not paths.pipeline_dir.is_dir():
        return ()
    generations: list[int] = []
    for child in paths.pipeline_dir.iterdir():
        if not child.is_dir():
            continue
        match = _GENERATION_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        generations.append(int(match.group(1)))
    return tuple(sorted(generations))


def _timestamp_now_utc() -> str:
    """Return the current UTC timestamp as ISO-8601 with timezone."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _render_generation_list(generations: tuple[int, ...]) -> str:
    """Render a stable generation list for worker logs."""
    if not generations:
        return "none"
    return ",".join(str(generation) for generation in generations)


def _tolerant_status_updated_at(path: Path) -> str | None:
    """Load one stage-status updated timestamp without failing on old payloads."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload_mapping = cast("dict[str, object]", payload)
    updated_at_utc = payload_mapping.get("updated_at_utc")
    return updated_at_utc if isinstance(updated_at_utc, str) else None


def _render_optional_log_value(value: object | None) -> str:
    """Render one optional log field as a stable string."""
    if value is None:
        return "none"
    return str(value)


def _tolerant_dataset_rows(manifest: MorpionPipelineGenerationManifest) -> int | str:
    """Return one tolerant dataset row-count hint for logs."""
    rows = manifest.metadata.get("dataset_rows")
    if isinstance(rows, int):
        return rows
    rows = manifest.metadata.get("num_rows")
    if isinstance(rows, int):
        return rows
    return "unknown"


def _latest_dataset_summary(
    paths: MorpionBootstrapPaths,
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
) -> _LatestDatasetSummary:
    """Return a tolerant summary of the latest completed dataset artifact."""
    completed_generations = tuple(
        generation
        for generation, manifest in sorted(manifests.items())
        if manifest.dataset_status == "done"
    )
    if not completed_generations:
        return _LatestDatasetSummary(
            generation=None,
            manifest_path=None,
            created_at_utc=None,
            rows=None,
            rows_path=None,
        )

    generation = completed_generations[-1]
    manifest = manifests[generation]
    manifest_path = paths.pipeline_manifest_path_for_generation(generation)
    rows_path = (
        None
        if manifest.rows_path is None
        else paths.resolve_work_dir_path(manifest.rows_path)
    )
    status_updated_at = _tolerant_status_updated_at(
        paths.pipeline_dataset_status_path_for_generation(generation)
    )
    created_at_utc = status_updated_at
    if created_at_utc is None:
        metadata_created_at = manifest.metadata.get("dataset_completed_at_utc")
        created_at_utc = (
            metadata_created_at if isinstance(metadata_created_at, str) else manifest.created_at_utc
        )

    return _LatestDatasetSummary(
        generation=generation,
        manifest_path=str(manifest_path),
        created_at_utc=created_at_utc,
        rows=_tolerant_dataset_rows(manifest),
        rows_path=None if rows_path is None else str(rows_path),
    )


def _load_dataset_claim_for_diagnostics(
    claim_path: Path,
) -> MorpionPipelineStageClaim | None:
    """Load one dataset claim for diagnostics without raising on absence."""
    try:
        return load_pipeline_stage_claim(claim_path)
    except MissingMorpionPipelineArtifactError:
        return None


def _build_dataset_selection_diagnostics(
    paths: MorpionBootstrapPaths,
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
    *,
    now_unix_s: float | None = None,
) -> _DatasetSelectionDiagnostics:
    """Scan dataset candidates and emit stable skip diagnostics."""
    generation_directories = set(_list_pipeline_generation_directories(paths))
    manifest_generations = set(manifests)
    for generation in sorted(generation_directories - manifest_generations):
        LOGGER.info(
            "[pipeline] dataset_skip generation=%s reason=missing_generation_manifest manifest=%s",
            generation,
            paths.pipeline_manifest_path_for_generation(generation),
        )

    latest_tree_generation = max(
        (
            generation
            for generation, manifest in manifests.items()
            if manifest.tree_snapshot_path is not None
        ),
        default=None,
    )
    pending_generations: list[int] = []
    claimable_generations: list[int] = []
    selected_generation: int | None = None
    selected_manifest: MorpionPipelineGenerationManifest | None = None

    for generation in sorted(manifests):
        manifest = manifests[generation]
        manifest_path = paths.pipeline_manifest_path_for_generation(generation)
        if manifest.dataset_status == "done":
            LOGGER.info(
                "[pipeline] dataset_skip generation=%s reason=dataset_already_exists manifest=%s rows_path=%s",
                generation,
                manifest_path,
                _render_optional_log_value(manifest.rows_path),
            )
            continue
        if manifest.tree_snapshot_path is None:
            LOGGER.info(
                "[pipeline] dataset_skip generation=%s reason=no_tree_snapshots_available manifest=%s",
                generation,
                manifest_path,
            )
            continue
        if manifest.dataset_status not in {"not_started", "failed"}:
            LOGGER.info(
                "[pipeline] dataset_skip generation=%s reason=dataset_status_%s manifest=%s",
                generation,
                manifest.dataset_status,
                manifest_path,
            )
            continue

        pending_generations.append(generation)
        claim_path = paths.pipeline_dataset_claim_path_for_generation(generation)
        claim = _load_dataset_claim_for_diagnostics(claim_path)
        if claim is None:
            claimable_generations.append(generation)
            if selected_generation is None:
                selected_generation = generation
                selected_manifest = manifest
            continue

        if pipeline_stage_claim_is_expired(claim, now_unix_s=now_unix_s):
            LOGGER.info(
                "[pipeline] dataset_skip generation=%s reason=claim_expired_taking_over claim_path=%s owner=%s expires_at=%s",
                generation,
                claim_path,
                _render_optional_log_value(claim.owner),
                claim.expires_at_utc,
            )
            claimable_generations.append(generation)
            if selected_generation is None:
                selected_generation = generation
                selected_manifest = manifest
            continue

        LOGGER.info(
            "[pipeline] dataset_skip generation=%s reason=active_claim_exists claim_path=%s owner=%s expires_at=%s",
            generation,
            claim_path,
            _render_optional_log_value(claim.owner),
            claim.expires_at_utc,
        )

    return _DatasetSelectionDiagnostics(
        latest_tree_generation=latest_tree_generation,
        pending_generations=tuple(pending_generations),
        claimable_generations=tuple(claimable_generations),
        selected_generation=selected_generation,
        selected_manifest=selected_manifest,
    )



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


def select_next_dataset_generation(
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
) -> int | None:
    """Return the oldest generation whose dataset stage is pending."""
    for generation in sorted(manifests):
        if dataset_stage_is_pending(manifests[generation]):
            return generation
    return None


def select_next_training_generation(
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
) -> int | None:
    """Return the oldest generation whose training stage is pending."""
    for generation in sorted(manifests):
        if training_stage_is_pending(manifests[generation]):
            return generation
    return None


def select_next_claimable_dataset_generation(
    paths: MorpionBootstrapPaths,
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
    *,
    now_unix_s: float | None = None,
) -> int | None:
    """Return the oldest pending dataset generation without an active claim."""
    for generation in sorted(manifests):
        if not dataset_stage_is_pending(manifests[generation]):
            continue
        claim = load_active_pipeline_stage_claim(
            paths.pipeline_dataset_claim_path_for_generation(generation),
            now_unix_s=now_unix_s,
        )
        if claim is None:
            return generation
    return None


def select_next_claimable_training_generation(
    paths: MorpionBootstrapPaths,
    manifests: Mapping[int, MorpionPipelineGenerationManifest],
    *,
    now_unix_s: float | None = None,
) -> int | None:
    """Return the oldest pending training generation without an active claim."""
    for generation in sorted(manifests):
        if not training_stage_is_pending(manifests[generation]):
            continue
        claim = load_active_pipeline_stage_claim(
            paths.pipeline_training_claim_path_for_generation(generation),
            now_unix_s=now_unix_s,
        )
        if claim is None:
            return generation
    return None


def run_next_pipeline_dataset_stage_once(
    args: MorpionBootstrapArgs,
    *,
    claim_ttl_seconds: float = 3600.0,
    claim_owner: str | None = None,
    now_unix_s: float | None = None,
) -> MorpionPipelineWorkerResult:
    """Run the oldest claimable pending dataset generation once, if any."""
    invocation_started_at = time.perf_counter()
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    LOGGER.info(
        "[pipeline] dataset_worker_start work_dir=%s now=%s",
        paths.work_dir,
        _timestamp_now_utc(),
    )
    manifests = load_available_pipeline_manifests(paths)
    latest_dataset = _latest_dataset_summary(paths, manifests)
    LOGGER.info(
        "[pipeline] dataset_status latest_dataset_generation=%s latest_dataset_manifest=%s latest_dataset_created_at=%s latest_dataset_rows=%s",
        _render_optional_log_value(latest_dataset.generation),
        _render_optional_log_value(latest_dataset.manifest_path),
        _render_optional_log_value(latest_dataset.created_at_utc),
        _render_optional_log_value(latest_dataset.rows),
    )
    diagnostics = _build_dataset_selection_diagnostics(
        paths,
        manifests,
        now_unix_s=now_unix_s,
    )
    LOGGER.info(
        "[pipeline] dataset_selection_start latest_tree_generation=%s pending_generations=%s claimable_generations=%s",
        _render_optional_log_value(diagnostics.latest_tree_generation),
        _render_generation_list(diagnostics.pending_generations),
        _render_generation_list(diagnostics.claimable_generations),
    )
    generation = diagnostics.selected_generation
    if generation is None:
        LOGGER.info(
            "[pipeline] dataset_worker_idle reason=no_claimable_generation latest_tree_generation=%s latest_dataset_generation=%s",
            _render_optional_log_value(diagnostics.latest_tree_generation),
            _render_optional_log_value(latest_dataset.generation),
        )
        LOGGER.info(
            "[pipeline] dataset_worker_done action=idle generation=none elapsed=%.3fs",
            time.perf_counter() - invocation_started_at,
        )
        return MorpionPipelineWorkerResult(
            stage="dataset",
            generation=None,
            ran_stage=False,
            reason="no_pending_work",
        )

    assert diagnostics.selected_manifest is not None
    LOGGER.info(
        "[pipeline] dataset_selection_done selected_generation=%s reason=oldest_claimable tree_export=%s manifest=%s",
        generation,
        _render_optional_log_value(diagnostics.selected_manifest.tree_snapshot_path),
        paths.pipeline_manifest_path_for_generation(generation),
    )

    try:
        run_pipeline_dataset_stage(
            args,
            generation=generation,
            claim_ttl_seconds=claim_ttl_seconds,
            claim_owner=claim_owner,
        )
    except Exception:
        LOGGER.exception(
            "[pipeline] dataset_worker_done action=error generation=%s elapsed=%.3fs",
            generation,
            time.perf_counter() - invocation_started_at,
        )
        raise

    LOGGER.info(
        "[pipeline] dataset_worker_done action=exported generation=%s elapsed=%.3fs",
        generation,
        time.perf_counter() - invocation_started_at,
    )
    return MorpionPipelineWorkerResult(
        stage="dataset",
        generation=generation,
        ran_stage=True,
        reason=None,
    )


def run_next_pipeline_training_stage_once(
    args: MorpionBootstrapArgs,
    *,
    claim_ttl_seconds: float = 3600.0,
    claim_owner: str | None = None,
    now_unix_s: float | None = None,
) -> MorpionPipelineWorkerResult:
    """Run the oldest claimable pending training generation once, if any."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    manifests = load_available_pipeline_manifests(paths)
    generation = select_next_claimable_training_generation(
        paths,
        manifests,
        now_unix_s=now_unix_s,
    )
    if generation is None:
        return MorpionPipelineWorkerResult(
            stage="training",
            generation=None,
            ran_stage=False,
            reason="no_pending_work",
        )

    run_pipeline_training_stage(
        args,
        generation=generation,
        claim_ttl_seconds=claim_ttl_seconds,
        claim_owner=claim_owner,
    )
    return MorpionPipelineWorkerResult(
        stage="training",
        generation=generation,
        ran_stage=True,
        reason=None,
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
    "MorpionPipelineWorkerResult",
    "dataset_stage_is_pending",
    "list_pipeline_manifest_generations",
    "load_available_pipeline_manifests",
    "run_morpion_artifact_pipeline_once",
    "run_next_pipeline_dataset_stage_once",
    "run_next_pipeline_training_stage_once",
    "select_next_claimable_dataset_generation",
    "select_next_claimable_training_generation",
    "select_next_dataset_generation",
    "select_next_training_generation",
    "training_stage_is_pending",
]
