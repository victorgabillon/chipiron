"""Phase 3 artifact-pipeline stage entrypoints for Morpion bootstrap."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRowsSource,
    load_morpion_supervised_rows,
    morpion_supervised_rows_source_from_path,
    save_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    morpion_streaming_split_policy,
)

from .bootstrap_errors import MissingSavedBootstrapArtifactError
from .bootstrap_memory import log_after_cycle_gc, memory_diagnostics_config_from_args
from .bootstrap_paths import MorpionBootstrapPaths
from .config import (
    MorpionBootstrapConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    load_bootstrap_config,
    save_bootstrap_config,
    validate_bootstrap_config_change,
)
from .control import (
    MorpionBootstrapControl,
    apply_control_to_args,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from .cycle_dataset import (
    export_training_snapshot_for_generation as _export_training_snapshot_for_generation,
)
from .cycle_dataset import (
    load_training_snapshot_for_generation as _load_training_snapshot_for_generation,
)
from .cycle_dataset import (
    streaming_rows_from_training_snapshot as _streaming_rows_from_training_snapshot,
)
from .cycle_metadata import build_bootstrap_event
from .cycle_metadata import build_event_metadata as _build_event_metadata
from .cycle_metadata import next_metadata as _next_metadata
from .cycle_metadata import pipeline_metadata as _pipeline_metadata
from .cycle_metadata import record_no_save_cycle_event as _record_no_save_cycle_event
from .cycle_metadata import with_config_hash_metadata as _with_config_hash_metadata
from .cycle_pipeline_manifest import (
    write_pipeline_manifest_for_generation as _write_pipeline_manifest_for_generation,
)
from .cycle_runtime import (
    GROWTH_BUDGET_ALREADY_EXHAUSTED_STATUS,
    GROWTH_STATUS_METADATA_KEY,
    ResolvedActiveMorpionModelBundle,
)
from .cycle_runtime import (
    build_growth_budget_exhausted_run_state as _build_growth_budget_exhausted_run_state,
)
from .cycle_runtime import build_no_save_run_state as _build_no_save_run_state
from .cycle_runtime import current_tree_branch_count as _current_tree_branch_count
from .cycle_runtime import no_growth_and_limit_reached as _no_growth_and_limit_reached
from .cycle_runtime import (
    prune_saved_generation_artifacts as _prune_saved_generation_artifacts,
)
from .cycle_runtime import resolve_runtime_restore_path as _resolve_runtime_restore_path
from .cycle_runtime import resolve_tree_status as _resolve_tree_status
from .cycle_timing import save_trigger_reason as _save_trigger_reason
from .cycle_timing import should_save_progress
from .cycle_timing import timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s
from .cycle_training import (
    restrict_evaluators_config as _restrict_evaluators_config,
)
from .cycle_training import train_and_select_evaluators as _train_and_select_evaluators
from .cycle_training import (
    train_and_select_evaluators_streaming as _train_and_select_evaluators_streaming,
)
from .cycle_validation import (
    previous_effective_runtime_config as _previous_effective_runtime_config,
)
from .cycle_validation import reevaluate_tree_for_policy as _reevaluate_tree_for_policy
from .cycle_validation import (
    validate_dataset_family_target_args as _validate_dataset_family_target_args,
)
from .cycle_validation import validate_forced_evaluator as _validate_forced_evaluator
from .cycle_validation import validate_pipeline_mode as _validate_pipeline_mode
from .cycle_validation import (
    validate_runtime_reconfiguration as _validate_runtime_reconfiguration,
)
from .history import MorpionBootstrapHistoryRecorder
from .memory_diagnostics import MemoryDiagnostics
from .pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionPipelineDatasetStatus,
    MorpionPipelineDatasetStatusArtifact,
    MorpionPipelineGenerationManifest,
    MorpionPipelineTrainingCursor,
    MorpionPipelineTrainingStatus,
    load_pipeline_active_model,
    load_pipeline_dataset_status_file,
    load_pipeline_manifest,
    load_pipeline_training_cursor,
    save_pipeline_active_model,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_cursor,
    save_pipeline_training_status_file,
)
from .pipeline_claims import (
    claim_pipeline_stage,
    release_pipeline_stage_claim,
)
from .pipeline_memory import log_available_ram_guard, log_pipeline_memory
from .record_status import (
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
    persist_certified_leaderboard_candidates,
    resolve_frontier_status_for_cycle,
    resolve_frontier_status_for_cycle_with_metadata,
    resolve_record_status_for_cycle,
)
from .reevaluation_patch_consumer import apply_pending_reevaluation_patch_to_runner
from .run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs
    from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)


def _artifact_pipeline_mode_required_error() -> ValueError:
    """Build the canonical artifact-pipeline mode requirement error."""
    return ValueError("artifact_pipeline mode required")


def _require_artifact_pipeline_mode(args: MorpionBootstrapArgs) -> None:
    """Require explicit artifact-pipeline mode for stage entrypoints."""
    if args.pipeline_mode != "artifact_pipeline":
        raise _artifact_pipeline_mode_required_error()


def require_artifact_pipeline_mode(args: MorpionBootstrapArgs) -> None:
    """Public wrapper around the artifact-pipeline mode requirement."""
    _require_artifact_pipeline_mode(args)


class MissingPipelineTreeSnapshotFileError(FileNotFoundError):
    """Raised when a pipeline dataset stage cannot find its tree snapshot."""

    @classmethod
    def from_path(
        cls, tree_snapshot_path: Path | None
    ) -> MissingPipelineTreeSnapshotFileError:
        """Build one missing-tree-snapshot error with the resolved path."""
        return cls(f"Pipeline tree snapshot does not exist: {tree_snapshot_path}")


class MissingPipelineRowsFileError(FileNotFoundError):
    """Raised when a pipeline training stage cannot find its rows file."""

    @classmethod
    def from_path(cls, rows_path: Path | None) -> MissingPipelineRowsFileError:
        """Build one missing-rows error with the resolved path."""
        return cls(f"Pipeline rows file does not exist: {rows_path}")


def _manifest_tree_snapshot_required_error() -> ValueError:
    """Build the canonical missing tree snapshot path error."""
    return ValueError("manifest.tree_snapshot_path is required")


def _manifest_rows_path_required_error() -> ValueError:
    """Build the canonical missing rows path error."""
    return ValueError("manifest.rows_path is required")


def _dataset_stage_requires_done_status_error() -> ValueError:
    """Build the canonical dataset-ready error for training stage entry."""
    return ValueError("manifest.dataset_status == 'done' is required")


def _dataset_rows_count_mismatch_error(
    *,
    generation: int,
    expected_rows: int,
    actual_rows: int,
) -> RuntimeError:
    """Build the canonical dataset-stage rows count mismatch error."""
    return RuntimeError(
        "Dataset rows metadata count mismatch "
        f"generation={generation} expected_rows={expected_rows} "
        f"actual_rows={actual_rows}."
    )


def _raise_missing_tree_snapshot_file_error(
    tree_snapshot_path: Path | None,
) -> NoReturn:
    """Raise the canonical dataset-stage missing snapshot file error."""
    raise MissingPipelineTreeSnapshotFileError.from_path(tree_snapshot_path)


def _raise_missing_rows_file_error(rows_path: Path | None) -> NoReturn:
    """Raise the canonical training-stage missing rows file error."""
    raise MissingPipelineRowsFileError.from_path(rows_path)


def _raise_dataset_rows_count_mismatch_error(
    *,
    generation: int,
    expected_rows: int,
    actual_rows: int,
) -> NoReturn:
    """Raise the canonical dataset-stage rows count mismatch error."""
    raise _dataset_rows_count_mismatch_error(
        generation=generation,
        expected_rows=expected_rows,
        actual_rows=actual_rows,
    )


def _pipeline_manifest_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the canonical manifest path for one pipeline generation."""
    return paths.pipeline_manifest_path_for_generation(generation)


def _now_timestamp_utc() -> str:
    """Return the current UTC timestamp formatted like the bootstrap loop."""
    return _timestamp_utc_from_unix_s(time.time())


def _optional_generation_max(left: int | None, right: int) -> int:
    """Return max for an optional generation value and a concrete generation."""
    return max(left if left is not None else -1, right)


def _active_model_generation_for_training_guard(
    paths: MorpionBootstrapPaths,
) -> int | None:
    """Return current active-model generation when the singleton artifact exists."""
    if not paths.pipeline_active_model_path.is_file():
        return None
    return load_pipeline_active_model(paths.pipeline_active_model_path).generation


def _training_lower_bound_generation(paths: MorpionBootstrapPaths) -> int:
    """Return the monotonic lower bound for an explicit training stage."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    active_generation = _active_model_generation_for_training_guard(paths)
    return max(
        active_generation if active_generation is not None else -1,
        (
            cursor.latest_started_generation
            if cursor.latest_started_generation is not None
            else -1
        ),
        (
            cursor.latest_completed_generation
            if cursor.latest_completed_generation is not None
            else -1
        ),
    )


def _save_training_cursor_started(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has started training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_started_generation=_optional_generation_max(
            cursor.latest_started_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor


def _training_rows_subset_path(
    paths: MorpionBootstrapPaths,
    generation: int,
) -> Path:
    """Return the debug training-row subset path for one generation."""
    return paths.rows_dir / f"generation_{generation:06d}.training_subset.json"


def _save_training_cursor_completed(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineTrainingCursor:
    """Persist that one generation has completed training."""
    cursor = load_pipeline_training_cursor(paths.pipeline_training_cursor_path)
    next_cursor = replace(
        cursor,
        latest_completed_generation=_optional_generation_max(
            cursor.latest_completed_generation,
            generation,
        ),
    )
    save_pipeline_training_cursor(next_cursor, paths.pipeline_training_cursor_path)
    return next_cursor


def _configure_linoo_selection_artifact_for_growth(
    *,
    runner: object,
    paths: MorpionBootstrapPaths,
    cycle_index: int,
    generation: int,
) -> None:
    """Configure latest Linoo table persistence when the runner supports it."""
    configure = getattr(runner, "configure_linoo_selection_table_artifact", None)
    if not callable(configure):
        return
    configure(
        path=paths.latest_linoo_selection_table_path,
        cycle_index=cycle_index,
        generation=generation,
    )


def _load_generation_manifest(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineGenerationManifest:
    """Load the persisted pipeline manifest for one generation."""
    return load_pipeline_manifest(_pipeline_manifest_path(paths, generation))


def _latest_prior_dataset_status_artifact(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionPipelineDatasetStatusArtifact | None:
    """Return the latest readable dataset-status artifact before one generation."""
    for previous_generation in range(generation - 1, -1, -1):
        status_path = paths.pipeline_dataset_status_path_for_generation(
            previous_generation
        )
        if not status_path.is_file():
            continue
        try:
            return load_pipeline_dataset_status_file(status_path)
        except Exception:
            LOGGER.warning(
                "Skipping unreadable dataset status artifact: %s",
                status_path,
                exc_info=True,
            )
    return None


def _resolve_previous_pipeline_record_status(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionBootstrapRecordStatus | None:
    """Return the previous record status for one pipeline dataset generation."""
    latest_dataset_status = _latest_prior_dataset_status_artifact(
        paths=paths,
        generation=generation,
    )
    if latest_dataset_status is not None:
        return latest_dataset_status.record_status
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path).latest_record_status
    return None


def _resolve_previous_pipeline_frontier_status(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
) -> MorpionBootstrapFrontierStatus | None:
    """Return the previous frontier status for one pipeline dataset generation."""
    latest_dataset_status = _latest_prior_dataset_status_artifact(
        paths=paths,
        generation=generation,
    )
    if latest_dataset_status is not None:
        return latest_dataset_status.frontier_status
    if paths.run_state_path.is_file():
        return load_bootstrap_run_state(paths.run_state_path).latest_frontier_status
    return None


def _save_dataset_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    dataset_status: MorpionPipelineDatasetStatus,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated dataset-stage manifest and matching status file."""
    next_manifest = replace(manifest, dataset_status=dataset_status)
    save_pipeline_manifest(
        next_manifest, _pipeline_manifest_path(paths, manifest.generation)
    )
    save_pipeline_dataset_status_file(
        generation=manifest.generation,
        dataset_status=next_manifest.dataset_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_dataset_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def _save_training_manifest_status(
    *,
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    training_status: MorpionPipelineTrainingStatus,
    timestamp_utc: str,
) -> MorpionPipelineGenerationManifest:
    """Persist one updated training-stage manifest and matching status file."""
    next_manifest = replace(manifest, training_status=training_status)
    save_pipeline_manifest(
        next_manifest, _pipeline_manifest_path(paths, manifest.generation)
    )
    save_pipeline_training_status_file(
        generation=manifest.generation,
        training_status=next_manifest.training_status,
        updated_at_utc=timestamp_utc,
        metadata=next_manifest.metadata,
        path=paths.pipeline_training_status_path_for_generation(manifest.generation),
    )
    return next_manifest


def _require_manifest_tree_snapshot_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest tree snapshot path for dataset extraction."""
    if manifest.tree_snapshot_path is None:
        raise _manifest_tree_snapshot_required_error()
    return manifest.tree_snapshot_path


def _require_manifest_rows_path(
    manifest: MorpionPipelineGenerationManifest,
) -> str:
    """Require one manifest rows path for pipeline training."""
    if manifest.rows_path is None:
        raise _manifest_rows_path_required_error()
    return manifest.rows_path


def _resolve_pipeline_active_model_for_growth(
    *,
    paths: MorpionBootstrapPaths,
    force_evaluator: str | None,
) -> ResolvedActiveMorpionModelBundle:
    """Resolve the active model for artifact-pipeline growth from the pipeline artifact."""
    if not paths.pipeline_active_model_path.is_file():
        LOGGER.info(
            "[growth] active_model_status source=none evaluator=none model_bundle=none"
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )

    active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    if force_evaluator is not None and active_model.evaluator_name != force_evaluator:
        LOGGER.warning(
            "[growth] active_model_force_evaluator_mismatch requested=%s active=%s artifact=%s",
            force_evaluator,
            active_model.evaluator_name,
            paths.pipeline_active_model_path,
        )
    model_bundle_path = paths.resolve_work_dir_path(active_model.model_bundle_path)
    if model_bundle_path is None or not model_bundle_path.exists():
        LOGGER.warning(
            "[growth] active_model_missing_bundle source=pipeline_active_model generation=%s evaluator=%s model_bundle=%s artifact=%s",
            active_model.generation,
            active_model.evaluator_name,
            model_bundle_path,
            paths.pipeline_active_model_path,
        )
        LOGGER.info(
            "[growth] active_model_status source=none evaluator=none model_bundle=none"
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )
    LOGGER.info(
        "[growth] active_model_status source=pipeline_active_model generation=%s evaluator=%s model_bundle=%s",
        active_model.generation,
        active_model.evaluator_name,
        model_bundle_path,
    )
    return ResolvedActiveMorpionModelBundle(
        active_evaluator_name=active_model.evaluator_name,
        model_bundle_path=model_bundle_path,
    )


def run_pipeline_growth_stage(
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    *,
    max_cycles: int = 1,
) -> MorpionBootstrapRunState:
    """Run the Phase 4 growth-only stage for artifact-pipeline mode."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    current_config = bootstrap_config_from_args(args)
    if paths.bootstrap_config_path.is_file():
        persisted_config = load_bootstrap_config(paths.bootstrap_config_path)
        validate_bootstrap_config_change(persisted_config, current_config)
    save_bootstrap_config(current_config, paths.bootstrap_config_path)
    config_hash = bootstrap_config_sha256(current_config)

    run_state = (
        load_bootstrap_run_state(paths.run_state_path)
        if paths.run_state_path.is_file()
        else initialize_bootstrap_run_state()
    )
    run_state = _with_config_hash_metadata(run_state, config_hash=config_hash)

    LOGGER.info("[pipeline] growth_start max_cycles=%s", max_cycles)
    cycles_run = 0
    while cycles_run < max_cycles:
        previous_generation = run_state.generation
        control = load_bootstrap_control(paths.control_path)
        effective_args = apply_control_to_args(args, control)
        run_state = _run_one_pipeline_growth_cycle(
            args=effective_args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            config_hash=config_hash,
            bootstrap_config=current_config,
        )
        save_bootstrap_run_state(run_state, paths.run_state_path)
        if run_state.generation > previous_generation:
            _prune_saved_generation_artifacts(paths)
        cycles_run += 1
        if (
            run_state.metadata.get(GROWTH_STATUS_METADATA_KEY)
            == GROWTH_BUDGET_ALREADY_EXHAUSTED_STATUS
        ):
            LOGGER.info(
                "[pipeline] growth_stop reason=growth_budget_already_exhausted cycle=%s generation=%s",
                run_state.cycle_index,
                run_state.generation,
            )
            break
    LOGGER.info(
        "[pipeline] growth_done generation=%s cycle=%s",
        run_state.generation,
        run_state.cycle_index,
    )
    return run_state


def _run_one_pipeline_growth_cycle(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None,
    config_hash: str,
    bootstrap_config: MorpionBootstrapConfig,
    now_unix_s: float | None = None,
) -> MorpionBootstrapRunState:
    """Run one growth-only artifact-pipeline cycle with memory hooks."""
    memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
    memory.log("cycle_start")
    try:
        return _run_one_pipeline_growth_cycle_impl(
            args=args,
            paths=paths,
            runner=runner,
            run_state=run_state,
            control=control,
            config_hash=config_hash,
            bootstrap_config=bootstrap_config,
            now_unix_s=now_unix_s,
            memory=memory,
        )
    finally:
        log_after_cycle_gc(memory)
        memory.close()


def _run_one_pipeline_growth_cycle_impl(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    control: MorpionBootstrapControl | None,
    config_hash: str,
    bootstrap_config: MorpionBootstrapConfig,
    now_unix_s: float | None,
    memory: MemoryDiagnostics,
) -> MorpionBootstrapRunState:
    """Run one artifact-pipeline cycle that only grows and exports artifacts."""
    cycle_started_at = time.perf_counter()
    _validate_pipeline_mode(args)
    _require_artifact_pipeline_mode(args)
    _validate_dataset_family_target_args(args)
    reevaluate_tree = _reevaluate_tree_for_policy(args.evaluator_update_policy)
    resolved_control = MorpionBootstrapControl() if control is None else control
    effective_runtime_config = effective_runtime_config_from_config_and_control(
        bootstrap_config,
        resolved_control,
    )
    previous_effective_runtime_config = _previous_effective_runtime_config(
        run_state.metadata,
        resolved_bootstrap_config=bootstrap_config,
    )
    _validate_runtime_reconfiguration(
        previous_effective_runtime_config=previous_effective_runtime_config,
        effective_runtime_config=effective_runtime_config,
    )
    resolved_evaluators_config = args.resolved_evaluators_config()
    _validate_forced_evaluator(
        force_evaluator=resolved_control.force_evaluator,
        evaluator_names=resolved_evaluators_config.evaluators,
    )

    cycle_index = run_state.cycle_index + 1
    LOGGER.info(
        "[pipeline] growth_cycle_start cycle=%s generation=%s",
        cycle_index,
        run_state.generation,
    )
    log_pipeline_memory(
        stage="growth",
        generation=run_state.generation,
        event="start",
    )
    history_recorder = MorpionBootstrapHistoryRecorder(paths.history_paths())
    resolved_active_model = _resolve_pipeline_active_model_for_growth(
        paths=paths,
        force_evaluator=resolved_control.force_evaluator,
    )
    restore_tree_path = _resolve_runtime_restore_path(paths=paths, run_state=run_state)
    if not log_available_ram_guard(
        stage="growth",
        generation=run_state.generation,
        action="checkpoint_load",
        required_mb=args.min_available_ram_mb,
    ):
        LOGGER.info(
            "[pipeline] growth_skip generation=%s reason=low_available_ram action=checkpoint_load",
            run_state.generation,
        )
        log_pipeline_memory(
            stage="growth",
            generation=run_state.generation,
            event="done",
            reason="low_available_ram",
        )
        return run_state
    runner.load_or_create(
        restore_tree_path,
        resolved_active_model.model_bundle_path,
        effective_runtime_config,
        reevaluate_tree=reevaluate_tree,
    )
    restored_tree_size = runner.current_tree_size()
    restored_branch_count = _current_tree_branch_count(runner)
    log_pipeline_memory(
        stage="growth",
        generation=run_state.generation,
        event="after_checkpoint_load",
        node_count=restored_tree_size,
        branch_count=restored_branch_count,
    )
    if not log_available_ram_guard(
        stage="growth",
        generation=run_state.generation,
        action="tree_growth",
        required_mb=args.min_available_ram_mb,
    ):
        LOGGER.info(
            "[pipeline] growth_skip generation=%s reason=low_available_ram action=tree_growth",
            run_state.generation,
        )
        log_pipeline_memory(
            stage="growth",
            generation=run_state.generation,
            event="done",
            node_count=restored_tree_size,
            branch_count=restored_branch_count,
            reason="low_available_ram",
        )
        return run_state
    reevaluation_patch_result = apply_pending_reevaluation_patch_to_runner(
        paths=paths,
        runner=runner,
    )
    memory.log("after_runtime_restore")
    memory.log("before_tree_growth")
    growth_started_at = time.perf_counter()
    tree_size_before_growth = runner.current_tree_size()
    branch_count_before_growth = _current_tree_branch_count(runner)
    log_pipeline_memory(
        stage="growth",
        generation=run_state.generation,
        event="before_growth",
        node_count=tree_size_before_growth,
        branch_count=branch_count_before_growth,
    )
    _configure_linoo_selection_artifact_for_growth(
        runner=runner,
        paths=paths,
        cycle_index=cycle_index,
        generation=run_state.generation,
    )
    runner.grow(args.max_growth_steps_per_cycle)
    growth_duration_s = time.perf_counter() - growth_started_at
    current_tree_size = runner.current_tree_size()
    memory.log("after_tree_growth")
    LOGGER.info(
        "[growth] cycle_done elapsed=%.3fs nodes_before=%s nodes_after=%s delta=%s",
        growth_duration_s,
        tree_size_before_growth,
        current_tree_size,
        current_tree_size - tree_size_before_growth,
    )
    nodes_added = current_tree_size - tree_size_before_growth
    branch_count = _current_tree_branch_count(runner)
    log_pipeline_memory(
        stage="growth",
        generation=run_state.generation,
        event="after_growth",
        node_count=current_tree_size,
        branch_count=branch_count,
        nodes_added=nodes_added,
    )
    tree_status = _resolve_tree_status(runner, current_tree_size=current_tree_size)
    frontier_status = resolve_frontier_status_for_cycle(
        snapshot=None,
        previous_frontier_status=run_state.latest_frontier_status,
    )
    current_time = time.time() if now_unix_s is None else now_unix_s
    timestamp_utc = _timestamp_utc_from_unix_s(current_time)
    save_triggered = should_save_progress(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    )
    save_reason = _save_trigger_reason(
        current_tree_size=current_tree_size,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        now_unix_s=current_time,
        last_save_unix_s=run_state.last_save_unix_s,
        save_after_tree_growth_factor=args.save_after_tree_growth_factor,
        save_after_seconds=args.save_after_seconds,
    )

    if _no_growth_and_limit_reached(
        nodes_added=nodes_added,
        branch_count=branch_count,
        tree_branch_limit=effective_runtime_config.tree_branch_limit,
    ) and not reevaluation_patch_result.patch_applied:
        assert branch_count is not None
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info(
            "[growth] no_op_limit_reached branch_count=%s limit=%s checkpoint_skipped=true",
            branch_count,
            effective_runtime_config.tree_branch_limit,
        )
        LOGGER.info(
            "[save] skipped reason=no_growth_changes nodes_added=%s",
            nodes_added,
        )
        LOGGER.info("[save] skipped reason=no_growth_and_limit_reached")
        LOGGER.info(
            "[timing] cycle_done growth=%.3fs training=%.3fs total_cycle=%.3fs",
            growth_duration_s,
            0.0,
            cycle_duration_s,
        )
        next_run_state = _build_growth_budget_exhausted_run_state(
            run_state=run_state,
            resolved_active_model=resolved_active_model,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            cycle_index=cycle_index,
            branch_count=branch_count,
            tree_branch_limit=effective_runtime_config.tree_branch_limit,
        )
        _record_no_save_cycle_event(
            history_recorder=history_recorder,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            frontier_status=frontier_status,
            run_state=run_state,
            next_run_state=next_run_state,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        )
        LOGGER.info(
            "[pipeline] growth_cycle_done cycle=%s generation=%s saved=false elapsed=%.3fs status=growth_budget_already_exhausted",
            cycle_index,
            next_run_state.generation,
            cycle_duration_s,
        )
        log_pipeline_memory(
            stage="growth",
            generation=next_run_state.generation,
            event="done",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        return next_run_state

    if (
        run_state.generation > 0
        and nodes_added <= 0
        and current_tree_size <= run_state.tree_size_at_last_save
        and not reevaluation_patch_result.patch_applied
    ):
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info(
            "[save] skipped reason=no_growth_changes nodes_added=%s",
            nodes_added,
        )
        LOGGER.info(
            "[timing] cycle_done growth=%.3fs training=%.3fs total_cycle=%.3fs",
            growth_duration_s,
            0.0,
            cycle_duration_s,
        )
        next_run_state = _build_no_save_run_state(
            run_state=run_state,
            resolved_active_model=resolved_active_model,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            cycle_index=cycle_index,
        )
        _record_no_save_cycle_event(
            history_recorder=history_recorder,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            frontier_status=frontier_status,
            run_state=run_state,
            next_run_state=next_run_state,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        )
        LOGGER.info(
            "[pipeline] growth_cycle_done cycle=%s generation=%s saved=false elapsed=%.3fs status=no_growth_changes",
            cycle_index,
            next_run_state.generation,
            cycle_duration_s,
        )
        log_pipeline_memory(
            stage="growth",
            generation=next_run_state.generation,
            event="done",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        return next_run_state

    if not save_triggered:
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info("[save] decision_done triggered=false reason=threshold_not_reached")
        LOGGER.info(
            "[timing] cycle_done growth=%.3fs training=%.3fs total_cycle=%.3fs",
            growth_duration_s,
            0.0,
            cycle_duration_s,
        )
        next_run_state = _build_no_save_run_state(
            run_state=run_state,
            resolved_active_model=resolved_active_model,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
            cycle_index=cycle_index,
        )
        _record_no_save_cycle_event(
            history_recorder=history_recorder,
            cycle_index=cycle_index,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            frontier_status=frontier_status,
            run_state=run_state,
            next_run_state=next_run_state,
            resolved_control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        )
        log_pipeline_memory(
            stage="growth",
            generation=next_run_state.generation,
            event="done",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        return next_run_state

    generation = run_state.generation + 1
    LOGGER.info(
        "[save] decision_done triggered=true reason=%s", save_reason or "unknown"
    )
    runtime_checkpoint_path = paths.runtime_checkpoint_path_for_generation(generation)
    relative_runtime_checkpoint_path: str | None = None
    save_checkpoint = getattr(runner, "save_checkpoint", None)
    if callable(save_checkpoint):
        if not log_available_ram_guard(
            stage="growth",
            generation=generation,
            action="checkpoint_save",
            required_mb=args.min_available_ram_mb,
        ):
            cycle_duration_s = time.perf_counter() - cycle_started_at
            LOGGER.info(
                "[pipeline] growth_skip generation=%s reason=low_available_ram action=checkpoint_save",
                generation,
            )
            LOGGER.info(
                "[save] skipped reason=low_available_ram unsaved_nodes_added=%s",
                nodes_added,
            )
            LOGGER.info(
                "[timing] cycle_done growth=%.3fs training=%.3fs total_cycle=%.3fs",
                growth_duration_s,
                0.0,
                cycle_duration_s,
            )
            next_run_state = _build_no_save_run_state(
                run_state=run_state,
                resolved_active_model=resolved_active_model,
                resolved_control=resolved_control,
                effective_runtime_config=effective_runtime_config,
                cycle_index=cycle_index,
            )
            _record_no_save_cycle_event(
                history_recorder=history_recorder,
                cycle_index=cycle_index,
                timestamp_utc=timestamp_utc,
                tree_status=tree_status,
                frontier_status=frontier_status,
                run_state=run_state,
                next_run_state=next_run_state,
                resolved_control=resolved_control,
                effective_runtime_config=effective_runtime_config,
            )
            log_pipeline_memory(
                stage="growth",
                generation=next_run_state.generation,
                event="done",
                node_count=current_tree_size,
                branch_count=branch_count,
                reason="low_available_ram",
            )
            return next_run_state
        log_pipeline_memory(
            stage="growth",
            generation=generation,
            event="before_checkpoint_save",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        save_checkpoint(runtime_checkpoint_path)
        if not runtime_checkpoint_path.is_file():
            raise MissingSavedBootstrapArtifactError(
                action="runner.save_checkpoint()",
                artifact_path=runtime_checkpoint_path,
            )
        relative_runtime_checkpoint_path = paths.relative_to_work_dir(
            runtime_checkpoint_path
        )
    else:
        LOGGER.info("[checkpoint] skipped reason=runner_has_no_save_checkpoint")

    tree_snapshot_path = _export_training_snapshot_for_generation(
        args=args,
        paths=paths,
        runner=runner,
        generation=generation,
    )
    if not tree_snapshot_path.is_file():
        raise MissingSavedBootstrapArtifactError(
            action="export_training_snapshot_for_generation()",
            artifact_path=tree_snapshot_path,
        )
    relative_tree_snapshot_path = paths.relative_to_work_dir(tree_snapshot_path)
    cycle_duration_s = time.perf_counter() - cycle_started_at
    next_run_state = MorpionBootstrapRunState(
        generation=generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=relative_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=resolved_active_model.active_evaluator_name,
        tree_size_at_last_save=current_tree_size,
        last_save_unix_s=current_time,
        latest_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        latest_record_status=run_state.latest_record_status,
        latest_frontier_status=frontier_status,
        metadata=_next_metadata(
            run_state.metadata,
            relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
            control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        ),
    )
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            runtime_checkpoint_path=relative_runtime_checkpoint_path,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=None,
            dataset_num_rows=None,
            dataset_num_samples=None,
            training_triggered=False,
            frontier_status=frontier_status,
            record_status=resolve_record_status_for_cycle(
                snapshot=None,
                previous_record_status=run_state.latest_record_status,
            ),
            metadata={
                **_build_event_metadata(
                    active_evaluator_name=next_run_state.active_evaluator_name,
                    config_hash=config_hash,
                    forced_evaluator=resolved_control.force_evaluator,
                    runtime_control=resolved_control.runtime,
                    effective_runtime_config=effective_runtime_config,
                ),
                **_pipeline_metadata(args=args),
            },
        )
    )
    _write_pipeline_manifest_for_generation(
        paths=paths,
        generation=generation,
        timestamp_utc=timestamp_utc,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        relative_tree_snapshot_path=relative_tree_snapshot_path,
        relative_rows_path=None,
        model_bundle_paths={},
        selected_evaluator_name=None,
        dataset_status="not_started",
        training_status="not_started",
        metadata=_pipeline_metadata(args=args),
    )
    LOGGER.info(
        "[pipeline] growth_cycle_done cycle=%s generation=%s saved=true elapsed=%.3fs",
        cycle_index,
        generation,
        cycle_duration_s,
    )
    log_pipeline_memory(
        stage="growth",
        generation=generation,
        event="done",
        node_count=current_tree_size,
        branch_count=branch_count,
    )
    return next_run_state


def run_pipeline_dataset_stage(
    args: MorpionBootstrapArgs,
    *,
    generation: int,
    claim_ttl_seconds: float = 3600.0,
    claim_owner: str | None = None,
) -> MorpionPipelineGenerationManifest:
    """Extract supervised rows for one persisted pipeline generation."""
    _require_artifact_pipeline_mode(args)
    stage_started_at = time.perf_counter()
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    manifest = _load_generation_manifest(paths=paths, generation=generation)
    guard_tree_snapshot_path = (
        paths.resolve_work_dir_path(manifest.tree_snapshot_path)
        if manifest.tree_snapshot_path is not None
        else None
    )
    if (
        guard_tree_snapshot_path is not None
        and guard_tree_snapshot_path.is_file()
        and not log_available_ram_guard(
            stage="dataset",
            generation=generation,
            action="snapshot_load",
            required_mb=args.min_available_ram_mb,
        )
    ):
        LOGGER.info(
            "[pipeline] dataset_skip generation=%s reason=low_available_ram action=snapshot_load",
            generation,
        )
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="done",
            reason="low_available_ram",
        )
        return manifest
    claim = claim_pipeline_stage(
        generation=generation,
        stage="dataset",
        claim_path=paths.pipeline_dataset_claim_path_for_generation(generation),
        ttl_seconds=claim_ttl_seconds,
        owner=claim_owner,
        metadata={"entrypoint": "run_pipeline_dataset_stage"},
    )
    LOGGER.info(
        "[pipeline] dataset_claim_created generation=%s claim_path=%s owner=%s expires_at=%s",
        generation,
        paths.pipeline_dataset_claim_path_for_generation(generation),
        "none" if claim.owner is None else claim.owner,
        claim.expires_at_utc,
    )
    timestamp_utc = _now_timestamp_utc()
    LOGGER.info("[pipeline] dataset_start generation=%s", generation)
    log_pipeline_memory(
        stage="dataset",
        generation=generation,
        event="start",
    )
    manifest = _save_dataset_manifest_status(
        paths=paths,
        manifest=manifest,
        dataset_status="extracting_rows",
        timestamp_utc=timestamp_utc,
    )
    try:
        tree_snapshot_path = paths.resolve_work_dir_path(
            _require_manifest_tree_snapshot_path(manifest)
        )
        rows_path = (
            paths.resolve_work_dir_path(manifest.rows_path)
            if manifest.rows_path is not None
            else paths.rows_jsonl_path_for_generation(generation)
        )
        if rows_path is None:
            rows_path = paths.rows_jsonl_path_for_generation(generation)
        LOGGER.info(
            "[pipeline] dataset_export_start generation=%s tree_export=%s rows_output=%s config={min_depth=%s, min_visit_count=%s, max_rows=%s, target_policy=%s, use_backed_up_value=%s, require_exact_or_terminal=%s}",
            generation,
            tree_snapshot_path,
            rows_path,
            args.min_depth,
            args.min_visit_count,
            args.max_rows,
            args.dataset_family_target_policy,
            args.use_backed_up_value,
            args.require_exact_or_terminal,
        )
        if tree_snapshot_path is None or not tree_snapshot_path.is_file():
            LOGGER.info(
                "[pipeline] dataset_skip generation=%s reason=missing_tree_export tree_export=%s manifest=%s",
                generation,
                tree_snapshot_path,
                _pipeline_manifest_path(paths, generation),
            )
            _raise_missing_tree_snapshot_file_error(tree_snapshot_path)
        export_started_at = time.perf_counter()
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="before_snapshot_load",
            tree_snapshot_path=tree_snapshot_path,
        )
        snapshot = _load_training_snapshot_for_generation(
            args=args,
            artifact_path=tree_snapshot_path,
        )
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="after_snapshot_load",
            node_count=len(snapshot.nodes),
        )
        previous_record_status = _resolve_previous_pipeline_record_status(
            paths=paths,
            generation=generation,
        )
        previous_frontier_status = _resolve_previous_pipeline_frontier_status(
            paths=paths,
            generation=generation,
        )
        LOGGER.info("[record] resolve_start nodes=%s", len(snapshot.nodes))
        record_started_at = time.perf_counter()
        record_status = resolve_record_status_for_cycle(
            snapshot=snapshot,
            previous_record_status=previous_record_status,
            generation=generation,
        )
        LOGGER.info(
            "[record] resolve_done generation=%s elapsed=%.3fs best_total_points=%s",
            generation,
            time.perf_counter() - record_started_at,
            record_status.current_best_total_points,
        )
        LOGGER.info("[frontier] resolve_start nodes=%s", len(snapshot.nodes))
        log_pipeline_memory(
            stage="frontier",
            generation=generation,
            event="resolve_start",
            nodes=len(snapshot.nodes),
        )
        frontier_started_at = time.perf_counter()
        frontier_resolution = resolve_frontier_status_for_cycle_with_metadata(
            snapshot=snapshot,
            previous_frontier_status=previous_frontier_status,
        )
        frontier_status = frontier_resolution.status
        LOGGER.info(
            "[frontier] resolve_done generation=%s elapsed=%.3fs candidates=%s best_total_points=%s method=depth_metadata",
            generation,
            time.perf_counter() - frontier_started_at,
            frontier_resolution.candidate_count,
            frontier_status.current_best_total_points,
        )
        log_pipeline_memory(
            stage="frontier",
            generation=generation,
            event="resolve_done",
            candidates=frontier_resolution.candidate_count,
        )
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="before_rows_write_stream",
            node_count=len(snapshot.nodes),
        )
        streaming_rows = _streaming_rows_from_training_snapshot(
            args=args,
            snapshot=snapshot,
            generation=generation,
        )
        LOGGER.info(
            "[pipeline] dataset_rows_stream_start generation=%s path=%s",
            generation,
            rows_path,
        )
        write_stats = save_morpion_supervised_rows_streaming(
            rows=streaming_rows.rows,
            metadata=streaming_rows.metadata,
            path=rows_path,
            progress_callback=lambda row_count: LOGGER.info(
                "[pipeline] dataset_rows_stream_progress generation=%s rows=%s",
                generation,
                row_count,
            ),
        )
        expected_num_rows = streaming_rows.metadata.get("num_rows")
        if (
            isinstance(expected_num_rows, int)
            and not isinstance(expected_num_rows, bool)
            and expected_num_rows != write_stats.row_count
        ):
            _raise_dataset_rows_count_mismatch_error(
                generation=generation,
                expected_rows=expected_num_rows,
                actual_rows=write_stats.row_count,
            )
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="after_rows_write_stream",
            rows=write_stats.row_count,
            rows_path=rows_path,
        )
        rows_bytes = write_stats.bytes_written
        export_elapsed_s = time.perf_counter() - export_started_at
        LOGGER.info(
            "[pipeline] dataset_rows_stream_done generation=%s rows=%s bytes=%s elapsed=%.3fs path=%s",
            generation,
            write_stats.row_count,
            rows_bytes,
            export_elapsed_s,
            rows_path,
        )
        LOGGER.info(
            "[pipeline] dataset_export_done generation=%s rows=%s output=%s elapsed=%.3fs bytes=%s",
            generation,
            write_stats.row_count,
            rows_path,
            export_elapsed_s,
            rows_bytes,
        )
        timestamp_utc = _now_timestamp_utc()
        manifest_metadata = dict(manifest.metadata)
        manifest_metadata["dataset_completed_at_utc"] = timestamp_utc
        manifest_metadata["dataset_rows"] = write_stats.row_count
        manifest_metadata["dataset_rows_bytes"] = rows_bytes
        manifest = replace(
            manifest,
            rows_path=paths.relative_to_work_dir(rows_path),
            dataset_status="done",
            metadata=manifest_metadata,
        )
        save_pipeline_manifest(manifest, _pipeline_manifest_path(paths, generation))
        LOGGER.info(
            "[leaderboard] persist_start generation=%s cycle=%s",
            generation,
            generation,
        )
        leaderboard_started_at = time.perf_counter()
        try:
            persist_certified_leaderboard_candidates(
                snapshot=snapshot,
                run_work_dir=paths.work_dir,
                generation=generation,
                cycle_index=generation,
                timestamp_utc=timestamp_utc,
            )
        finally:
            LOGGER.info(
                "[leaderboard] persist_done elapsed=%.3fs",
                time.perf_counter() - leaderboard_started_at,
            )
        save_pipeline_dataset_status_file(
            generation=generation,
            dataset_status=manifest.dataset_status,
            updated_at_utc=timestamp_utc,
            metadata=manifest.metadata,
            record_status=record_status,
            frontier_status=frontier_status,
            path=paths.pipeline_dataset_status_path_for_generation(generation),
        )
        LOGGER.info(
            "[pipeline] dataset_manifest_written generation=%s manifest=%s created_at=%s rows=%s",
            generation,
            _pipeline_manifest_path(paths, generation),
            timestamp_utc,
            write_stats.row_count,
        )
        LOGGER.info(
            "[pipeline] dataset_done generation=%s rows=%s",
            generation,
            write_stats.row_count,
        )
        log_pipeline_memory(
            stage="dataset",
            generation=generation,
            event="done",
            rows=write_stats.row_count,
        )
    except Exception:
        timestamp_utc = _now_timestamp_utc()
        _save_dataset_manifest_status(
            paths=paths,
            manifest=manifest,
            dataset_status="failed",
            timestamp_utc=timestamp_utc,
        )
        LOGGER.exception("[pipeline] dataset_fail generation=%s", generation)
        raise
    else:
        LOGGER.info(
            "[pipeline] dataset_worker_done action=exported generation=%s elapsed=%.3fs",
            generation,
            time.perf_counter() - stage_started_at,
        )
        return manifest
    finally:
        release_pipeline_stage_claim(
            claim_path=paths.pipeline_dataset_claim_path_for_generation(generation),
            claim_id=claim.claim_id,
        )


def run_pipeline_training_stage(
    args: MorpionBootstrapArgs,
    *,
    generation: int,
    claim_ttl_seconds: float = 3600.0,
    claim_owner: str | None = None,
) -> MorpionPipelineGenerationManifest:
    """Train evaluators and select the active model for one pipeline generation."""
    _require_artifact_pipeline_mode(args)
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    manifest = _load_generation_manifest(paths=paths, generation=generation)
    lower_bound_generation = _training_lower_bound_generation(paths)
    if generation <= lower_bound_generation:
        LOGGER.info(
            "[pipeline] training_skip generation=%s reason=stale_generation lower_bound_generation=%s",
            generation,
            lower_bound_generation,
        )
        return manifest
    if manifest.dataset_status != "done":
        raise _dataset_stage_requires_done_status_error()
    guard_rows_path = (
        paths.resolve_work_dir_path(manifest.rows_path)
        if manifest.rows_path is not None
        else None
    )
    if (
        guard_rows_path is not None
        and guard_rows_path.is_file()
        and not log_available_ram_guard(
            stage="training",
            generation=generation,
            action="rows_load",
            required_mb=args.min_available_ram_mb,
        )
    ):
        LOGGER.info(
            "[pipeline] training_skip generation=%s reason=low_available_ram action=rows_load",
            generation,
        )
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="done",
            reason="low_available_ram",
        )
        return manifest
    claim = claim_pipeline_stage(
        generation=generation,
        stage="training",
        claim_path=paths.pipeline_training_claim_path_for_generation(generation),
        ttl_seconds=claim_ttl_seconds,
        owner=claim_owner,
        metadata={"entrypoint": "run_pipeline_training_stage"},
    )
    timestamp_utc = _now_timestamp_utc()
    LOGGER.info("[pipeline] training_start generation=%s", generation)
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="start",
    )
    manifest = _save_training_manifest_status(
        paths=paths,
        manifest=manifest,
        training_status="training",
        timestamp_utc=timestamp_utc,
    )
    try:
        resolved_evaluators_config = _restrict_evaluators_config(
            args.resolved_evaluators_config(),
            args.training_evaluator_names,
        )
        _save_training_cursor_started(paths=paths, generation=generation)
        rows_path = paths.resolve_work_dir_path(_require_manifest_rows_path(manifest))
        if rows_path is None or not rows_path.is_file():
            _raise_missing_rows_file_error(rows_path)
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="before_dataset_load",
            rows_path=rows_path,
        )
        rows = None
        if rows_path.suffix == ".jsonl":
            rows_source = morpion_supervised_rows_source_from_path(rows_path)
        else:
            rows = load_morpion_supervised_rows(rows_path)
            rows_source = MorpionSupervisedRowsSource(
                path=rows_path,
                metadata=dict(rows.metadata),
                row_count=len(rows.rows),
                format_kind="json",
            )
        original_training_rows = rows_source.row_count
        training_rows_used = (
            None
            if original_training_rows is None
            else (
                min(original_training_rows, args.training_max_rows)
                if args.training_max_rows is not None
                else original_training_rows
            )
        )
        training_rows_path = rows_path
        manifest_metadata = dict(manifest.metadata)
        manifest_metadata["training_row_source_format"] = rows_source.format_kind
        if original_training_rows is not None:
            manifest_metadata["training_rows_original"] = original_training_rows
        if training_rows_used is not None:
            manifest_metadata["training_rows_used"] = training_rows_used
        if rows_source.format_kind == "jsonl":
            manifest_metadata["training_row_chunk_size"] = args.training_row_chunk_size
            manifest_metadata["training_split_policy"] = morpion_streaming_split_policy(
                args.validation_fraction
            )
        if rows_source.format_kind == "json":
            log_pipeline_memory(
                stage="training",
                generation=generation,
                event="after_dataset_load",
                rows=len(rows.rows),
            )
            original_training_rows = len(rows.rows)
            training_rows_used = original_training_rows
            if args.training_max_rows is not None:
                subset_rows = rows.rows[: args.training_max_rows]
                rows = replace(
                    rows,
                    rows=subset_rows,
                    metadata={
                        **rows.metadata,
                        "training_subset_policy": "first_n",
                        "training_rows_original": original_training_rows,
                        "training_rows_used": len(subset_rows),
                        "training_max_rows": args.training_max_rows,
                    },
                )
                training_rows_used = len(rows.rows)
                training_rows_path = _training_rows_subset_path(paths, generation)
                save_morpion_supervised_rows(rows, training_rows_path)
                manifest_metadata["training_rows_used"] = training_rows_used
                manifest_metadata["training_rows_original"] = original_training_rows
            manifest_metadata["training_row_source_format"] = "json"
        else:
            log_pipeline_memory(
                stage="training",
                generation=generation,
                event="after_dataset_source",
                rows=training_rows_used,
                row_count=original_training_rows,
                row_format=rows_source.format_kind,
            )
        if args.training_max_rows is not None:
            LOGGER.info(
                "[train] row_subset original_rows=%s used_rows=%s policy=%s",
                original_training_rows,
                training_rows_used,
                "first_n",
            )
            manifest_metadata["training_max_rows"] = args.training_max_rows
        if args.skip_evaluator_diagnostics:
            manifest_metadata["skip_evaluator_diagnostics"] = True
        manifest = replace(manifest, metadata=manifest_metadata)
        run_state = (
            load_bootstrap_run_state(paths.run_state_path)
            if paths.run_state_path.is_file()
            else initialize_bootstrap_run_state()
        )
        resolved_control = load_bootstrap_control(paths.control_path)
        memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
        try:
            if rows_source.format_kind == "jsonl":
                training_result = _train_and_select_evaluators_streaming(
                    args=args,
                    paths=paths,
                    run_state=run_state,
                    rows_path=training_rows_path,
                    rows_source=rows_source,
                    generation=generation,
                    timestamp_utc=timestamp_utc,
                    resolved_evaluators_config=resolved_evaluators_config,
                    resolved_control=resolved_control,
                    memory=memory,
                    max_rows=args.training_max_rows,
                    chunk_size=args.training_row_chunk_size,
                )
            else:
                if rows is None:
                    raise AssertionError
                training_result = _train_and_select_evaluators(
                    args=args,
                    paths=paths,
                    run_state=run_state,
                    rows=rows,
                    rows_path=training_rows_path,
                    generation=generation,
                    timestamp_utc=timestamp_utc,
                    resolved_evaluators_config=resolved_evaluators_config,
                    resolved_control=resolved_control,
                    memory=memory,
                )
        finally:
            log_after_cycle_gc(memory)
            memory.close()
        timestamp_utc = _now_timestamp_utc()
        manifest_metadata["training_evaluator_names"] = list(
            resolved_evaluators_config.evaluators
        )
        manifest = replace(
            manifest,
            model_bundle_paths=training_result.model_bundle_paths,
            selected_evaluator_name=training_result.selected_evaluator_name,
            training_status="done",
            metadata=manifest_metadata,
        )
        save_pipeline_manifest(manifest, _pipeline_manifest_path(paths, generation))
        save_pipeline_training_status_file(
            generation=generation,
            training_status=manifest.training_status,
            updated_at_utc=timestamp_utc,
            metadata=manifest.metadata,
            selected_evaluator_name=training_result.selected_evaluator_name,
            selection_policy=training_result.selection_policy,
            evaluator_results=training_result.evaluator_results,
            path=paths.pipeline_training_status_path_for_generation(generation),
        )
        current_active_generation = _active_model_generation_for_training_guard(paths)
        if (
            current_active_generation is not None
            and generation <= current_active_generation
        ):
            LOGGER.info(
                "[pipeline] active_model_update_skipped generation=%s reason=stale_generation active_generation=%s selected=%s",
                generation,
                current_active_generation,
                training_result.selected_evaluator_name,
            )
        else:
            save_pipeline_active_model(
                MorpionPipelineActiveModel(
                    generation=generation,
                    evaluator_name=training_result.selected_evaluator_name,
                    model_bundle_path=training_result.model_bundle_paths[
                        training_result.selected_evaluator_name
                    ],
                    updated_at_utc=timestamp_utc,
                    metadata={"selection_policy": training_result.selection_policy},
                ),
                paths.pipeline_active_model_path,
            )
            LOGGER.info(
                "[pipeline] active_model_update generation=%s evaluator=%s model_bundle=%s",
                generation,
                training_result.selected_evaluator_name,
                training_result.model_bundle_paths[
                    training_result.selected_evaluator_name
                ],
            )
        _save_training_cursor_completed(paths=paths, generation=generation)
        LOGGER.info(
            "[pipeline] training_done generation=%s selected=%s",
            generation,
            training_result.selected_evaluator_name,
        )
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="done",
            selected=training_result.selected_evaluator_name,
        )
    except Exception:
        timestamp_utc = _now_timestamp_utc()
        _save_training_manifest_status(
            paths=paths,
            manifest=manifest,
            training_status="failed",
            timestamp_utc=timestamp_utc,
        )
        LOGGER.exception("[pipeline] training_fail generation=%s", generation)
        raise
    else:
        return manifest
    finally:
        release_pipeline_stage_claim(
            claim_path=paths.pipeline_training_claim_path_for_generation(generation),
            claim_id=claim.claim_id,
        )


__all__ = [
    "run_pipeline_dataset_stage",
    "run_pipeline_growth_stage",
    "run_pipeline_training_stage",
]
