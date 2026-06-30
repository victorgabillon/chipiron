"""Phase 3 artifact-pipeline stage entrypoints for Morpion bootstrap."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from chipiron.environments.morpion.bootstrap.bootstrap_errors import (
    MissingSavedBootstrapArtifactError,
)
from chipiron.environments.morpion.bootstrap.bootstrap_memory import (
    log_after_cycle_gc,
    memory_diagnostics_config_from_args,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
    runtime_checkpoint_artifact_exists,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapConfig,
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    load_bootstrap_config,
    save_bootstrap_config,
    validate_bootstrap_config_change,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapControl,
    apply_control_to_args,
    effective_runtime_config_from_config_and_control,
    load_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    export_training_snapshot_for_generation as _export_training_snapshot_for_generation,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    load_training_snapshot_for_generation as _load_training_snapshot_for_generation,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    streaming_rows_from_training_snapshot as _streaming_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_metadata import build_bootstrap_event
from chipiron.environments.morpion.bootstrap.cycle_metadata import (
    build_event_metadata as _build_event_metadata,
)
from chipiron.environments.morpion.bootstrap.cycle_metadata import (
    next_metadata as _next_metadata,
)
from chipiron.environments.morpion.bootstrap.cycle_metadata import (
    pipeline_metadata as _pipeline_metadata,
)
from chipiron.environments.morpion.bootstrap.cycle_metadata import (
    record_no_save_cycle_event as _record_no_save_cycle_event,
)
from chipiron.environments.morpion.bootstrap.cycle_metadata import (
    with_config_hash_metadata as _with_config_hash_metadata,
)
from chipiron.environments.morpion.bootstrap.cycle_pipeline_manifest import (
    write_pipeline_manifest_for_generation as _write_pipeline_manifest_for_generation,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    GROWTH_BUDGET_ALREADY_EXHAUSTED_STATUS,
    GROWTH_STATUS_METADATA_KEY,
    CandidateCheckpointLoadDeferredError,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    build_growth_budget_exhausted_run_state as _build_growth_budget_exhausted_run_state,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    build_no_save_run_state as _build_no_save_run_state,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    current_tree_branch_count as _current_tree_branch_count,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    no_growth_and_limit_reached as _no_growth_and_limit_reached,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    prune_saved_generation_artifacts as _prune_saved_generation_artifacts,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    resolve_runtime_restore_path as _resolve_runtime_restore_path,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    resolve_tree_status as _resolve_tree_status,
)
from chipiron.environments.morpion.bootstrap.cycle_timing import (
    save_trigger_reason as _save_trigger_reason,
)
from chipiron.environments.morpion.bootstrap.cycle_timing import should_save_progress
from chipiron.environments.morpion.bootstrap.cycle_timing import (
    timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    restrict_evaluators_config as _restrict_evaluators_config,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    train_and_select_evaluators as _train_and_select_evaluators,
)
from chipiron.environments.morpion.bootstrap.cycle_training import (
    train_and_select_evaluators_streaming as _train_and_select_evaluators_streaming,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    previous_effective_runtime_config as _previous_effective_runtime_config,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    reevaluate_tree_for_policy as _reevaluate_tree_for_policy,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    validate_dataset_family_target_args as _validate_dataset_family_target_args,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    validate_forced_evaluator as _validate_forced_evaluator,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    validate_pipeline_mode as _validate_pipeline_mode,
)
from chipiron.environments.morpion.bootstrap.cycle_validation import (
    validate_runtime_reconfiguration as _validate_runtime_reconfiguration,
)
from chipiron.environments.morpion.bootstrap.history import (
    MorpionBootstrapHistoryRecorder,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionPipelineGenerationManifest,
    save_pipeline_active_model,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_status_file,
)
from chipiron.environments.morpion.bootstrap.pipeline_claims import (
    claim_pipeline_stage,
    release_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    current_rss_mb,
    format_metric,
    log_available_ram_guard,
    log_pipeline_memory,
)
from chipiron.environments.morpion.bootstrap.profiling.growth_memory import (
    log_growth_runtime_memory_profile as _log_growth_runtime_memory_profile,
)
from chipiron.environments.morpion.bootstrap.profiling.memory_diagnostics import (
    MemoryDiagnostics,
)
from chipiron.environments.morpion.bootstrap.profiling.recursive_memory import (
    log_growth_recursive_memory_profile as _log_growth_recursive_memory_profile,
)
from chipiron.environments.morpion.bootstrap.record_status import (
    persist_certified_leaderboard_candidates,
    resolve_frontier_status_for_cycle,
    resolve_frontier_status_for_cycle_with_metadata,
    resolve_record_status_for_cycle,
)
from chipiron.environments.morpion.bootstrap.reevaluation_patch_consumer import (
    apply_pending_reevaluation_patch_to_runner,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    MorpionBootstrapRunState,
    initialize_bootstrap_run_state,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.training_logging import (
    TrainingActiveModelCursorSummary,
    log_training_cycle_done,
    log_training_cycle_start,
)
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

from .active_model import resolve_pipeline_active_model_for_growth
from .checkpoint_loading import (
    candidate_checkpoint_payload_loader,
    log_candidate_checkpoint_load_profile,
    runtime_checkpoint_artifact_bytes,
    should_load_candidate_checkpoint,
)
from .cursors import (
    MorpionTrainingLowerBound,
    active_model_generation_for_training_guard,
    save_training_cursor_completed,
    save_training_cursor_started,
    training_lower_bound_details,
    training_rows_subset_path,
)
from .growth_budget import growth_budget_runtime_config
from .manifests import (
    MissingPipelineRowsFileError,
    MissingPipelineTreeSnapshotFileError,
    load_generation_manifest,
    pipeline_manifest_path,
    require_manifest_rows_path,
    require_manifest_tree_snapshot_path,
    resolve_previous_pipeline_frontier_status,
    resolve_previous_pipeline_record_status,
    save_dataset_manifest_status,
    save_training_manifest_status,
)
from .observability import (
    build_observability_metadata_for_dashboard as _build_observability_metadata_for_dashboard,
)
from .observability import (
    configure_linoo_selection_artifact_for_growth,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.search_runner_protocol import (
        MorpionSearchRunner,
    )

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


def _observability_metadata_for_dashboard(
    *,
    runner: object,
    generation: int,
    node_count: int,
    branch_count: int | None,
    branch_count_before_growth: int | None = None,
    growth_budget_metadata: Mapping[str, object] | None = None,
    nodes_added: int,
    growth_duration_s: float,
    cycle_duration_s: float,
) -> dict[str, object]:
    """Compatibility wrapper preserving pipeline_stages RSS monkeypatching."""
    return _build_observability_metadata_for_dashboard(
        runner=runner,
        generation=generation,
        node_count=node_count,
        branch_count=branch_count,
        branch_count_before_growth=branch_count_before_growth,
        growth_budget_metadata=growth_budget_metadata,
        nodes_added=nodes_added,
        growth_duration_s=growth_duration_s,
        cycle_duration_s=cycle_duration_s,
        current_rss_provider=current_rss_mb,
    )


def _now_timestamp_utc() -> str:
    """Return the current UTC timestamp formatted like the bootstrap loop."""
    return _timestamp_utc_from_unix_s(time.time())


def _training_active_model_cursor_summary(
    lower_bound: MorpionTrainingLowerBound,
) -> TrainingActiveModelCursorSummary:
    """Adapt lower-bound details into the human training-log DTO."""
    return TrainingActiveModelCursorSummary(
        active_model_generation=lower_bound.active_model_generation,
        active_model_source_generation=lower_bound.active_model_source_generation,
        active_model_source=lower_bound.active_model_source,
        cursor_started_generation=lower_bound.cursor_started_generation,
        cursor_completed_generation=lower_bound.cursor_completed_generation,
        local_lower_bound_generation=lower_bound.generation,
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
        if run_state.metadata.get(GROWTH_STATUS_METADATA_KEY) == (
            "diagnostic_stop_after_growth"
        ):
            LOGGER.info(
                "[pipeline] growth_stop reason=diagnostic_stop_after_growth cycle=%s generation=%s",
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
        previous_runtime_config=previous_effective_runtime_config,
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
    resolved_active_model = resolve_pipeline_active_model_for_growth(
        paths=paths,
        force_evaluator=resolved_control.force_evaluator,
    )
    try:
        restore_tree_path = _resolve_runtime_restore_path(
            paths=paths,
            run_state=run_state,
            before_candidate_checkpoint_load=lambda _source, _path: (
                should_load_candidate_checkpoint(
                    args=args,
                    generation=run_state.generation,
                    source=_source,
                    candidate_path=_path,
                )
            ),
            after_candidate_checkpoint_load=log_candidate_checkpoint_load_profile
            if args.growth_memory_profile
            else None,
            candidate_checkpoint_payload_loader=(
                candidate_checkpoint_payload_loader(args)
                if args.growth_memory_profile
                else None
            ),
        )
    except CandidateCheckpointLoadDeferredError as exc:
        LOGGER.info(
            "[pipeline] growth_skip generation=%s reason=low_available_ram action=%s",
            run_state.generation,
            exc.action,
        )
        log_pipeline_memory(
            stage="growth",
            generation=run_state.generation,
            event="done",
            reason="low_available_ram",
        )
        return run_state
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
    effective_runtime_config, growth_budget_metadata = growth_budget_runtime_config(
        args=args,
        runner=runner,
        current_branch_count=restored_branch_count,
        effective_runtime_config=effective_runtime_config,
    )
    log_pipeline_memory(
        stage="growth",
        generation=run_state.generation,
        event="after_checkpoint_load",
        node_count=restored_tree_size,
        branch_count=restored_branch_count,
    )
    _log_growth_profile_if_enabled(
        args=args,
        runner=runner,
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
    _log_growth_profile_if_enabled(
        args=args,
        runner=runner,
        generation=run_state.generation,
        event="before_growth",
        node_count=tree_size_before_growth,
        branch_count=branch_count_before_growth,
    )
    configure_linoo_selection_artifact_for_growth(
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
    _log_growth_profile_if_enabled(
        args=args,
        runner=runner,
        generation=run_state.generation,
        event="after_growth",
        node_count=current_tree_size,
        branch_count=branch_count,
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
    if args.growth_save_and_exit:
        save_triggered = True
        save_reason = "growth_save_and_exit"

    if args.diagnostic_stop_after_growth:
        cycle_duration_s = time.perf_counter() - cycle_started_at
        LOGGER.info(
            "[save] skipped reason=diagnostic_stop_after_growth unsaved_nodes_added=%s",
            nodes_added,
        )
        LOGGER.info(
            "[pipeline] growth_stop reason=diagnostic_stop_after_growth cycle=%s generation=%s",
            cycle_index,
            run_state.generation,
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
        metadata = dict(next_run_state.metadata)
        metadata["growth_status"] = "diagnostic_stop_after_growth"
        metadata["checkpoint_skipped_reason"] = "diagnostic_stop_after_growth"
        metadata["checkpoint_skipped"] = True
        next_run_state = replace(next_run_state, metadata=metadata)
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
            "[pipeline] growth_cycle_done cycle=%s generation=%s saved=false elapsed=%.3fs status=diagnostic_stop_after_growth",
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
            reason="diagnostic_stop_after_growth",
        )
        return next_run_state

    if (
        _no_growth_and_limit_reached(
            nodes_added=nodes_added,
            branch_count=branch_count,
            tree_branch_limit=effective_runtime_config.tree_branch_limit,
        )
        and not reevaluation_patch_result.patch_applied
    ):
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

    is_after_initial_generation = run_state.generation > 0
    has_no_growth = nodes_added <= 0
    did_not_expand_tree = current_tree_size <= run_state.tree_size_at_last_save
    has_no_reevaluation_patch = not reevaluation_patch_result.patch_applied
    if (
        is_after_initial_generation
        and has_no_growth
        and did_not_expand_tree
        and has_no_reevaluation_patch
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
    runtime_checkpoint_path = paths.runtime_checkpoint_path_for_generation_with_format(
        generation,
        args.runtime_checkpoint_format,
    )
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
        _log_growth_profile_if_enabled(
            args=args,
            runner=runner,
            generation=generation,
            event="before_checkpoint_save",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        checkpoint_save_started_at = time.perf_counter()
        checkpoint_save_rss_before_mb = (
            current_rss_mb() if args.growth_memory_profile else None
        )
        save_checkpoint(runtime_checkpoint_path)
        if not runtime_checkpoint_artifact_exists(runtime_checkpoint_path):
            raise MissingSavedBootstrapArtifactError(
                action="runner.save_checkpoint()",
                artifact_path=runtime_checkpoint_path,
            )
        checkpoint_save_elapsed_s = time.perf_counter() - checkpoint_save_started_at
        if args.growth_memory_profile:
            checkpoint_save_rss_after_mb = current_rss_mb()
            checkpoint_bytes = runtime_checkpoint_artifact_bytes(
                runtime_checkpoint_path
            )
            LOGGER.info(
                "[growth-profile] event=checkpoint_save_done generation=%s "
                "rss_before_mb=%s rss_after_mb=%s rss_delta_mb=%s "
                "checkpoint_bytes=%s checkpoint_bytes_per_node=%s save_elapsed=%.3fs",
                generation,
                format_metric(checkpoint_save_rss_before_mb),
                format_metric(checkpoint_save_rss_after_mb),
                format_metric(
                    None
                    if checkpoint_save_rss_before_mb is None
                    or checkpoint_save_rss_after_mb is None
                    else checkpoint_save_rss_after_mb - checkpoint_save_rss_before_mb
                ),
                checkpoint_bytes,
                format_metric(
                    None
                    if current_tree_size <= 0 or checkpoint_bytes is None
                    else checkpoint_bytes / current_tree_size
                ),
                checkpoint_save_elapsed_s,
            )
        _log_growth_profile_if_enabled(
            args=args,
            runner=runner,
            generation=generation,
            event="after_checkpoint_save",
            node_count=current_tree_size,
            branch_count=branch_count,
        )
        relative_runtime_checkpoint_path = paths.relative_to_work_dir(
            runtime_checkpoint_path
        )
    else:
        LOGGER.info("[checkpoint] skipped reason=runner_has_no_save_checkpoint")

    relative_tree_snapshot_path: str | None
    training_export_override: dict[str, object] | None = None
    if args.growth_skip_training_export:
        LOGGER.info("[save] training_export_skipped reason=config")
        relative_tree_snapshot_path = None
        training_export_override = {"status": "skipped", "reason": "config"}
    else:
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
    observability_metadata = _observability_metadata_for_dashboard(
        runner=runner,
        generation=generation,
        node_count=current_tree_size,
        branch_count=branch_count,
        branch_count_before_growth=branch_count_before_growth,
        growth_budget_metadata=growth_budget_metadata,
        nodes_added=nodes_added,
        growth_duration_s=growth_duration_s,
        cycle_duration_s=cycle_duration_s,
    )
    checkpoint_metadata = observability_metadata["checkpoint"]
    if isinstance(checkpoint_metadata, dict):
        checkpoint_metadata.setdefault(
            "status",
            "saved" if relative_runtime_checkpoint_path is not None else "skipped",
        )
    if training_export_override is not None:
        observability_metadata["training_export"] = training_export_override
    else:
        training_export_metadata = observability_metadata["training_export"]
        if isinstance(training_export_metadata, dict):
            training_export_metadata.setdefault("status", "written")
    run_state_metadata = _next_metadata(
        run_state.metadata,
        relative_runtime_checkpoint_path=relative_runtime_checkpoint_path,
        control=resolved_control,
        effective_runtime_config=effective_runtime_config,
    )
    run_state_metadata.update(observability_metadata)
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
        metadata=run_state_metadata,
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
                **observability_metadata,
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
        metadata={**_pipeline_metadata(args=args), **observability_metadata},
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


def _log_growth_profile_if_enabled(
    *,
    args: MorpionBootstrapArgs,
    runner: MorpionSearchRunner,
    generation: int,
    event: str,
    node_count: int | None,
    branch_count: int | None,
) -> None:
    """Log opt-in shallow growth runtime memory attribution."""
    if not args.growth_memory_profile:
        return
    _log_growth_runtime_memory_profile(
        runner=runner,
        generation=generation,
        event=event,
        node_count=node_count,
        branch_count=branch_count,
        sample_nodes=args.growth_memory_profile_sample_nodes,
        top_n=args.growth_memory_profile_top_n,
    )
    LOGGER.info(
        "[growth-recursive-profile-debug] event=%s enabled=%s events=%s "
        "max_objects=%s max_depth=%s complete_map=%s",
        event,
        args.growth_memory_profile_recursive,
        args.growth_memory_profile_recursive_events,
        args.growth_memory_profile_recursive_max_objects,
        args.growth_memory_profile_recursive_max_depth,
        args.growth_memory_profile_recursive_complete_map,
    )
    if (
        args.growth_memory_profile_recursive
        and event in args.growth_memory_profile_recursive_events
    ):
        _log_growth_recursive_memory_profile(
            runner=runner,
            generation=generation,
            event=event,
            node_count=node_count,
            branch_count=branch_count,
            max_objects=args.growth_memory_profile_recursive_max_objects,
            max_depth=args.growth_memory_profile_recursive_max_depth,
            top_n=args.growth_memory_profile_top_n,
            complete_map=args.growth_memory_profile_recursive_complete_map,
            max_depth_explicit=(
                args.growth_memory_profile_recursive_max_depth_explicit
            ),
            context_node_cap=(args.growth_memory_profile_recursive_context_node_cap),
        )


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
    manifest = load_generation_manifest(paths=paths, generation=generation)
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
    manifest = save_dataset_manifest_status(
        paths=paths,
        manifest=manifest,
        dataset_status="extracting_rows",
        timestamp_utc=timestamp_utc,
    )
    try:
        tree_snapshot_path = paths.resolve_work_dir_path(
            require_manifest_tree_snapshot_path(manifest)
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
                pipeline_manifest_path(paths, generation),
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
        previous_record_status = resolve_previous_pipeline_record_status(
            paths=paths,
            generation=generation,
        )
        previous_frontier_status = resolve_previous_pipeline_frontier_status(
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
        save_pipeline_manifest(manifest, pipeline_manifest_path(paths, generation))
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
            pipeline_manifest_path(paths, generation),
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
        save_dataset_manifest_status(
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
    manifest = load_generation_manifest(paths=paths, generation=generation)
    lower_bound = training_lower_bound_details(paths)
    if generation <= lower_bound.generation:
        LOGGER.info(
            "[pipeline] training_skip generation=%s reason=stale_generation lower_bound_generation=%s",
            generation,
            lower_bound.generation,
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
    training_cycle_started_at = time.perf_counter()
    LOGGER.info("[pipeline] training_start generation=%s", generation)
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="start",
    )
    manifest = save_training_manifest_status(
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
        save_training_cursor_started(paths=paths, generation=generation)
        rows_path = paths.resolve_work_dir_path(require_manifest_rows_path(manifest))
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
            assert rows is not None
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
                training_rows_path = training_rows_subset_path(paths, generation)
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
        else:
            manifest_metadata["evaluator_diagnostics_max_rows"] = (
                args.evaluator_diagnostics_max_rows
            )
            manifest_metadata["evaluator_diagnostics_sample_policy"] = "first_n"
        manifest = replace(manifest, metadata=manifest_metadata)
        log_training_cycle_start(
            generation=generation,
            rows_path=training_rows_path,
            row_count=original_training_rows,
            row_format=rows_source.format_kind,
            chunk_size=args.training_row_chunk_size
            if rows_source.format_kind == "jsonl"
            else None,
            evaluator_names=tuple(resolved_evaluators_config.evaluators),
            active_model_cursor_summary=_training_active_model_cursor_summary(
                lower_bound
            ),
            max_rows=args.training_max_rows,
        )
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
        save_pipeline_manifest(manifest, pipeline_manifest_path(paths, generation))
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
        current_active_generation = active_model_generation_for_training_guard(paths)
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
        save_training_cursor_completed(paths=paths, generation=generation)
        LOGGER.info(
            "[pipeline] training_done generation=%s selected=%s",
            generation,
            training_result.selected_evaluator_name,
        )
        log_training_cycle_done(
            generation=generation,
            elapsed_s=time.perf_counter() - training_cycle_started_at,
            evaluators_done=len(training_result.evaluator_results),
            evaluator_count=len(resolved_evaluators_config.evaluators),
            row_count=training_rows_used,
            outputs=paths.relative_to_work_dir(
                paths.model_generation_dir_for_generation(generation)
            ),
            active_model_candidate=training_result.model_bundle_paths[
                training_result.selected_evaluator_name
            ],
        )
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="done",
            selected=training_result.selected_evaluator_name,
        )
    except Exception:
        timestamp_utc = _now_timestamp_utc()
        save_training_manifest_status(
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
