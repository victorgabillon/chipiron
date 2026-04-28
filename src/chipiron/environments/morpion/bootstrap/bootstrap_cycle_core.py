"""Shared cycle primitives for single-process and artifact-pipeline bootstrap modes."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anemone.training_export import load_training_tree_snapshot

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)

from .bootstrap_errors import (
    IncompatibleMorpionResumeArtifactError,
    MissingActiveMorpionEvaluatorError,
    MissingBootstrapDatasetRowsError,
    MissingBootstrapFrontierStatusError,
    MissingBootstrapRecordStatusError,
    MissingBootstrapSelectedEvaluatorError,
    MissingForcedMorpionEvaluatorBundleError,
    MissingSavedBootstrapArtifactError,
    NoSelectableMorpionEvaluatorError,
    UnknownActiveMorpionEvaluatorError,
    UnknownForcedMorpionEvaluatorError,
    UnsupportedMorpionRuntimeReconfigurationError,
)
from .bootstrap_memory import log_after_cycle_gc
from .bootstrap_paths import (
    DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS,
    DEFAULT_KEEP_LATEST_TREE_EXPORTS,
    MorpionBootstrapPaths,
    prune_generation_files,
)
from .config import BOOTSTRAP_CONFIG_HASH_METADATA_KEY, MorpionBootstrapConfig
from .control import (
    BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY,
    BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY,
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapRuntimeControl,
    bootstrap_control_to_dict,
    bootstrap_runtime_control_to_dict,
    effective_runtime_config_from_metadata,
    effective_runtime_config_sha256,
    effective_runtime_config_to_dict,
)
from .dataset_family_targets import apply_dataset_family_target_policy
from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
from .evaluator_diagnostics import (
    append_evaluator_training_diagnostics_history,
    build_evaluator_training_diagnostics,
    diagnostics_path,
    load_previous_evaluator_for_diagnostics,
    save_evaluator_training_diagnostics,
)
from .history import (
    MorpionBootstrapArtifacts,
    MorpionBootstrapDatasetStatus,
    MorpionBootstrapEvent,
    MorpionBootstrapHistoryRecorder,
    MorpionBootstrapTrainingStatus,
    MorpionBootstrapTreeStatus,
    MorpionEvaluatorMetrics,
)
from .memory_diagnostics import MemoryDiagnostics
from .pipeline_artifacts import (
    MorpionPipelineActiveModel,
    MorpionPipelineDatasetStatus,
    MorpionPipelineGenerationManifest,
    MorpionPipelineTrainingStatus,
    save_pipeline_active_model,
    save_pipeline_dataset_status_file,
    save_pipeline_manifest,
    save_pipeline_training_status_file,
)
from .pipeline_config import MorpionEvaluatorUpdatePolicy
from .record_status import (
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
    carried_forward_morpion_frontier_status,
    default_morpion_record_status,
    morpion_bootstrap_experiment_metadata,
    resolve_frontier_status_for_cycle,
    resolve_frontier_status_for_cycle_with_metadata,
    resolve_record_status_for_cycle,
)
from .run_state import MorpionBootstrapRunState
from .search_runner_protocol import MorpionSearchRunner

if TYPE_CHECKING:
    from anemone.training_export import TrainingTreeSnapshot

    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .bootstrap_loop import MorpionBootstrapArgs

LOGGER = logging.getLogger(__name__)

RUNTIME_CHECKPOINT_METADATA_KEY = "runtime_checkpoint_path"
TRAINING_SKIPPED_REASON_METADATA_KEY = "training_skipped_reason"
EMPTY_DATASET_TRAINING_SKIPPED_REASON = "empty_dataset"

_CURRENT_TREE_STATUS_TYPE_ERROR = (
    "Morpion bootstrap runner current_tree_status() must return "
    "MorpionBootstrapTreeStatus or a mapping."
)


def _tree_status_int_error(field_name: str) -> TypeError:
    return TypeError(
        f"Morpion bootstrap tree-status field `{field_name}` must be an int or null."
    )


def _tree_status_mapping_error(field_name: str) -> TypeError:
    return TypeError(
        f"Morpion bootstrap tree-status field `{field_name}` must be a mapping."
    )


def _tree_status_key_error(field_name: str) -> TypeError:
    return TypeError(
        f"Morpion bootstrap tree-status field `{field_name}` must use integer-like keys."
    )


def _tree_status_value_error(field_name: str) -> TypeError:
    return TypeError(
        f"Morpion bootstrap tree-status field `{field_name}` must use int values."
    )


def _unknown_pipeline_mode_error(pipeline_mode: object) -> ValueError:
    return ValueError(f"Unknown Morpion pipeline mode: {pipeline_mode!r}")


def _unknown_evaluator_update_policy_error(policy: object) -> ValueError:
    return ValueError(f"Unknown Morpion evaluator update policy: {policy!r}")


@dataclass(frozen=True, slots=True)
class BootstrapDatasetBuildResult:
    """Artifacts and statuses produced by the bootstrap dataset build phase."""

    snapshot: TrainingTreeSnapshot
    rows: MorpionSupervisedRows
    record_status: MorpionBootstrapRecordStatus
    frontier_status: MorpionBootstrapFrontierStatus
    relative_tree_snapshot_path: str
    relative_rows_path: str
    dataset_elapsed_s: float
    num_rows: int


@dataclass(frozen=True, slots=True)
class BootstrapTrainingResult:
    """Metrics and selected evaluator produced by bootstrap model training."""

    evaluator_metrics: dict[str, MorpionEvaluatorMetrics]
    model_bundle_paths: dict[str, str]
    selected_evaluator_name: str
    training_duration_s: float


@dataclass(frozen=True, slots=True)
class ResolvedActiveMorpionModelBundle:
    """Resolved active evaluator identity and bundle path for one cycle."""

    active_evaluator_name: str | None
    model_bundle_path: Path | None


def build_bootstrap_event(
    *,
    cycle_index: int,
    generation: int,
    timestamp_utc: str,
    tree_status: MorpionBootstrapTreeStatus,
    tree_snapshot_path: str | None,
    rows_path: str | None,
    dataset_num_rows: int | None,
    dataset_num_samples: int | None,
    training_triggered: bool,
    frontier_status: MorpionBootstrapFrontierStatus | None = None,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics] | None = None,
    model_bundle_paths: Mapping[str, str] | None = None,
    record_status: MorpionBootstrapRecordStatus | None = None,
    event_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MorpionBootstrapEvent:
    """Build one structured bootstrap history event from cycle outputs."""
    return MorpionBootstrapEvent(
        event_id=f"cycle_{cycle_index:06d}" if event_id is None else event_id,
        cycle_index=cycle_index,
        generation=generation,
        timestamp_utc=timestamp_utc,
        tree=tree_status,
        dataset=MorpionBootstrapDatasetStatus(
            num_rows=dataset_num_rows,
            num_samples=dataset_num_samples,
        ),
        training=MorpionBootstrapTrainingStatus(triggered=training_triggered),
        record=default_morpion_record_status()
        if record_status is None
        else record_status,
        frontier=carried_forward_morpion_frontier_status(frontier_status),
        artifacts=MorpionBootstrapArtifacts(
            tree_snapshot_path=tree_snapshot_path,
            rows_path=rows_path,
            model_bundle_paths=dict(model_bundle_paths or {}),
        ),
        evaluators=dict(evaluator_metrics or {}),
        metadata=dict(metadata or {}),
    )


def should_save_progress(
    *,
    current_tree_size: int,
    tree_size_at_last_save: int,
    now_unix_s: float,
    last_save_unix_s: float | None,
    save_after_tree_growth_factor: float,
    save_after_seconds: float,
) -> bool:
    """Return whether the bootstrap loop should checkpoint and retrain now."""
    if last_save_unix_s is None:
        return True
    if (
        tree_size_at_last_save > 0
        and current_tree_size >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return True
    return now_unix_s - last_save_unix_s >= save_after_seconds


def validate_dataset_family_target_args(args: MorpionBootstrapArgs) -> None:
    """Validate dataset-family target blending arguments."""
    if not 0.0 <= args.dataset_family_prediction_blend <= 1.0:
        raise ValueError("dataset_family_prediction_blend must be between 0 and 1.")


def validate_pipeline_mode(args: MorpionBootstrapArgs) -> None:
    """Validate the configured bootstrap pipeline mode."""
    if args.pipeline_mode == "single_process":
        return
    if args.pipeline_mode == "artifact_pipeline":
        return
    raise _unknown_pipeline_mode_error(args.pipeline_mode)


def require_single_process_mode(args: MorpionBootstrapArgs) -> None:
    """Require explicit single-process mode for the canonical loop entrypoint."""
    if args.pipeline_mode != "single_process":
        raise NotImplementedError(
            "run_morpion_bootstrap_loop only supports pipeline_mode='single_process'. "
            "Use the dedicated artifact-pipeline stage entrypoints instead."
        )


def reevaluate_tree_for_policy(policy: MorpionEvaluatorUpdatePolicy) -> bool:
    """Resolve whether the runner should reevaluate existing tree nodes."""
    if policy == "future_only":
        return False
    if policy == "reevaluate_all":
        return True
    if policy == "reevaluate_frontier":
        raise NotImplementedError(
            "Morpion evaluator_update_policy='reevaluate_frontier' is reserved "
            "for future partial tree reevaluation."
        )
    raise _unknown_evaluator_update_policy_error(policy)


def extract_rows_from_training_snapshot(
    *,
    args: MorpionBootstrapArgs,
    snapshot: TrainingTreeSnapshot,
    generation: int,
) -> MorpionSupervisedRows:
    """Extract and post-process supervised rows from one training snapshot."""
    rows = training_tree_snapshot_to_morpion_supervised_rows(
        snapshot,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        max_rows=args.max_rows,
        use_backed_up_value=args.use_backed_up_value,
        metadata={"bootstrap_generation": generation},
    )
    return apply_dataset_family_target_policy(
        snapshot=snapshot,
        rows=rows,
        family_target_policy=args.dataset_family_target_policy,
        family_prediction_blend=args.dataset_family_prediction_blend,
        use_backed_up_value=args.use_backed_up_value,
    )


def build_and_save_dataset_for_generation(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    run_state: MorpionBootstrapRunState,
    generation: int,
    memory: MemoryDiagnostics,
) -> BootstrapDatasetBuildResult:
    """Build, annotate, and persist the supervised rows for one generation."""
    training_snapshot_path = paths.tree_snapshot_path_for_generation(generation)
    rows_path = paths.rows_path_for_generation(generation)
    dataset_started_at = time.perf_counter()
    LOGGER.info("[dataset] build_start snapshot=%s", str(training_snapshot_path))
    memory.log("before_snapshot_load_or_export")
    runner.export_training_tree_snapshot(training_snapshot_path)
    if not training_snapshot_path.is_file():
        raise MissingSavedBootstrapArtifactError(
            action="runner.export_training_tree_snapshot()",
            artifact_path=training_snapshot_path,
        )
    snapshot = load_training_tree_snapshot(training_snapshot_path)
    memory.log("after_snapshot_load_or_export")
    LOGGER.info("[record] resolve_start nodes=%s", len(snapshot.nodes))
    record_started_at = time.perf_counter()
    record_status: MorpionBootstrapRecordStatus | None = None
    try:
        record_status = cast(
            "MorpionBootstrapRecordStatus | None",
            resolve_record_status_for_cycle(
                snapshot=snapshot,
                previous_record_status=run_state.latest_record_status,
            ),
        )
    finally:
        LOGGER.info(
            "[record] resolve_done elapsed=%.3fs best_total_points=%s",
            time.perf_counter() - record_started_at,
            None if record_status is None else record_status.current_best_total_points,
        )
    if record_status is None:
        raise MissingBootstrapRecordStatusError()

    LOGGER.info("[frontier] resolve_start nodes=%s", len(snapshot.nodes))
    frontier_started_at = time.perf_counter()
    frontier_candidate_count = 0
    frontier_status: MorpionBootstrapFrontierStatus | None = None
    try:
        frontier_resolution = resolve_frontier_status_for_cycle_with_metadata(
            snapshot=snapshot,
            previous_frontier_status=run_state.latest_frontier_status,
        )
        frontier_candidate_count = frontier_resolution.candidate_count
        frontier_status = cast(
            "MorpionBootstrapFrontierStatus | None",
            frontier_resolution.status,
        )
    finally:
        LOGGER.info(
            "[frontier] resolve_done elapsed=%.3fs candidates=%s best_total_points=%s method=depth_metadata",
            time.perf_counter() - frontier_started_at,
            frontier_candidate_count,
            None if frontier_status is None else frontier_status.current_best_total_points,
        )
    if frontier_status is None:
        raise MissingBootstrapFrontierStatusError()

    LOGGER.info("[dataset] extract_start snapshot_nodes=%s", len(snapshot.nodes))
    memory.log("before_dataset_extract")
    extract_started_at = time.perf_counter()
    rows: MorpionSupervisedRows | None = None
    try:
        rows = extract_rows_from_training_snapshot(
            args=args,
            snapshot=snapshot,
            generation=generation,
        )
        memory.log("after_dataset_extract")
        memory.log("after_dataset_family_target_policy")
    finally:
        LOGGER.info(
            "[dataset] extract_done rows=%s elapsed=%.3fs",
            None if rows is None else len(rows.rows),
            time.perf_counter() - extract_started_at,
        )
    if rows is None:
        raise MissingBootstrapDatasetRowsError()

    LOGGER.info(
        "[dataset] family_targets policy=%s blend=%.3f rows_in_exact_family=%s num_exact_families=%s effective_minus_raw_mean_abs=%s effective_minus_raw_max_abs=%s",
        rows.metadata.get("dataset_family_target_policy"),
        rows.metadata.get("dataset_family_prediction_blend"),
        rows.metadata.get("fraction_rows_in_exact_family"),
        rows.metadata.get("num_exact_families"),
        rows.metadata.get("effective_minus_raw_mean_abs"),
        rows.metadata.get("effective_minus_raw_max_abs"),
    )
    LOGGER.info("[dataset] save_start path=%s", str(rows_path))
    rows_save_started_at = time.perf_counter()
    try:
        save_morpion_supervised_rows(rows, rows_path)
    finally:
        LOGGER.info(
            "[dataset] save_done elapsed=%.3fs",
            time.perf_counter() - rows_save_started_at,
        )
    memory.log("after_rows_save")
    num_rows = len(rows.rows)
    dataset_elapsed_s = time.perf_counter() - dataset_started_at
    LOGGER.info(
        "[dataset] build_done rows=%s output=%s elapsed=%.3fs",
        num_rows,
        str(rows_path),
        dataset_elapsed_s,
    )

    return BootstrapDatasetBuildResult(
        snapshot=snapshot,
        rows=rows,
        record_status=record_status,
        frontier_status=frontier_status,
        relative_tree_snapshot_path=paths.relative_to_work_dir(training_snapshot_path),
        relative_rows_path=paths.relative_to_work_dir(rows_path),
        dataset_elapsed_s=dataset_elapsed_s,
        num_rows=num_rows,
    )


def train_and_select_evaluators(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    rows: MorpionSupervisedRows,
    rows_path: Path,
    generation: int,
    timestamp_utc: str,
    resolved_evaluators_config: MorpionEvaluatorsConfig,
    resolved_control: MorpionBootstrapControl,
    memory: MemoryDiagnostics,
) -> BootstrapTrainingResult:
    """Train configured evaluators and select the active evaluator for search."""
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] = {}
    model_bundle_paths: dict[str, str] = {}
    training_started_at = time.perf_counter()
    memory.log("before_training")
    LOGGER.info(
        "[train] start evaluators=%s rows=%s",
        len(resolved_evaluators_config.evaluators),
        len(rows.rows),
    )
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        previous_model = load_previous_evaluator_for_diagnostics(
            resolve_previous_model_bundle_path(
                paths=paths,
                run_state=run_state,
                evaluator_name=evaluator_name,
            )
        )
        LOGGER.info("[train] evaluator_start name=%s", evaluator_name)
        evaluator_started_at = time.perf_counter()
        trained_model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_bundle_path,
                batch_size=spec.batch_size,
                num_epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                shuffle=args.shuffle,
                model_kind=spec.model_type,
                feature_subset_name=spec.feature_subset_name,
                feature_names=spec.feature_names,
                hidden_sizes=spec.hidden_sizes,
            )
        )
        memory.log("after_model_save")
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=float(metrics["final_loss"]),
            num_epochs=int(metrics["num_epochs"]),
            num_samples=int(metrics["num_samples"]),
        )
        LOGGER.info(
            "[train] evaluator_done name=%s final_loss=%s elapsed=%.3fs",
            evaluator_name,
            evaluator_metrics[evaluator_name].final_loss,
            time.perf_counter() - evaluator_started_at,
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(
            model_bundle_path
        )
        persist_evaluator_training_diagnostics(
            paths=paths,
            generation=generation,
            evaluator_name=evaluator_name,
            rows=rows,
            created_at=timestamp_utc,
            spec=spec,
            model_before=previous_model,
            model_after=trained_model,
        )
        memory.log("after_diagnostics")
        del previous_model
        del trained_model
        log_after_cycle_gc(memory, tag=f"after_evaluator:{evaluator_name}")

    LOGGER.info("[train] selection_start evaluators=%s", len(evaluator_metrics))
    selection_started_at = time.perf_counter()
    selected_evaluator_name: str | None = None
    try:
        selected_evaluator_name = cast(
            "str | None",
            select_or_force_active_evaluator_name(
                evaluator_metrics=evaluator_metrics,
                force_evaluator=resolved_control.force_evaluator,
            ),
        )
    finally:
        LOGGER.info(
            "[train] selection_done elapsed=%.3fs selected=%s policy=lowest_final_loss",
            time.perf_counter() - selection_started_at,
            selected_evaluator_name,
        )
    if selected_evaluator_name is None:
        raise MissingBootstrapSelectedEvaluatorError()

    training_duration_s = time.perf_counter() - training_started_at
    memory.log("after_training")
    LOGGER.info("[train] done elapsed=%.3fs", training_duration_s)
    return BootstrapTrainingResult(
        evaluator_metrics=evaluator_metrics,
        model_bundle_paths=model_bundle_paths,
        selected_evaluator_name=selected_evaluator_name,
        training_duration_s=training_duration_s,
    )


def build_no_save_run_state(
    *,
    run_state: MorpionBootstrapRunState,
    resolved_active_model: ResolvedActiveMorpionModelBundle,
    resolved_control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    cycle_index: int,
) -> MorpionBootstrapRunState:
    """Build the carried-forward run state for a no-save cycle."""
    return MorpionBootstrapRunState(
        generation=run_state.generation,
        cycle_index=cycle_index,
        latest_tree_snapshot_path=run_state.latest_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=resolved_active_model.active_evaluator_name,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        last_save_unix_s=run_state.last_save_unix_s,
        latest_runtime_checkpoint_path=run_state.latest_runtime_checkpoint_path,
        latest_record_status=run_state.latest_record_status,
        latest_frontier_status=run_state.latest_frontier_status,
        metadata=next_metadata(
            run_state.metadata,
            relative_runtime_checkpoint_path=None,
            control=resolved_control,
            effective_runtime_config=effective_runtime_config,
        ),
    )


def record_no_save_cycle_event(
    *,
    history_recorder: MorpionBootstrapHistoryRecorder,
    cycle_index: int,
    timestamp_utc: str,
    tree_status: MorpionBootstrapTreeStatus,
    frontier_status: MorpionBootstrapFrontierStatus,
    run_state: MorpionBootstrapRunState,
    next_run_state: MorpionBootstrapRunState,
    resolved_control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Record the history event for a no-save cycle."""
    history_recorder.record(
        build_bootstrap_event(
            cycle_index=cycle_index,
            generation=next_run_state.generation,
            timestamp_utc=timestamp_utc,
            tree_status=tree_status,
            tree_snapshot_path=None,
            rows_path=None,
            dataset_num_rows=None,
            dataset_num_samples=None,
            training_triggered=False,
            frontier_status=frontier_status,
            record_status=resolve_record_status_for_cycle(
                snapshot=None,
                previous_record_status=run_state.latest_record_status,
            ),
            metadata=build_event_metadata(
                active_evaluator_name=next_run_state.active_evaluator_name,
                config_hash=bootstrap_config_hash_from_metadata(run_state.metadata),
                forced_evaluator=resolved_control.force_evaluator,
                runtime_control=resolved_control.runtime,
                effective_runtime_config=effective_runtime_config,
            ),
        )
    )


def timestamp_utc_from_unix_s(timestamp_unix_s: float) -> str:
    """Format one Unix timestamp as an ISO 8601 UTC string."""
    timestamp = datetime.fromtimestamp(timestamp_unix_s, tz=UTC)
    timespec = "seconds" if timestamp.microsecond == 0 else "microseconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


def prune_saved_generation_artifacts(paths: MorpionBootstrapPaths) -> None:
    """Prune retained tree exports and checkpoints only after run-state persistence."""
    LOGGER.info(
        "[retention] prune_start kind=checkpoint keep_latest=%s",
        DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS,
    )
    prune_generation_files(
        paths.runtime_checkpoint_dir,
        keep_latest=DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS,
    )
    LOGGER.info(
        "[retention] prune_start kind=tree_export keep_latest=%s",
        DEFAULT_KEEP_LATEST_TREE_EXPORTS,
    )
    prune_generation_files(
        paths.tree_snapshot_dir,
        keep_latest=DEFAULT_KEEP_LATEST_TREE_EXPORTS,
    )


def save_trigger_reason(
    *,
    current_tree_size: int,
    tree_size_at_last_save: int,
    now_unix_s: float,
    last_save_unix_s: float | None,
    save_after_tree_growth_factor: float,
    save_after_seconds: float,
) -> str | None:
    """Return the reason a save trigger fired, if any."""
    if last_save_unix_s is None:
        return "first_cycle"
    if (
        tree_size_at_last_save > 0
        and current_tree_size >= tree_size_at_last_save * save_after_tree_growth_factor
    ):
        return "growth_factor_reached"
    if now_unix_s - last_save_unix_s >= save_after_seconds:
        return "time_elapsed"
    return None


def resolve_active_model_bundle(
    *,
    paths: MorpionBootstrapPaths,
    latest_model_bundle_paths: Mapping[str, str] | None,
    active_evaluator_name: str | None,
    force_evaluator: str | None = None,
) -> ResolvedActiveMorpionModelBundle:
    """Resolve the active evaluator identity and bundle path for runner bootstrap."""
    if not latest_model_bundle_paths:
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=None,
            model_bundle_path=None,
        )
    if force_evaluator is not None:
        selected_path = latest_model_bundle_paths.get(force_evaluator)
        if selected_path is None:
            raise MissingForcedMorpionEvaluatorBundleError(force_evaluator)
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=force_evaluator,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    if active_evaluator_name is not None:
        selected_path = latest_model_bundle_paths.get(active_evaluator_name)
        if selected_path is None:
            raise UnknownActiveMorpionEvaluatorError(active_evaluator_name)
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=active_evaluator_name,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    if len(latest_model_bundle_paths) == 1:
        inferred_active_evaluator_name, selected_path = next(
            iter(latest_model_bundle_paths.items())
        )
        return ResolvedActiveMorpionModelBundle(
            active_evaluator_name=inferred_active_evaluator_name,
            model_bundle_path=paths.resolve_work_dir_path(selected_path),
        )
    raise MissingActiveMorpionEvaluatorError


def resolve_previous_model_bundle_path(
    *,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    evaluator_name: str,
) -> Path | None:
    """Return the previous evaluator bundle path when one exists."""
    if run_state.latest_model_bundle_paths is None:
        return None
    relative_path = run_state.latest_model_bundle_paths.get(evaluator_name)
    if relative_path is None:
        return None
    return paths.resolve_work_dir_path(relative_path)


def persist_evaluator_training_diagnostics(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    evaluator_name: str,
    rows: MorpionSupervisedRows,
    created_at: str,
    spec: MorpionEvaluatorSpec,
    model_before: MorpionRegressor | None,
    model_after: MorpionRegressor,
) -> None:
    """Persist evaluator diagnostics without changing bootstrap semantics."""
    try:
        diagnostics = build_evaluator_training_diagnostics(
            generation=generation,
            evaluator_name=evaluator_name,
            rows=rows,
            created_at=created_at,
            feature_subset_name=spec.feature_subset_name,
            feature_names=spec.feature_names,
            model_before=model_before,
            model_after=model_after,
        )
        output_path = diagnostics_path(paths.work_dir, generation, evaluator_name)
        save_evaluator_training_diagnostics(diagnostics, output_path)
        append_evaluator_training_diagnostics_history(diagnostics, paths.work_dir)
        LOGGER.info(
            "[diagnostics] saved generation=%s evaluator=%s path=%s examples=%s worst=%s",
            generation,
            evaluator_name,
            output_path,
            len(diagnostics.representative_examples),
            len(diagnostics.worst_examples),
        )
    except Exception:
        LOGGER.exception(
            "[diagnostics] save_failed generation=%s evaluator=%s",
            generation,
            evaluator_name,
        )


def build_event_metadata(
    *,
    active_evaluator_name: str | None,
    selected_evaluator_name: str | None = None,
    config_hash: str | None = None,
    forced_evaluator: str | None = None,
    runtime_control: MorpionBootstrapRuntimeControl | None = None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Build history metadata describing the active and selected evaluators."""
    metadata = morpion_bootstrap_experiment_metadata()
    if active_evaluator_name is not None:
        metadata["active_evaluator_name"] = active_evaluator_name
    if selected_evaluator_name is not None:
        metadata["selected_evaluator_name"] = selected_evaluator_name
        metadata["selection_policy"] = "lowest_final_loss"
    if config_hash is not None:
        metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    if forced_evaluator is not None:
        metadata["forced_evaluator"] = forced_evaluator
    if runtime_control is not None:
        metadata[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
            bootstrap_runtime_control_to_dict(runtime_control)
        )
    if effective_runtime_config is not None:
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
            effective_runtime_config_to_dict(effective_runtime_config)
        )
        metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
            effective_runtime_config_sha256(effective_runtime_config)
        )
    if training_skipped_reason is not None:
        metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return metadata


def pipeline_metadata(
    *,
    args: MorpionBootstrapArgs,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Build pipeline-manifest metadata mirrored from the current loop config."""
    metadata: dict[str, object] = {
        "pipeline_mode": args.pipeline_mode,
        "evaluator_update_policy": args.evaluator_update_policy,
    }
    if training_skipped_reason is not None:
        metadata[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return metadata


def write_pipeline_status_files(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    timestamp_utc: str,
    dataset_status: MorpionPipelineDatasetStatus,
    training_status: MorpionPipelineTrainingStatus,
    metadata: Mapping[str, object],
) -> None:
    """Persist lightweight pipeline stage-status files for one generation."""
    save_pipeline_dataset_status_file(
        generation=generation,
        dataset_status=dataset_status,
        updated_at_utc=timestamp_utc,
        metadata=metadata,
        path=paths.pipeline_dataset_status_path_for_generation(generation),
    )
    save_pipeline_training_status_file(
        generation=generation,
        training_status=training_status,
        updated_at_utc=timestamp_utc,
        metadata=metadata,
        path=paths.pipeline_training_status_path_for_generation(generation),
    )


def write_pipeline_manifest_for_generation(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    timestamp_utc: str,
    relative_runtime_checkpoint_path: str | None,
    relative_tree_snapshot_path: str,
    relative_rows_path: str | None,
    model_bundle_paths: Mapping[str, str],
    selected_evaluator_name: str | None,
    dataset_status: MorpionPipelineDatasetStatus,
    training_status: MorpionPipelineTrainingStatus,
    metadata: Mapping[str, object],
) -> None:
    """Persist one pipeline manifest mirroring saved bootstrap artifacts."""
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=generation,
            created_at_utc=timestamp_utc,
            runtime_checkpoint_path=relative_runtime_checkpoint_path,
            tree_snapshot_path=relative_tree_snapshot_path,
            rows_path=relative_rows_path,
            model_bundle_paths=dict(model_bundle_paths),
            selected_evaluator_name=selected_evaluator_name,
            dataset_status=dataset_status,
            training_status=training_status,
            metadata=dict(metadata),
        ),
        paths.pipeline_manifest_path_for_generation(generation),
    )
    write_pipeline_status_files(
        paths=paths,
        generation=generation,
        timestamp_utc=timestamp_utc,
        dataset_status=dataset_status,
        training_status=training_status,
        metadata=metadata,
    )


def write_pipeline_active_model(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    selected_evaluator_name: str,
    model_bundle_paths: Mapping[str, str],
    timestamp_utc: str,
) -> None:
    """Persist the currently selected active model for pipeline consumers."""
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=generation,
            evaluator_name=selected_evaluator_name,
            model_bundle_path=model_bundle_paths[selected_evaluator_name],
            updated_at_utc=timestamp_utc,
            metadata={"selection_policy": "lowest_final_loss"},
        ),
        paths.pipeline_active_model_path,
    )


def resolve_tree_status(
    runner: MorpionSearchRunner,
    *,
    current_tree_size: int,
) -> MorpionBootstrapTreeStatus:
    """Return the best available tree monitoring status for the current runner."""
    current_tree_status = getattr(runner, "current_tree_status", None)
    if callable(current_tree_status):
        raw_status = current_tree_status()
        if isinstance(raw_status, MorpionBootstrapTreeStatus):
            return MorpionBootstrapTreeStatus(
                num_nodes=current_tree_size,
                num_expanded_nodes=raw_status.num_expanded_nodes,
                num_simulations=raw_status.num_simulations,
                root_visit_count=raw_status.root_visit_count,
                min_depth_present=raw_status.min_depth_present,
                max_depth_present=raw_status.max_depth_present,
                depth_node_counts=dict(raw_status.depth_node_counts),
            )
        if isinstance(raw_status, Mapping):
            status_mapping = cast("Mapping[str, object]", raw_status)
            return MorpionBootstrapTreeStatus(
                num_nodes=current_tree_size,
                num_expanded_nodes=optional_tree_int(
                    status_mapping.get("num_expanded_nodes"),
                    field_name="num_expanded_nodes",
                ),
                num_simulations=optional_tree_int(
                    status_mapping.get("num_simulations"),
                    field_name="num_simulations",
                ),
                root_visit_count=optional_tree_int(
                    status_mapping.get("root_visit_count"),
                    field_name="root_visit_count",
                ),
                min_depth_present=optional_tree_int(
                    status_mapping.get("min_depth_present"),
                    field_name="min_depth_present",
                ),
                max_depth_present=optional_tree_int(
                    status_mapping.get("max_depth_present"),
                    field_name="max_depth_present",
                ),
                depth_node_counts=optional_tree_int_mapping(
                    status_mapping.get("depth_node_counts"),
                    field_name="depth_node_counts",
                ),
            )
        raise TypeError(_CURRENT_TREE_STATUS_TYPE_ERROR)
    return MorpionBootstrapTreeStatus(num_nodes=current_tree_size)


def optional_tree_int(value: object, *, field_name: str) -> int | None:
    """Return one optional integer tree-status field or raise clearly."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise _tree_status_int_error(field_name)
    if isinstance(value, int):
        return value
    raise _tree_status_int_error(field_name)


def optional_tree_int_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[int, int]:
    """Return one optional int-to-int tree-status mapping or raise clearly."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _tree_status_mapping_error(field_name)
    mapping: dict[int, int] = {}
    raw_mapping = cast("Mapping[object, object]", value)
    for raw_key, raw_item_value in raw_mapping.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int | str):
            raise _tree_status_key_error(field_name)
        if isinstance(raw_item_value, bool) or not isinstance(raw_item_value, int):
            raise _tree_status_value_error(field_name)
        try:
            coerced_key = int(raw_key)
        except ValueError as exc:
            raise _tree_status_key_error(field_name) from exc
        mapping[coerced_key] = raw_item_value
    return mapping


def resolve_runtime_restore_path(
    *,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
) -> Path | None:
    """Resolve the best available persisted runtime restore path for one cycle."""
    from .anemone_runner import (
        InvalidMorpionSearchCheckpointError,
        load_morpion_search_checkpoint_payload,
    )

    candidates: list[tuple[str, Path | None]] = [
        (
            "run_state.latest_runtime_checkpoint_path",
            paths.resolve_work_dir_path(run_state.latest_runtime_checkpoint_path),
        ),
    ]
    metadata_runtime_checkpoint = run_state.metadata.get(
        RUNTIME_CHECKPOINT_METADATA_KEY
    )
    if isinstance(metadata_runtime_checkpoint, str):
        candidates.append(
            (
                "run_state.metadata.runtime_checkpoint_path",
                paths.resolve_work_dir_path(metadata_runtime_checkpoint),
            )
        )
    if run_state.generation > 0:
        candidates.append(
            (
                "canonical search_checkpoints path for latest generation",
                paths.runtime_checkpoint_path_for_generation(run_state.generation),
            )
        )
    if not run_state.latest_model_bundle_paths:
        candidates.append(
            (
                "run_state.latest_tree_snapshot_path",
                paths.resolve_work_dir_path(run_state.latest_tree_snapshot_path),
            )
        )

    seen_paths: set[Path] = set()
    first_incompatible_error: IncompatibleMorpionResumeArtifactError | None = None
    for source, candidate_path in candidates:
        if candidate_path is None or candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        if not candidate_path.is_file():
            continue
        LOGGER.info(
            "[checkpoint] candidate_validate_start source=%s path=%s",
            source,
            str(candidate_path),
        )
        try:
            load_morpion_search_checkpoint_payload(candidate_path)
        except InvalidMorpionSearchCheckpointError as exc:
            LOGGER.info(
                "[checkpoint] candidate_validate_invalid source=%s path=%s reason=%s",
                source,
                str(candidate_path),
                str(exc),
            )
            if first_incompatible_error is None:
                first_incompatible_error = IncompatibleMorpionResumeArtifactError(
                    source=source,
                    artifact_path=candidate_path,
                    reason=str(exc),
                )
            continue
        LOGGER.info(
            "[checkpoint] candidate_validate_done source=%s path=%s",
            source,
            str(candidate_path),
        )
        return candidate_path

    if first_incompatible_error is not None:
        raise first_incompatible_error
    return None


def bootstrap_config_hash_from_metadata(metadata: Mapping[str, object]) -> str | None:
    """Return the persisted bootstrap config hash from one metadata mapping."""
    value = metadata.get(BOOTSTRAP_CONFIG_HASH_METADATA_KEY)
    return value if isinstance(value, str) else None


def with_config_hash_metadata(
    run_state: MorpionBootstrapRunState,
    *,
    config_hash: str,
) -> MorpionBootstrapRunState:
    """Return one run state with the accepted bootstrap config hash recorded."""
    next_metadata = dict(run_state.metadata)
    next_metadata[BOOTSTRAP_CONFIG_HASH_METADATA_KEY] = config_hash
    return MorpionBootstrapRunState(
        generation=run_state.generation,
        cycle_index=run_state.cycle_index,
        latest_tree_snapshot_path=run_state.latest_tree_snapshot_path,
        latest_rows_path=run_state.latest_rows_path,
        latest_model_bundle_paths=None
        if run_state.latest_model_bundle_paths is None
        else dict(run_state.latest_model_bundle_paths),
        active_evaluator_name=run_state.active_evaluator_name,
        tree_size_at_last_save=run_state.tree_size_at_last_save,
        last_save_unix_s=run_state.last_save_unix_s,
        latest_runtime_checkpoint_path=run_state.latest_runtime_checkpoint_path,
        latest_record_status=run_state.latest_record_status,
        latest_frontier_status=run_state.latest_frontier_status,
        metadata=next_metadata,
    )


def validate_forced_evaluator(
    *,
    force_evaluator: str | None,
    evaluator_names: Mapping[str, MorpionEvaluatorSpec],
) -> None:
    """Validate one optional forced evaluator against the configured set."""
    if force_evaluator is None:
        return
    if force_evaluator not in evaluator_names:
        raise UnknownForcedMorpionEvaluatorError(force_evaluator)


def select_active_evaluator_name(
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
) -> str:
    """Select the active evaluator using the lowest available final loss."""
    selectable_losses = {
        evaluator_name: metrics.final_loss
        for evaluator_name, metrics in evaluator_metrics.items()
        if metrics.final_loss is not None and math.isfinite(metrics.final_loss)
    }
    if not selectable_losses:
        raise NoSelectableMorpionEvaluatorError
    return min(
        selectable_losses,
        key=lambda evaluator_name: selectable_losses[evaluator_name],
    )


def select_or_force_active_evaluator_name(
    *,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
    force_evaluator: str | None,
) -> str:
    """Return the forced evaluator when present, else the default auto-selection."""
    if force_evaluator is not None:
        if force_evaluator not in evaluator_metrics:
            raise UnknownForcedMorpionEvaluatorError(force_evaluator)
        return force_evaluator
    return select_active_evaluator_name(evaluator_metrics)


def next_metadata(
    current_metadata: Mapping[str, object],
    *,
    relative_runtime_checkpoint_path: str | None,
    control: MorpionBootstrapControl,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    training_skipped_reason: str | None = None,
) -> dict[str, object]:
    """Return updated run metadata after one cycle boundary."""
    next_payload = dict(current_metadata)
    if relative_runtime_checkpoint_path is None:
        next_payload.pop(RUNTIME_CHECKPOINT_METADATA_KEY, None)
    else:
        next_payload[RUNTIME_CHECKPOINT_METADATA_KEY] = relative_runtime_checkpoint_path
    next_payload[BOOTSTRAP_APPLIED_CONTROL_METADATA_KEY] = bootstrap_control_to_dict(
        control
    )
    next_payload[BOOTSTRAP_APPLIED_RUNTIME_CONTROL_METADATA_KEY] = (
        bootstrap_runtime_control_to_dict(control.runtime)
    )
    next_payload[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] = (
        effective_runtime_config_to_dict(effective_runtime_config)
    )
    next_payload[BOOTSTRAP_EFFECTIVE_RUNTIME_HASH_METADATA_KEY] = (
        effective_runtime_config_sha256(effective_runtime_config)
    )
    if training_skipped_reason is None:
        next_payload.pop(TRAINING_SKIPPED_REASON_METADATA_KEY, None)
    else:
        next_payload[TRAINING_SKIPPED_REASON_METADATA_KEY] = training_skipped_reason
    return next_payload


def previous_effective_runtime_config(
    metadata: Mapping[str, object],
    *,
    resolved_bootstrap_config: MorpionBootstrapConfig,
) -> MorpionBootstrapEffectiveRuntimeConfig | None:
    """Return the last applied runtime config, falling back for legacy metadata."""
    persisted_runtime = effective_runtime_config_from_metadata(
        metadata.get(BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY)
    )
    if persisted_runtime is not None:
        return persisted_runtime
    runtime_checkpoint_path = metadata.get(RUNTIME_CHECKPOINT_METADATA_KEY)
    if runtime_checkpoint_path is None:
        return None
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=resolved_bootstrap_config.runtime.tree_branch_limit,
    )


def validate_runtime_reconfiguration(
    *,
    previous_effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Validate that the requested runtime change stays within the supported subset."""
    if previous_effective_runtime_config is None:
        return
    if (
        effective_runtime_config.tree_branch_limit
        > previous_effective_runtime_config.tree_branch_limit
    ):
        raise UnsupportedMorpionRuntimeReconfigurationError(
            previous_tree_branch_limit=previous_effective_runtime_config.tree_branch_limit,
            requested_tree_branch_limit=effective_runtime_config.tree_branch_limit,
        )


__all__ = [
    "EMPTY_DATASET_TRAINING_SKIPPED_REASON",
    "RUNTIME_CHECKPOINT_METADATA_KEY",
    "TRAINING_SKIPPED_REASON_METADATA_KEY",
    "BootstrapDatasetBuildResult",
    "BootstrapTrainingResult",
    "ResolvedActiveMorpionModelBundle",
    "build_and_save_dataset_for_generation",
    "build_bootstrap_event",
    "build_event_metadata",
    "build_no_save_run_state",
    "bootstrap_config_hash_from_metadata",
    "extract_rows_from_training_snapshot",
    "next_metadata",
    "pipeline_metadata",
    "previous_effective_runtime_config",
    "prune_saved_generation_artifacts",
    "record_no_save_cycle_event",
    "reevaluate_tree_for_policy",
    "require_single_process_mode",
    "resolve_active_model_bundle",
    "resolve_runtime_restore_path",
    "resolve_tree_status",
    "save_trigger_reason",
    "select_active_evaluator_name",
    "select_or_force_active_evaluator_name",
    "should_save_progress",
    "timestamp_utc_from_unix_s",
    "train_and_select_evaluators",
    "validate_dataset_family_target_args",
    "validate_forced_evaluator",
    "validate_pipeline_mode",
    "validate_runtime_reconfiguration",
    "with_config_hash_metadata",
    "write_pipeline_active_model",
    "write_pipeline_manifest_for_generation",
    "write_pipeline_status_files",
]