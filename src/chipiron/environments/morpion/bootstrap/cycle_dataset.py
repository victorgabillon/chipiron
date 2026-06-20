"""Dataset extraction helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anemone.training_export import load_training_tree_snapshot

from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_rows_from_training_snapshot,
    morpion_supervised_rows_metadata_from_training_snapshot,
    save_morpion_supervised_rows,
    training_tree_snapshot_to_morpion_supervised_rows,
)

from .bootstrap_errors import (
    MissingBootstrapFrontierStatusError,
    MissingBootstrapRecordStatusError,
    MissingSavedBootstrapArtifactError,
)
from .dataset_family_targets import (
    apply_dataset_family_target_policy,
    build_dataset_family_target_context,
)
from .record_status import (
    MorpionBootstrapFrontierStatus,
    MorpionBootstrapRecordStatus,
    resolve_frontier_status_for_cycle_with_metadata,
    resolve_record_status_for_cycle,
)
from .sharded_training_export import load_morpion_sharded_training_tree_snapshot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anemone.training_export import TrainingTreeSnapshot

    from chipiron.environments.morpion.learning import (
        MorpionSupervisedRow,
        MorpionSupervisedRows,
    )

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .memory_diagnostics import MemoryDiagnostics
    from .run_state import MorpionBootstrapRunState
    from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)


def _invalid_sharded_training_export_path_error() -> TypeError:
    """Return the stable invalid sharded-export path error."""
    return TypeError("sharded training export must return a filesystem path")


def _unsupported_sharded_training_export_error() -> TypeError:
    """Return the canonical missing sharded-export capability error."""
    return TypeError(
        "runner must implement export_sharded_training_tree_snapshot() when "
        "training_export_mode is 'sharded' or 'both'."
    )


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
class MorpionSupervisedRowsStream:
    """Streaming rows plus metadata for one dataset export."""

    rows: Iterable[MorpionSupervisedRow]
    metadata: dict[str, object]


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
    rows = apply_dataset_family_target_policy(
        snapshot=snapshot,
        rows=rows,
        family_target_policy=args.dataset_family_target_policy,
        family_prediction_blend=args.dataset_family_prediction_blend,
        use_backed_up_value=args.use_backed_up_value,
    )
    target_source_counts = rows.metadata.get("target_source_counts", {})
    LOGGER.info(
        "[dataset-targets] generation=%s rows=%s backed_up=%s exact_or_terminal_direct=%s direct_frontier_fallback=%s skipped_no_target=%s",
        generation,
        len(rows.rows),
        target_source_counts.get("backed_up_value", 0),
        target_source_counts.get("exact_or_terminal_direct_value", 0),
        target_source_counts.get("direct_value_frontier_fallback", 0),
        rows.metadata.get("skipped_no_target_count", 0),
    )
    return rows


def iter_rows_from_training_snapshot(
    *,
    args: MorpionBootstrapArgs,
    snapshot: TrainingTreeSnapshot,
    generation: int,
) -> Iterable[MorpionSupervisedRow]:
    """Yield post-processed supervised rows from one training snapshot."""
    del generation
    context = build_dataset_family_target_context(
        snapshot=snapshot,
        family_target_policy=args.dataset_family_target_policy,
        family_prediction_blend=args.dataset_family_prediction_blend,
        use_backed_up_value=args.use_backed_up_value,
    )
    for row in iter_morpion_supervised_rows_from_training_snapshot(
        snapshot,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        max_rows=args.max_rows,
        use_backed_up_value=args.use_backed_up_value,
    ):
        yield context.adjust_row(row)


def streaming_rows_from_training_snapshot(
    *,
    args: MorpionBootstrapArgs,
    snapshot: TrainingTreeSnapshot,
    generation: int,
) -> MorpionSupervisedRowsStream:
    """Return fresh streaming rows plus extraction metadata for a dataset export."""
    context = build_dataset_family_target_context(
        snapshot=snapshot,
        family_target_policy=args.dataset_family_target_policy,
        family_prediction_blend=args.dataset_family_prediction_blend,
        use_backed_up_value=args.use_backed_up_value,
    )

    def _adjusted_rows() -> Iterable[MorpionSupervisedRow]:
        for row in iter_morpion_supervised_rows_from_training_snapshot(
            snapshot,
            require_exact_or_terminal=args.require_exact_or_terminal,
            min_depth=args.min_depth,
            min_visit_count=args.min_visit_count,
            max_rows=args.max_rows,
            use_backed_up_value=args.use_backed_up_value,
        ):
            yield context.adjust_row(row)

    metadata = morpion_supervised_rows_metadata_from_training_snapshot(
        snapshot,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        max_rows=args.max_rows,
        use_backed_up_value=args.use_backed_up_value,
        metadata={"bootstrap_generation": generation},
    )
    # TODO: compute family-target summary during the write pass once row metadata
    # can be finalized after streaming, e.g. via a sidecar rows manifest.
    metadata.update(context.summary_for_rows(_adjusted_rows()))
    target_source_counts = metadata.get("target_source_counts", {})
    if not isinstance(target_source_counts, dict):
        target_source_counts = {}
    LOGGER.info(
        "[dataset-targets] generation=%s rows=%s backed_up=%s exact_or_terminal_direct=%s direct_frontier_fallback=%s skipped_no_target=%s",
        generation,
        metadata.get("num_rows", 0),
        target_source_counts.get("backed_up_value", 0),
        target_source_counts.get("exact_or_terminal_direct_value", 0),
        target_source_counts.get("direct_value_frontier_fallback", 0),
        metadata.get("skipped_no_target_count", 0),
    )
    return MorpionSupervisedRowsStream(rows=_adjusted_rows(), metadata=metadata)


def export_training_snapshot_for_generation(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    runner: MorpionSearchRunner,
    generation: int,
) -> Path:
    """Export the configured training artifact and return the primary path."""
    flat_path = paths.tree_snapshot_path_for_generation(generation)
    if args.training_export_mode == "flat":
        runner.export_training_tree_snapshot(flat_path)
        LOGGER.info(
            "[save] training_export_selected mode=%s output=%s",
            args.training_export_mode,
            str(flat_path),
        )
        return flat_path

    export_sharded = getattr(runner, "export_sharded_training_tree_snapshot", None)
    if not callable(export_sharded):
        raise _unsupported_sharded_training_export_error()

    exported_path = export_sharded(
        paths.sharded_tree_snapshot_dir,
        generation=generation,
    )
    if isinstance(exported_path, Path):
        sharded_path = exported_path
    elif isinstance(exported_path, str):
        sharded_path = Path(exported_path)
    else:
        raise _invalid_sharded_training_export_path_error()
    if args.training_export_mode == "sharded":
        LOGGER.info(
            "[save] training_export_selected mode=%s output=%s",
            args.training_export_mode,
            str(sharded_path),
        )
        return sharded_path

    runner.export_training_tree_snapshot(flat_path)
    LOGGER.info(
        "[save] training_export_selected mode=%s output=%s compatibility_output=%s",
        args.training_export_mode,
        str(flat_path),
        str(sharded_path),
    )
    return flat_path


def load_training_snapshot_for_generation(
    *,
    args: MorpionBootstrapArgs,
    artifact_path: str | Path,
) -> TrainingTreeSnapshot:
    """Load the training snapshot represented by the configured artifact path."""
    if args.training_export_mode == "sharded":
        return load_morpion_sharded_training_tree_snapshot(artifact_path)
    return load_training_tree_snapshot(artifact_path)


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
    training_snapshot_path = (
        paths.tree_snapshot_path_for_generation(generation)
        if args.training_export_mode != "sharded"
        else paths.sharded_tree_snapshot_path_for_generation(generation)
    )
    rows_path = paths.rows_path_for_generation(generation)
    dataset_started_at = time.perf_counter()
    LOGGER.info("[dataset] build_start snapshot=%s", str(training_snapshot_path))
    memory.log("before_snapshot_load_or_export")
    training_snapshot_path = export_training_snapshot_for_generation(
        args=args,
        paths=paths,
        runner=runner,
        generation=generation,
    )
    if not training_snapshot_path.is_file():
        raise MissingSavedBootstrapArtifactError(
            action="export_training_snapshot_for_generation()",
            artifact_path=training_snapshot_path,
        )
    snapshot = load_training_snapshot_for_generation(
        args=args,
        artifact_path=training_snapshot_path,
    )
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
        record_error = MissingBootstrapRecordStatusError()
        raise record_error

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
            None
            if frontier_status is None
            else frontier_status.current_best_total_points,
        )
    if frontier_status is None:
        frontier_error = MissingBootstrapFrontierStatusError()
        raise frontier_error

    LOGGER.info("[dataset] extract_start snapshot_nodes=%s", len(snapshot.nodes))
    memory.log("before_dataset_extract")
    extract_started_at = time.perf_counter()
    rows: MorpionSupervisedRows
    try:
        rows = extract_rows_from_training_snapshot(
            args=args,
            snapshot=snapshot,
            generation=generation,
        )
        memory.log("after_dataset_extract")
        memory.log("after_dataset_family_target_policy")
    except Exception:
        LOGGER.info(
            "[dataset] extract_done rows=%s elapsed=%.3fs",
            None,
            time.perf_counter() - extract_started_at,
        )
        raise
    LOGGER.info(
        "[dataset] extract_done rows=%s elapsed=%.3fs",
        len(rows.rows),
        time.perf_counter() - extract_started_at,
    )

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


__all__ = [
    "BootstrapDatasetBuildResult",
    "MorpionSupervisedRowsStream",
    "build_and_save_dataset_for_generation",
    "export_training_snapshot_for_generation",
    "extract_rows_from_training_snapshot",
    "iter_rows_from_training_snapshot",
    "load_training_snapshot_for_generation",
    "streaming_rows_from_training_snapshot",
]
