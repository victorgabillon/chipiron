"""Producer-only reevaluation worker for bounded Morpion patch artifacts."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from anemone.training_export import TrainingTreeSnapshot, load_training_tree_snapshot

from .bootstrap_paths import MorpionBootstrapPaths
from .cycle_timing import timestamp_utc_from_unix_s
from .pipeline_artifacts import (
    MissingMorpionPipelineArtifactError,
    MorpionPipelineActiveModel,
    MorpionReevaluationCursor,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    load_pipeline_active_model,
    load_reevaluation_cursor,
    save_reevaluation_cursor,
    save_reevaluation_patch,
)
from .pipeline_orchestrator import load_available_pipeline_manifests

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs


def _negative_max_nodes_per_patch_error() -> ValueError:
    """Build the stable invalid max-nodes-per-patch error."""
    return ValueError("max_nodes_per_patch must be >= 0")


@dataclass(frozen=True, slots=True)
class MorpionReevaluationWorkerResult:
    """Summary of one reevaluation-worker pass."""

    patch_written: bool
    reason: str | None
    patch_id: str | None
    num_rows: int
    evaluator_generation: int | None
    evaluator_name: str | None
    start_cursor: str | None
    end_cursor: str | None
    completed_full_pass_count: int | None


class MorpionNodeReevaluationEvaluator(Protocol):
    """Adapter protocol for producing reevaluation patch rows from a snapshot."""

    def evaluate_patch_rows(
        self,
        snapshot: TrainingTreeSnapshot,
        node_ids: Sequence[str],
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Return reevaluation rows for the selected snapshot node ids."""


def cursor_matches_active_model(
    cursor: MorpionReevaluationCursor,
    *,
    active_model: MorpionPipelineActiveModel,
) -> bool:
    """Return whether one persisted cursor belongs to the active evaluator."""
    return (
        cursor.evaluator_generation == active_model.generation
        and cursor.evaluator_name == active_model.evaluator_name
        and cursor.model_bundle_path == active_model.model_bundle_path
    )


def resolve_latest_reevaluation_tree_snapshot(
    paths: MorpionBootstrapPaths,
) -> tuple[int, Path] | None:
    """Return the newest manifest generation that still has a usable tree snapshot."""
    manifests = load_available_pipeline_manifests(paths)
    for generation in sorted(manifests, reverse=True):
        manifest = manifests[generation]
        if manifest.tree_snapshot_path is None:
            continue
        snapshot_path = paths.resolve_work_dir_path(manifest.tree_snapshot_path)
        if snapshot_path is None or not snapshot_path.is_file():
            continue
        return generation, snapshot_path
    return None


def select_reevaluation_node_window(
    node_ids: Sequence[str],
    *,
    start_cursor: str | None,
    max_nodes: int,
) -> tuple[tuple[str, ...], str | None, bool]:
    """Select one deterministic bounded reevaluation window from sorted node ids."""
    ordered_node_ids = tuple(node_ids)
    if not ordered_node_ids or max_nodes <= 0:
        return (), None, False

    try:
        start_index = 0 if start_cursor is None else ordered_node_ids.index(start_cursor)
    except ValueError:
        start_index = 0

    if max_nodes >= len(ordered_node_ids):
        selected = (
            ordered_node_ids[start_index:] + ordered_node_ids[:start_index]
        )
        return selected, ordered_node_ids[0], True

    selected_node_ids: list[str] = []
    next_index = start_index
    completed_full_pass = False
    while len(selected_node_ids) < max_nodes:
        selected_node_ids.append(ordered_node_ids[next_index])
        next_index += 1
        if next_index == len(ordered_node_ids):
            next_index = 0
            completed_full_pass = True

    return (
        tuple(selected_node_ids),
        ordered_node_ids[next_index],
        completed_full_pass,
    )


def snapshot_values_to_patch_rows(
    snapshot: TrainingTreeSnapshot,
    node_ids: Sequence[str],
) -> tuple[MorpionReevaluationPatchRow, ...]:
    """Build reevaluation rows from the values already stored in one snapshot."""
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}
    rows: list[MorpionReevaluationPatchRow] = []
    for node_id in node_ids:
        node = nodes_by_id[node_id]
        backed_up_value = (
            None
            if node.backed_up_value_scalar is None
            else float(node.backed_up_value_scalar)
        )
        if node.direct_value_scalar is not None:
            direct_value = float(node.direct_value_scalar)
        elif backed_up_value is not None:
            direct_value = backed_up_value
        else:
            direct_value = 0.0
        rows.append(
            MorpionReevaluationPatchRow(
                node_id=node_id,
                direct_value=direct_value,
                backed_up_value=backed_up_value,
                is_exact=node.is_exact,
                is_terminal=node.is_terminal,
                metadata={"source": "snapshot_existing_values"},
            )
        )
    return tuple(rows)


def run_morpion_reevaluation_worker_once(
    args: MorpionBootstrapArgs,
    *,
    evaluator: MorpionNodeReevaluationEvaluator | None = None,
    max_nodes_per_patch: int = 10_000,
    now_unix_s: float | None = None,
    patch_id: str | None = None,
) -> MorpionReevaluationWorkerResult:
    """Produce at most one reevaluation patch and advance the reevaluation cursor."""
    if max_nodes_per_patch < 0:
        raise _negative_max_nodes_per_patch_error()

    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()

    if max_nodes_per_patch == 0:
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="max_nodes_per_patch_zero",
            patch_id=None,
            num_rows=0,
            evaluator_generation=None,
            evaluator_name=None,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )

    try:
        active_model = load_pipeline_active_model(paths.pipeline_active_model_path)
    except MissingMorpionPipelineArtifactError:
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="missing_active_model",
            patch_id=None,
            num_rows=0,
            evaluator_generation=None,
            evaluator_name=None,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )

    if paths.pipeline_reevaluation_patch_path.exists():
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="pending_patch_exists",
            patch_id=None,
            num_rows=0,
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )

    latest_snapshot = resolve_latest_reevaluation_tree_snapshot(paths)
    if latest_snapshot is None:
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="missing_tree_snapshot",
            patch_id=None,
            num_rows=0,
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )
    tree_generation, snapshot_path = latest_snapshot

    snapshot = load_training_tree_snapshot(snapshot_path)
    sorted_node_ids = tuple(sorted(node.node_id for node in snapshot.nodes))
    if not sorted_node_ids:
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="empty_tree_snapshot",
            patch_id=None,
            num_rows=0,
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )

    try:
        persisted_cursor = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)
    except MissingMorpionPipelineArtifactError:
        persisted_cursor = None

    if persisted_cursor is None or not cursor_matches_active_model(
        persisted_cursor,
        active_model=active_model,
    ):
        start_cursor = None
        completed_full_pass_count = 0
    else:
        start_cursor = persisted_cursor.next_node_cursor
        completed_full_pass_count = persisted_cursor.completed_full_pass_count

    selected_node_ids, next_node_cursor, completed_full_pass = (
        select_reevaluation_node_window(
            sorted_node_ids,
            start_cursor=start_cursor,
            max_nodes=max_nodes_per_patch,
        )
    )
    selected_start_cursor = selected_node_ids[0] if selected_node_ids else None
    selected_end_cursor = selected_node_ids[-1] if selected_node_ids else None

    if evaluator is None:
        patch_rows = snapshot_values_to_patch_rows(snapshot, selected_node_ids)
    else:
        patch_rows = tuple(evaluator.evaluate_patch_rows(snapshot, selected_node_ids))

    if paths.pipeline_reevaluation_patch_path.exists():
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="pending_patch_exists",
            patch_id=None,
            num_rows=0,
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
        )

    resolved_now_unix_s = time.time() if now_unix_s is None else now_unix_s
    timestamp_utc = timestamp_utc_from_unix_s(resolved_now_unix_s)
    resolved_patch_id = str(uuid.uuid4()) if patch_id is None else patch_id
    patch = MorpionReevaluationPatch(
        patch_id=resolved_patch_id,
        created_at_utc=timestamp_utc,
        evaluator_generation=active_model.generation,
        evaluator_name=active_model.evaluator_name,
        model_bundle_path=active_model.model_bundle_path,
        rows=patch_rows,
        tree_generation=tree_generation,
        start_cursor=selected_start_cursor,
        end_cursor=selected_end_cursor,
        metadata={
            "completed_full_pass": completed_full_pass,
            "max_nodes_per_patch": max_nodes_per_patch,
            "next_node_cursor": next_node_cursor,
            "source": "reevaluation_worker",
        },
    )
    save_reevaluation_patch(patch, paths.pipeline_reevaluation_patch_path)

    next_completed_full_pass_count = completed_full_pass_count + int(
        completed_full_pass
    )
    save_reevaluation_cursor(
        MorpionReevaluationCursor(
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            model_bundle_path=active_model.model_bundle_path,
            next_node_cursor=next_node_cursor,
            updated_at_utc=timestamp_utc,
            tree_generation=tree_generation,
            completed_full_pass_count=next_completed_full_pass_count,
            last_patch_id=patch.patch_id,
            metadata={"source": "reevaluation_worker"},
        ),
        paths.pipeline_reevaluation_cursor_path,
    )

    return MorpionReevaluationWorkerResult(
        patch_written=True,
        reason=None,
        patch_id=patch.patch_id,
        num_rows=len(patch.rows),
        evaluator_generation=patch.evaluator_generation,
        evaluator_name=patch.evaluator_name,
        start_cursor=patch.start_cursor,
        end_cursor=patch.end_cursor,
        completed_full_pass_count=next_completed_full_pass_count,
    )


__all__ = [
    "MorpionNodeReevaluationEvaluator",
    "MorpionReevaluationWorkerResult",
    "cursor_matches_active_model",
    "resolve_latest_reevaluation_tree_snapshot",
    "run_morpion_reevaluation_worker_once",
    "select_reevaluation_node_window",
    "snapshot_values_to_patch_rows",
]
