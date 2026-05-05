"""Runtime and retention helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .bootstrap_errors import (
    IncompatibleMorpionResumeArtifactError,
    MissingActiveMorpionEvaluatorError,
    MissingForcedMorpionEvaluatorBundleError,
    UnknownActiveMorpionEvaluatorError,
)
from .bootstrap_paths import (
    DEFAULT_KEEP_LATEST_RUNTIME_CHECKPOINTS,
    DEFAULT_KEEP_LATEST_TREE_EXPORTS,
    MorpionBootstrapPaths,
    prune_generation_files,
)
from .cycle_metadata import RUNTIME_CHECKPOINT_METADATA_KEY, next_metadata
from .history import MorpionBootstrapTreeStatus
from .run_state import MorpionBootstrapRunState

if TYPE_CHECKING:
    from pathlib import Path

    from .control import (
        MorpionBootstrapControl,
        MorpionBootstrapEffectiveRuntimeConfig,
    )
    from .search_runner_protocol import MorpionSearchRunner

LOGGER = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class ResolvedActiveMorpionModelBundle:
    """Resolved active evaluator identity and bundle path for one cycle."""

    active_evaluator_name: str | None
    model_bundle_path: Path | None


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
                num_expanded_nodes=_optional_tree_int(
                    status_mapping.get("num_expanded_nodes"),
                    field_name="num_expanded_nodes",
                ),
                num_simulations=_optional_tree_int(
                    status_mapping.get("num_simulations"),
                    field_name="num_simulations",
                ),
                root_visit_count=_optional_tree_int(
                    status_mapping.get("root_visit_count"),
                    field_name="root_visit_count",
                ),
                min_depth_present=_optional_tree_int(
                    status_mapping.get("min_depth_present"),
                    field_name="min_depth_present",
                ),
                max_depth_present=_optional_tree_int(
                    status_mapping.get("max_depth_present"),
                    field_name="max_depth_present",
                ),
                depth_node_counts=_optional_tree_int_mapping(
                    status_mapping.get("depth_node_counts"),
                    field_name="depth_node_counts",
                ),
            )
        raise TypeError(_CURRENT_TREE_STATUS_TYPE_ERROR)
    return MorpionBootstrapTreeStatus(num_nodes=current_tree_size)


def _optional_tree_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise _tree_status_int_error(field_name)
    if isinstance(value, int):
        return value
    raise _tree_status_int_error(field_name)


def _optional_tree_int_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[int, int]:
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
        cache_morpion_search_checkpoint_payload_for_restore,
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
            payload = load_morpion_search_checkpoint_payload(candidate_path)
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
        cache_morpion_search_checkpoint_payload_for_restore(candidate_path, payload)
        return candidate_path

    if first_incompatible_error is not None:
        raise first_incompatible_error
    return None


__all__ = [
    "ResolvedActiveMorpionModelBundle",
    "build_no_save_run_state",
    "prune_saved_generation_artifacts",
    "resolve_active_model_bundle",
    "resolve_runtime_restore_path",
    "resolve_tree_status",
]
