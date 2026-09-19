"""Producer-only reevaluation worker for bounded Morpion patch artifacts."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from anemone.training_export import TrainingTreeSnapshot, load_training_tree_snapshot
from anemone.training_export.serialization import MalformedNodesFieldError
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

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
    reevaluation_patch_to_dict,
    save_reevaluation_cursor,
)
from .pipeline_memory import log_available_ram_guard, log_pipeline_memory
from .pipeline_orchestrator import load_available_pipeline_manifests
from .runtime.runner import load_morpion_evaluator_from_model_bundle
from .sharded_training_export import load_morpion_sharded_training_tree_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from .bootstrap_args import MorpionBootstrapArgs


LOGGER = logging.getLogger(__name__)
REEVALUATION_IDLE_HEARTBEAT_SECONDS = 60.0
_REEVALUATION_CYCLE_SEPARATOR = "=" * 70


def _negative_max_nodes_per_patch_error() -> ValueError:
    """Build the stable invalid max-nodes-per-patch error."""
    return ValueError("max_nodes_per_patch must be >= 0")


def _reevaluation_bundle_missing_error(
    path: object,
) -> MissingMorpionPipelineArtifactError:
    """Build the stable missing-active-model-bundle error."""
    return MissingMorpionPipelineArtifactError(
        f"Morpion reevaluation active-model bundle does not exist: {path}"
    )


def _non_finite_direct_value_error(node_id: str) -> ValueError:
    """Build the stable non-finite evaluator-value error."""
    return ValueError(
        f"Morpion reevaluation evaluator returned a non-finite value for node {node_id!r}"
    )


def _finite_direct_value(raw_value: object, *, node_id: str) -> float:
    """Coerce one evaluator output to a finite patch-row scalar."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise _non_finite_direct_value_error(node_id)
    try:
        direct_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise _non_finite_direct_value_error(node_id) from exc
    if not math.isfinite(direct_value):
        raise _non_finite_direct_value_error(node_id)
    return direct_value


def _extract_score(raw_evaluation: object, *, node_id: str) -> float:
    """Extract one scalar score from the evaluator's return value."""
    if hasattr(raw_evaluation, "score"):
        return _finite_direct_value(
            cast("Any", raw_evaluation).score,
            node_id=node_id,
        )
    return _finite_direct_value(raw_evaluation, node_id=node_id)


def _required_terminal_value(value: object | None, *, node_id: str) -> float:
    """Return one already-persisted terminal value for reevaluation."""
    if value is None:
        raise _non_finite_direct_value_error(node_id)
    return _finite_direct_value(value, node_id=node_id)


def _save_reevaluation_patch_exclusive(
    patch: MorpionReevaluationPatch,
    path: Path,
) -> bool:
    """Create one pending patch only if no patch artifact exists yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                reevaluation_patch_to_dict(patch), handle, indent=2, sort_keys=True
            )
            handle.write("\n")
    except FileExistsError:
        return False
    return True


def _metric_value(value: object | None) -> str:
    """Render one optional reevaluation log field as a stable string."""
    if value is None:
        return "none"
    return str(value)


def _log_field_value(value: object) -> str:
    """Render one structured human-log field value."""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _format_log_fields(**fields: object | None) -> str:
    """Return compact key=value fields, omitting absent values."""
    return " ".join(
        f"{key}={_log_field_value(value)}"
        for key, value in fields.items()
        if value is not None
    )


def _reevaluation_worker_state_path(paths: MorpionBootstrapPaths) -> Path:
    """Return the cross-process idle-log suppression state path."""
    return paths.work_dir / "logs" / "reevaluation_worker_state.json"


def _read_reevaluation_worker_state(path: Path) -> dict[str, object]:
    """Load idle-log suppression state, treating corrupt state as absent."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_reevaluation_worker_state(
    path: Path,
    state: dict[str, object],
) -> None:
    """Persist idle-log suppression state for the next one-shot worker pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def _relative_to_work_dir(
    paths: MorpionBootstrapPaths, path: Path | None
) -> str | None:
    """Render one path relative to the work dir when possible."""
    if path is None:
        return None
    return paths.relative_to_work_dir(path)


def _idle_signature(
    *,
    status: str,
    reason: str,
    active_model: MorpionPipelineActiveModel | None,
    latest_generation: int | None,
    pending_patch_path: str | None,
) -> dict[str, object]:
    """Build the state-change identity for idle reevaluation summaries."""
    return {
        "status": status,
        "reason": reason,
        "active_model_generation": (
            None if active_model is None else active_model.generation
        ),
        "active_model_evaluator": (
            None if active_model is None else active_model.evaluator_name
        ),
        "latest_generation": latest_generation,
        "pending_patch_path": pending_patch_path,
    }


def _maybe_log_reevaluation_idle(
    *,
    paths: MorpionBootstrapPaths,
    status: str,
    reason: str,
    active_model: MorpionPipelineActiveModel | None,
    latest_generation: int | None,
    pending_patch_path: str | None = None,
    action: str | None = None,
    next_check_s: int | None = None,
    now_unix_s: float | None = None,
) -> None:
    """Log idle/blocking status only on state changes or periodic heartbeats."""
    resolved_now_unix_s = time.time() if now_unix_s is None else now_unix_s
    state_path = _reevaluation_worker_state_path(paths)
    state = _read_reevaluation_worker_state(state_path)
    signature = _idle_signature(
        status=status,
        reason=reason,
        active_model=active_model,
        latest_generation=latest_generation,
        pending_patch_path=pending_patch_path,
    )
    previous_signature = state.get("last_idle_signature")
    signature_changed = previous_signature != signature
    previous_checks = state.get("checks")
    checks = int(previous_checks) + 1 if isinstance(previous_checks, int) else 1
    idle_since_unix_s = state.get("idle_since_unix_s")
    if signature_changed or not isinstance(idle_since_unix_s, int | float):
        idle_since_unix_s = resolved_now_unix_s
        checks = 1
    last_logged_unix_s_object = state.get("last_logged_unix_s")
    last_logged_unix_s = (
        float(last_logged_unix_s_object)
        if isinstance(last_logged_unix_s_object, int | float)
        else None
    )
    should_log = signature_changed or last_logged_unix_s is None
    if not should_log and last_logged_unix_s is not None:
        should_log = (
            resolved_now_unix_s - last_logged_unix_s
            >= REEVALUATION_IDLE_HEARTBEAT_SECONDS
        )

    if should_log and signature_changed:
        LOGGER.info(
            "[reevaluation-cycle] %s",
            _format_log_fields(
                status=status,
                reason=reason,
                active_model=(
                    None if active_model is None else active_model.evaluator_name
                ),
                source_generation=(
                    None if active_model is None else active_model.generation
                ),
                latest_generation=latest_generation,
                pending_patch=pending_patch_path,
                action=action,
                next_check_s=next_check_s,
            ),
        )
    elif should_log:
        LOGGER.info(
            "[reevaluation-watch] %s",
            _format_log_fields(
                status="IDLE",
                reason=reason,
                active_model=(
                    None if active_model is None else active_model.evaluator_name
                ),
                source_generation=(
                    None if active_model is None else active_model.generation
                ),
                latest_generation=latest_generation,
                pending_patch=pending_patch_path,
                idle_for=f"{int(resolved_now_unix_s - float(idle_since_unix_s))}s",
                checks=checks,
            ),
        )

    state["last_idle_signature"] = signature
    state["idle_since_unix_s"] = idle_since_unix_s
    state["checks"] = checks
    state["updated_unix_s"] = resolved_now_unix_s
    if should_log:
        state["last_logged_unix_s"] = resolved_now_unix_s
    _write_reevaluation_worker_state(state_path, state)


def _record_reevaluation_action(
    paths: MorpionBootstrapPaths,
    *,
    status: str,
    now_unix_s: float | None = None,
) -> None:
    """Clear idle suppression after real reevaluation action."""
    resolved_now_unix_s = time.time() if now_unix_s is None else now_unix_s
    _write_reevaluation_worker_state(
        _reevaluation_worker_state_path(paths),
        {
            "last_action_status": status,
            "last_idle_signature": None,
            "updated_unix_s": resolved_now_unix_s,
        },
    )


def _log_reevaluation_start_block(
    *,
    active_model: MorpionPipelineActiveModel,
    tree_generation: int,
    snapshot_path: str,
    max_nodes_per_patch: int,
    selected_rows: int,
    total_rows: int,
) -> None:
    """Emit the high-visibility operator log block for real reevaluation work."""
    LOGGER.info(_REEVALUATION_CYCLE_SEPARATOR)
    LOGGER.info(
        "[reevaluation-cycle] %s",
        _format_log_fields(
            generation=active_model.generation,
            evaluator=active_model.evaluator_name,
            status="STARTED",
            latest_tree_generation=tree_generation,
            rows=selected_rows,
            total_rows=total_rows,
            max_nodes=max_nodes_per_patch,
            target="tree_snapshot_nodes",
        ),
    )
    LOGGER.info("[reevaluation-cycle] checkpoint=%s", snapshot_path)
    LOGGER.info(_REEVALUATION_CYCLE_SEPARATOR)


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
    tree_generation: int | None = None
    pending_patch_path: str | None = None
    model_bundle_path: str | None = None


class MorpionNodeReevaluationEvaluator(Protocol):
    """Adapter protocol for producing reevaluation patch rows from a snapshot."""

    def evaluate_patch_rows(
        self,
        snapshot: TrainingTreeSnapshot,
        node_ids: Sequence[str],
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Return reevaluation rows for the selected snapshot node ids."""
        ...


@dataclass(slots=True)
class MorpionActiveModelNodeReevaluationEvaluator:
    """Reevaluation adapter backed by the currently selected active model bundle."""

    model_bundle_path: Path
    _dynamics: MorpionDynamics = field(
        default_factory=MorpionDynamics,
        init=False,
        repr=False,
    )
    _state_codec: MorpionStateCheckpointCodec = field(
        default_factory=MorpionStateCheckpointCodec,
        init=False,
        repr=False,
    )
    _evaluator: object | None = field(default=None, init=False, repr=False)

    def _load_snapshot_state(self, payload: object) -> MorpionState:
        """Decode one training-snapshot state payload for model evaluation."""
        atomheart_state = self._state_codec.load_state_ref(payload)
        return self._dynamics.wrap_atomheart_state(atomheart_state)

    def _resolved_evaluator(self) -> object:
        """Load and memoize the active-model evaluator bundle."""
        if self._evaluator is None:
            self._evaluator = load_morpion_evaluator_from_model_bundle(
                self.model_bundle_path
            )
        return self._evaluator

    def evaluate_patch_rows(
        self,
        snapshot: TrainingTreeSnapshot,
        node_ids: Sequence[str],
    ) -> tuple[MorpionReevaluationPatchRow, ...]:
        """Evaluate one bounded snapshot node set with the active model bundle."""
        nodes_by_id = {node.node_id: node for node in snapshot.nodes}
        evaluator = self._resolved_evaluator()
        rows: list[MorpionReevaluationPatchRow] = []
        for node_id in node_ids:
            node = nodes_by_id[node_id]
            if node.is_terminal and node.backed_up_value_scalar is not None:
                direct_value = _required_terminal_value(
                    node.backed_up_value_scalar,
                    node_id=node_id,
                )
                source = "terminal_existing_value"
            else:
                state = self._load_snapshot_state(node.state_ref_payload)
                raw_evaluation = cast("Any", evaluator).evaluate(state)
                direct_value = _extract_score(raw_evaluation, node_id=node_id)
                source = "active_model_reevaluation"
            rows.append(
                MorpionReevaluationPatchRow(
                    node_id=node_id,
                    direct_value=direct_value,
                    backed_up_value=None,
                    is_exact=node.is_exact,
                    is_terminal=node.is_terminal,
                    metadata={
                        "model_bundle_path": str(self.model_bundle_path),
                        "source": source,
                    },
                )
            )
        return tuple(rows)


def build_active_model_reevaluation_evaluator(
    *,
    paths: MorpionBootstrapPaths,
    active_model: MorpionPipelineActiveModel,
) -> MorpionActiveModelNodeReevaluationEvaluator:
    """Resolve one active-model bundle path into the default reevaluation adapter."""
    model_bundle_path = paths.resolve_work_dir_path(active_model.model_bundle_path)
    if model_bundle_path is None or not model_bundle_path.is_dir():
        missing_path = (
            active_model.model_bundle_path
            if model_bundle_path is None
            else model_bundle_path
        )
        raise _reevaluation_bundle_missing_error(missing_path)
    return MorpionActiveModelNodeReevaluationEvaluator(
        model_bundle_path=model_bundle_path,
    )


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


def load_reevaluation_training_tree_snapshot(
    snapshot_path: Path,
) -> TrainingTreeSnapshot:
    """Load a flat or Morpion sharded training tree snapshot for reevaluation."""
    if _is_sharded_training_tree_snapshot_path(snapshot_path):
        return load_morpion_sharded_training_tree_snapshot(snapshot_path)
    try:
        return load_training_tree_snapshot(snapshot_path)
    except MalformedNodesFieldError:
        return load_morpion_sharded_training_tree_snapshot(snapshot_path)


def _is_sharded_training_tree_snapshot_path(snapshot_path: Path) -> bool:
    """Return whether one snapshot reference points at a sharded export."""
    return "tree_exports_sharded" in snapshot_path.parts


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
        start_index = (
            0 if start_cursor is None else ordered_node_ids.index(start_cursor)
        )
    except ValueError:
        start_index = 0

    if max_nodes >= len(ordered_node_ids):
        selected = ordered_node_ids[start_index:] + ordered_node_ids[:start_index]
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
    use_snapshot_value_fallback: bool = False,
) -> MorpionReevaluationWorkerResult:
    """Produce at most one reevaluation patch and advance the reevaluation cursor."""
    if max_nodes_per_patch < 0:
        raise _negative_max_nodes_per_patch_error()

    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    paths.ensure_directories()
    log_pipeline_memory(
        stage="reevaluation",
        event="start",
        max_nodes=max_nodes_per_patch,
    )

    if max_nodes_per_patch == 0:
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="NO_WORK",
            reason="max_nodes_per_patch_zero",
            active_model=None,
            latest_generation=None,
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            event="done",
            rows=0,
            reason="max_nodes_per_patch_zero",
        )
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
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="IDLE",
            reason="missing_active_model",
            active_model=None,
            latest_generation=None,
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            event="done",
            rows=0,
            reason="missing_active_model",
        )
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

    LOGGER.debug(
        "[reevaluation] active_model generation=%s evaluator=%s bundle=%s",
        active_model.generation,
        active_model.evaluator_name,
        active_model.model_bundle_path,
    )

    if paths.pipeline_reevaluation_patch_path.exists():
        pending_patch_path = _relative_to_work_dir(
            paths,
            paths.pipeline_reevaluation_patch_path,
        )
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="BLOCKED",
            reason="pending_patch_exists",
            active_model=active_model,
            latest_generation=None,
            pending_patch_path=pending_patch_path,
            action="waiting_for_growth_to_consume_patch",
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=active_model.generation,
            event="done",
            rows=0,
            reason="pending_patch_exists",
        )
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
            pending_patch_path=pending_patch_path,
            model_bundle_path=active_model.model_bundle_path,
        )

    latest_snapshot = resolve_latest_reevaluation_tree_snapshot(paths)
    if latest_snapshot is None:
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="IDLE",
            reason="missing_tree_snapshot",
            active_model=active_model,
            latest_generation=None,
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=active_model.generation,
            event="done",
            rows=0,
            reason="missing_tree_snapshot",
        )
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
            model_bundle_path=active_model.model_bundle_path,
        )
    tree_generation, snapshot_path = latest_snapshot
    snapshot_log_path = _relative_to_work_dir(paths, snapshot_path)

    if not log_available_ram_guard(
        stage="reevaluation",
        generation=tree_generation,
        action="snapshot_load",
        required_mb=args.min_available_ram_mb,
    ):
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="IDLE",
            reason="low_available_ram",
            active_model=active_model,
            latest_generation=tree_generation,
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=tree_generation,
            event="done",
            rows=0,
            reason="low_available_ram",
        )
        return MorpionReevaluationWorkerResult(
            patch_written=False,
            reason="low_available_ram",
            patch_id=None,
            num_rows=0,
            evaluator_generation=active_model.generation,
            evaluator_name=active_model.evaluator_name,
            start_cursor=None,
            end_cursor=None,
            completed_full_pass_count=None,
            tree_generation=tree_generation,
            model_bundle_path=active_model.model_bundle_path,
        )

    log_pipeline_memory(
        stage="reevaluation",
        generation=tree_generation,
        event="before_snapshot_load",
        tree_snapshot_path=snapshot_path,
    )
    snapshot = load_reevaluation_training_tree_snapshot(snapshot_path)
    log_pipeline_memory(
        stage="reevaluation",
        generation=tree_generation,
        event="after_snapshot_load",
        node_count=len(snapshot.nodes),
    )
    sorted_node_ids = tuple(sorted(node.node_id for node in snapshot.nodes))
    if not sorted_node_ids:
        _maybe_log_reevaluation_idle(
            paths=paths,
            status="NO_WORK",
            reason="empty_tree_snapshot",
            active_model=active_model,
            latest_generation=tree_generation,
            next_check_s=5,
            now_unix_s=now_unix_s,
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=tree_generation,
            event="done",
            rows=0,
            reason="empty_tree_snapshot",
        )
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
            tree_generation=tree_generation,
            model_bundle_path=active_model.model_bundle_path,
        )

    try:
        persisted_cursor = load_reevaluation_cursor(
            paths.pipeline_reevaluation_cursor_path
        )
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
    _record_reevaluation_action(paths, status="STARTED", now_unix_s=now_unix_s)
    _log_reevaluation_start_block(
        active_model=active_model,
        tree_generation=tree_generation,
        snapshot_path=snapshot_log_path or str(snapshot_path),
        max_nodes_per_patch=max_nodes_per_patch,
        selected_rows=len(selected_node_ids),
        total_rows=len(sorted_node_ids),
    )

    log_pipeline_memory(
        stage="reevaluation",
        generation=tree_generation,
        event="before_patch_rows_build",
        rows=len(selected_node_ids),
    )
    patch_rows_start_unix_s = time.time()
    if evaluator is not None:
        patch_rows = tuple(evaluator.evaluate_patch_rows(snapshot, selected_node_ids))
    elif use_snapshot_value_fallback:
        patch_rows = snapshot_values_to_patch_rows(snapshot, selected_node_ids)
    else:
        active_model_evaluator = build_active_model_reevaluation_evaluator(
            paths=paths,
            active_model=active_model,
        )
        patch_rows = active_model_evaluator.evaluate_patch_rows(
            snapshot,
            selected_node_ids,
        )
    log_pipeline_memory(
        stage="reevaluation",
        generation=tree_generation,
        event="after_patch_rows_build",
        rows=len(patch_rows),
    )
    patch_rows_elapsed_s = max(time.time() - patch_rows_start_unix_s, 0.0)
    rows_per_s = (
        len(patch_rows) / patch_rows_elapsed_s if patch_rows_elapsed_s > 0.0 else 0.0
    )
    LOGGER.info(
        "[reevaluation-progress] %s",
        _format_log_fields(
            generation=active_model.generation,
            evaluator=active_model.evaluator_name,
            rows=f"{len(patch_rows)}/{len(selected_node_ids)}",
            percent=100.0 if selected_node_ids else 0.0,
            elapsed=f"{patch_rows_elapsed_s:.1f}s",
            rows_per_s=f"{rows_per_s:.1f}",
        ),
    )

    if paths.pipeline_reevaluation_patch_path.exists():
        pending_patch_path = _relative_to_work_dir(
            paths,
            paths.pipeline_reevaluation_patch_path,
        )
        LOGGER.info(
            "[reevaluation-cycle] %s",
            _format_log_fields(
                status="BLOCKED",
                reason="pending_patch_exists",
                active_model=active_model.evaluator_name,
                source_generation=active_model.generation,
                latest_generation=tree_generation,
                pending_patch=pending_patch_path,
                action="waiting_for_growth_to_consume_patch",
            ),
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=tree_generation,
            event="done",
            rows=0,
            reason="pending_patch_exists",
        )
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
            tree_generation=tree_generation,
            pending_patch_path=pending_patch_path,
            model_bundle_path=active_model.model_bundle_path,
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
    if not _save_reevaluation_patch_exclusive(
        patch,
        paths.pipeline_reevaluation_patch_path,
    ):
        pending_patch_path = _relative_to_work_dir(
            paths,
            paths.pipeline_reevaluation_patch_path,
        )
        LOGGER.info(
            "[reevaluation-cycle] %s",
            _format_log_fields(
                status="BLOCKED",
                reason="pending_patch_exists",
                active_model=active_model.evaluator_name,
                source_generation=active_model.generation,
                latest_generation=tree_generation,
                pending_patch=pending_patch_path,
                action="waiting_for_growth_to_consume_patch",
            ),
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=tree_generation,
            event="done",
            rows=0,
            reason="pending_patch_exists",
        )
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
            tree_generation=tree_generation,
            pending_patch_path=pending_patch_path,
            model_bundle_path=active_model.model_bundle_path,
        )

    LOGGER.info(
        "[reevaluation-patch] create_done patch_id=%s rows=%s direct_updates=%s tree_generation=%s start_cursor=%s end_cursor=%s",
        patch.patch_id,
        len(patch.rows),
        len(patch.rows),
        _metric_value(patch.tree_generation),
        _metric_value(patch.start_cursor),
        _metric_value(patch.end_cursor),
    )
    patch_output_path = _relative_to_work_dir(
        paths, paths.pipeline_reevaluation_patch_path
    )
    LOGGER.info(
        "[reevaluation-patch] %s",
        _format_log_fields(
            status="WRITTEN",
            generation=patch.evaluator_generation,
            evaluator=patch.evaluator_name,
            rows=len(patch.rows),
            output=patch_output_path,
            patch_id=patch.patch_id,
        ),
    )
    LOGGER.info(
        "[reevaluation-cycle] %s",
        _format_log_fields(
            generation=patch.evaluator_generation,
            evaluator=patch.evaluator_name,
            status="PATCH_WRITTEN",
            rows=len(patch.rows),
            output=patch_output_path,
            latest_tree_generation=tree_generation,
        ),
    )
    _record_reevaluation_action(
        paths,
        status="PATCH_WRITTEN",
        now_unix_s=now_unix_s,
    )

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
    log_pipeline_memory(
        stage="reevaluation",
        generation=tree_generation,
        event="done",
        rows=len(patch.rows),
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
        tree_generation=tree_generation,
        pending_patch_path=patch_output_path,
        model_bundle_path=patch.model_bundle_path,
    )


__all__ = [
    "MorpionActiveModelNodeReevaluationEvaluator",
    "MorpionNodeReevaluationEvaluator",
    "MorpionReevaluationWorkerResult",
    "build_active_model_reevaluation_evaluator",
    "cursor_matches_active_model",
    "resolve_latest_reevaluation_tree_snapshot",
    "run_morpion_reevaluation_worker_once",
    "select_reevaluation_node_window",
    "snapshot_values_to_patch_rows",
]
