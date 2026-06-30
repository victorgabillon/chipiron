"""Live-tree reevaluation patch application for Morpion runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone.value_updates import NodeValueUpdate, NodeValueUpdateResult

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )

__all__ = [
    "ReevaluationBlendMetrics",
    "apply_blended_reevaluation_patch",
]


@dataclass(slots=True)
class ReevaluationBlendMetrics:
    """Aggregate diagnostics for smoothed reevaluation patch updates."""

    count: int = 0
    old_sum: float = 0.0
    new_sum: float = 0.0
    blended_sum: float = 0.0

    def record(
        self,
        *,
        old_value: float,
        new_value: float,
        blended_value: float,
    ) -> None:
        """Record one actually blended node update."""
        self.count += 1
        self.old_sum += old_value
        self.new_sum += new_value
        self.blended_sum += blended_value


def _uninitialized_reevaluation_patch_runtime_error() -> RuntimeError:
    """Build the stable missing-runtime error for live patch application."""
    return RuntimeError(
        "Cannot apply Morpion reevaluation patch before the Anemone search runtime is initialized."
    )


def apply_blended_reevaluation_patch(
    *,
    runtime: Any,
    patch: MorpionReevaluationPatch,
    blend_alpha: float,
    blend_metrics: ReevaluationBlendMetrics,
) -> tuple[NodeValueUpdateResult, bool]:
    """Apply one smoothed patch using Anemone's existing single node lookup pass."""
    nodes_by_id = runtime._nodes_by_public_id()  # pylint: disable=protected-access
    missing_node_ids = tuple(
        row.node_id for row in patch.rows if row.node_id not in nodes_by_id
    )
    applied_nodes: list[object] = []
    changed_nodes: list[object] = []
    for row in patch.rows:
        live_node = nodes_by_id.get(row.node_id)
        if live_node is None:
            continue
        update = NodeValueUpdate(
            node_id=row.node_id,
            direct_value=_reevaluation_patch_direct_value(
                row=row,
                live_node=live_node,
                blend_alpha=blend_alpha,
                blend_metrics=blend_metrics,
            ),
            backed_up_value=row.backed_up_value,
            is_exact=row.is_exact,
            is_terminal=row.is_terminal,
            metadata=row.metadata,
        )
        changed = bool(
            runtime._apply_node_value_update(  # pylint: disable=protected-access
                node=live_node,
                update=update,
            )
        )
        applied_nodes.append(live_node)
        if changed:
            changed_nodes.append(live_node)

    recomputed_count = _recompute_after_blended_reevaluation(
        runtime=runtime,
        changed_nodes=changed_nodes,
    )
    selector_invalidated = False
    if changed_nodes:
        selector_invalidated = _invalidate_selector_cache_if_supported(runtime)
    return (
        NodeValueUpdateResult(
            requested_count=len(patch.rows),
            applied_count=len(applied_nodes),
            missing_node_ids=missing_node_ids,
            recomputed_count=recomputed_count,
        ),
        selector_invalidated,
    )


def _recompute_after_blended_reevaluation(
    *,
    runtime: Any,
    changed_nodes: list[object],
) -> int:
    """Refresh backups and exploration indices after local smoothed value writes."""
    if not changed_nodes:
        return 0
    tree_manager = runtime.tree_manager
    recomputed_nodes = (
        tree_manager.value_propagator.propagate_after_local_value_changes(changed_nodes)
    )
    tree_manager.refresh_exploration_indices(tree=runtime.tree)
    return len(recomputed_nodes)


def _reevaluation_patch_direct_value(
    *,
    row: MorpionReevaluationPatchRow,
    live_node: object | None,
    blend_alpha: float,
    blend_metrics: ReevaluationBlendMetrics,
) -> float:
    """Return the direct value to write for one reevaluation patch row."""
    new_value = float(row.direct_value)
    if blend_alpha >= 1.0 or live_node is None:
        return new_value
    if _patch_row_is_authoritative(row) or _live_node_is_authoritative(live_node):
        return new_value

    old_value = _live_node_direct_value_score(live_node)
    if old_value is None:
        return new_value

    blended_value = ((1.0 - blend_alpha) * old_value) + (blend_alpha * new_value)
    blend_metrics.record(
        old_value=old_value,
        new_value=new_value,
        blended_value=blended_value,
    )
    return blended_value


def _patch_row_is_authoritative(row: object) -> bool:
    """Return whether one patch row should bypass smoothing."""
    return (
        getattr(row, "is_exact", None) is True
        or getattr(row, "is_terminal", None) is True
    )


def _live_node_is_authoritative(node: object) -> bool:
    """Return whether one live node has terminal or exact authoritative value state."""
    tree_evaluation = getattr(node, "tree_evaluation", None)
    has_exact_value = getattr(tree_evaluation, "has_exact_value", None)
    if callable(has_exact_value) and bool(has_exact_value()):
        return True

    state = getattr(node, "state", None)
    is_game_over = getattr(state, "is_game_over", None)
    if callable(is_game_over) and bool(is_game_over()):
        return True

    is_terminal = getattr(tree_evaluation, "is_terminal", None)
    return bool(is_terminal()) if callable(is_terminal) else False


def _live_node_direct_value_score(node: object) -> float | None:
    """Return one live node's direct-value score when present."""
    tree_evaluation = getattr(node, "tree_evaluation", None)
    direct_value = getattr(tree_evaluation, "direct_value", None)
    if direct_value is None:
        return None
    score = getattr(direct_value, "score", direct_value)
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def _blend_average(
    value_sum: float,
    metrics: ReevaluationBlendMetrics,
) -> float | None:
    """Return one blend aggregate average, or None when no rows were blended."""
    if metrics.count == 0:
        return None
    return value_sum / metrics.count


def _invalidate_selector_cache_if_supported(runtime: object) -> bool:
    """Invalidate selector caches when the runtime or selector exposes a hook."""
    invalidate_runtime = getattr(
        runtime, "_invalidate_selector_cache_if_supported", None
    )
    if callable(invalidate_runtime):
        return bool(invalidate_runtime())

    selector = getattr(runtime, "node_selector", None)
    invalidate_selector = getattr(selector, "invalidate", None)
    if callable(invalidate_selector):
        invalidate_selector()
        return True

    return False
