"""Helpers for Morpion tests that build Anemone training snapshots."""

from __future__ import annotations

from typing import Any

from anemone.training_export import TrainingNodeSnapshot


def make_training_node_snapshot(
    *,
    node_id: str,
    parent_ids: tuple[str, ...] = (),
    child_ids: tuple[str, ...] = (),
    depth: int = 0,
    state_ref_payload: object | None = None,
    direct_value_scalar: float | None = None,
    backed_up_value_scalar: float | None = None,
    tree_value_scalar: float | None = None,
    effective_value_scalar: float | None = None,
    effective_value_source: str | None = None,
    target_value_scalar: float | None = None,
    is_terminal: bool = False,
    is_exact: bool = False,
    over_event_label: str | None = None,
    visit_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrainingNodeSnapshot:
    """Build a TrainingNodeSnapshot with legacy direct/backed-up defaults."""
    resolved_tree_value_scalar = (
        backed_up_value_scalar if tree_value_scalar is None else tree_value_scalar
    )
    resolved_effective_value_scalar = effective_value_scalar
    if resolved_effective_value_scalar is None:
        resolved_effective_value_scalar = (
            backed_up_value_scalar
            if backed_up_value_scalar is not None
            else direct_value_scalar
        )
    resolved_target_value_scalar = target_value_scalar
    if resolved_target_value_scalar is None:
        resolved_target_value_scalar = (
            resolved_tree_value_scalar
            if resolved_tree_value_scalar is not None
            else resolved_effective_value_scalar
        )
    if resolved_target_value_scalar is None:
        resolved_target_value_scalar = direct_value_scalar
    resolved_effective_value_source = effective_value_source
    if resolved_effective_value_source is None:
        resolved_effective_value_source = (
            "tree_value" if resolved_tree_value_scalar is not None else "direct_value"
        )
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=parent_ids,
        child_ids=child_ids,
        depth=depth,
        state_ref_payload=state_ref_payload,
        direct_value_scalar=direct_value_scalar,
        tree_value_scalar=resolved_tree_value_scalar,
        effective_value_scalar=resolved_effective_value_scalar,
        effective_value_source=resolved_effective_value_source,
        target_value_scalar=resolved_target_value_scalar,
        backed_up_value_scalar=backed_up_value_scalar,
        is_terminal=is_terminal,
        is_exact=is_exact,
        over_event_label=over_event_label,
        visit_count=visit_count,
        metadata={} if metadata is None else dict(metadata),
    )
