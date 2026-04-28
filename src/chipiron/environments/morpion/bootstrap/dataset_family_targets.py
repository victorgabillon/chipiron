"""Dataset target smoothing helpers for Morpion bootstrap training rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
)

from .pv_family_targets import (
    PvFamilyTargetPolicy,
    PvFamilyTargets,
    family_adjusted_targets,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot


def apply_dataset_family_target_policy(
    *,
    snapshot: TrainingTreeSnapshot,
    rows: MorpionSupervisedRows,
    family_target_policy: PvFamilyTargetPolicy,
    family_prediction_blend: float = 0.25,
    use_backed_up_value: bool = True,
) -> MorpionSupervisedRows:
    """Apply PV-family target smoothing to exported bootstrap training rows."""
    if not 0.0 <= family_prediction_blend <= 1.0:
        raise ValueError("dataset_family_prediction_blend must be between 0 and 1.")  # noqa: TRY003
    raw_targets = _snapshot_raw_targets(
        snapshot=snapshot,
        use_backed_up_value=use_backed_up_value,
    )
    prediction_values = _snapshot_prediction_values(
        snapshot=snapshot,
        raw_targets=raw_targets,
    )
    exact_or_terminal_node_ids = {
        node.node_id for node in snapshot.nodes if node.is_exact or node.is_terminal
    }
    selected_child_by_node = _snapshot_selected_child_by_raw_target(
        snapshot=snapshot,
        raw_targets=raw_targets,
    )
    family_targets = family_adjusted_targets(
        raw_targets=raw_targets,
        prediction_values=prediction_values,
        exact_or_terminal_node_ids=exact_or_terminal_node_ids,
        selected_child_by_node=selected_child_by_node,
        family_target_policy=family_target_policy,
        family_prediction_blend=family_prediction_blend,
    )
    adjusted_rows = tuple(
        _dataset_row_with_family_target_metadata(
            row,
            raw_target=raw_targets.get(row.node_id, row.target_value),
            selected_child_id=selected_child_by_node.get(row.node_id),
            family_targets=family_targets,
        )
        for row in rows.rows
    )
    summary = _dataset_family_target_summary(
        rows=adjusted_rows,
        family_targets=family_targets,
        family_target_policy=family_target_policy,
        family_prediction_blend=family_prediction_blend,
    )
    return MorpionSupervisedRows(
        rows=adjusted_rows,
        metadata={**rows.metadata, **summary},
    )


def _snapshot_raw_targets(
    *,
    snapshot: TrainingTreeSnapshot,
    use_backed_up_value: bool,
) -> dict[str, float]:
    raw_targets: dict[str, float] = {}
    for node in snapshot.nodes:
        target = _node_raw_target(node, use_backed_up_value=use_backed_up_value)
        if target is not None:
            raw_targets[node.node_id] = target
    return raw_targets


def _node_raw_target(
    node: TrainingNodeSnapshot,
    *,
    use_backed_up_value: bool,
) -> float | None:
    if use_backed_up_value and node.backed_up_value_scalar is not None:
        return float(node.backed_up_value_scalar)
    if node.direct_value_scalar is not None:
        return float(node.direct_value_scalar)
    if node.backed_up_value_scalar is not None:
        return float(node.backed_up_value_scalar)
    return None


def _snapshot_prediction_values(
    *,
    snapshot: TrainingTreeSnapshot,
    raw_targets: Mapping[str, float],
) -> dict[str, float]:
    prediction_values: dict[str, float] = {}
    for node in snapshot.nodes:
        if node.direct_value_scalar is not None:
            prediction_values[node.node_id] = float(node.direct_value_scalar)
        elif node.node_id in raw_targets:
            prediction_values[node.node_id] = float(raw_targets[node.node_id])
    return prediction_values


def _snapshot_selected_child_by_raw_target(
    *,
    snapshot: TrainingTreeSnapshot,
    raw_targets: Mapping[str, float],
) -> dict[str, str | None]:
    selected_child_by_node: dict[str, str | None] = {}
    for node in snapshot.nodes:
        child_values = [
            (child_id, raw_targets[child_id])
            for child_id in sorted(node.child_ids, key=str)
            if child_id in raw_targets
        ]
        selected_child_by_node[node.node_id] = (
            None if not child_values else max(child_values, key=lambda item: item[1])[0]
        )
    return selected_child_by_node


def _dataset_row_with_family_target_metadata(
    row: MorpionSupervisedRow,
    *,
    raw_target: float,
    selected_child_id: str | None,
    family_targets: PvFamilyTargets,
) -> MorpionSupervisedRow:
    effective_target = family_targets.effective_targets.get(row.node_id, raw_target)
    metadata = dict(row.metadata)
    metadata.update(
        {
            "raw_target": raw_target,
            "effective_target": effective_target,
            "target_source": _dataset_row_target_source(
                row,
                selected_child_id=selected_child_id,
            ),
            "selected_child_id": selected_child_id,
            "family_representative_node_id": family_targets.representative_by_node.get(
                row.node_id,
                row.node_id,
            ),
            "family_size": family_targets.family_size_by_node.get(row.node_id, 1),
            "family_has_exact_or_terminal": (
                family_targets.family_has_exact_by_node.get(row.node_id, row.is_exact)
            ),
            "family_exact_target": family_targets.family_exact_target_by_node.get(
                row.node_id,
            ),
            "family_target_rule": family_targets.family_target_rule_by_node.get(
                row.node_id,
                "raw_backup",
            ),
            "family_num_exact_or_terminal": (
                family_targets.family_num_exact_by_node.get(
                    row.node_id,
                    int(row.is_exact or row.is_terminal),
                )
            ),
        },
    )
    return MorpionSupervisedRow(
        node_id=row.node_id,
        state_ref_payload=dict(row.state_ref_payload),
        target_value=effective_target,
        is_terminal=row.is_terminal,
        is_exact=row.is_exact,
        depth=row.depth,
        visit_count=row.visit_count,
        direct_value=row.direct_value,
        over_event_label=row.over_event_label,
        metadata=metadata,
    )


def _dataset_row_target_source(
    row: MorpionSupervisedRow,
    *,
    selected_child_id: str | None,
) -> str:
    if row.is_exact or row.is_terminal:
        return "ground_truth_exact_or_terminal"
    if selected_child_id is not None:
        return "child_backup"
    return "frontier_prediction"


def _dataset_family_target_summary(
    *,
    rows: tuple[MorpionSupervisedRow, ...],
    family_targets: PvFamilyTargets,
    family_target_policy: PvFamilyTargetPolicy,
    family_prediction_blend: float,
) -> dict[str, object]:
    exact_family_deltas: list[float] = []
    non_exact_family_deltas: list[float] = []
    all_deltas: list[float] = []
    exact_family_representatives = {
        family_targets.representative_by_node[node_id]
        for node_id, has_exact in family_targets.family_has_exact_by_node.items()
        if has_exact and node_id in family_targets.representative_by_node
    }
    for row in rows:
        raw_target = _metadata_float(row.metadata.get("raw_target"))
        effective_target = _metadata_float(row.metadata.get("effective_target"))
        if raw_target is None or effective_target is None:
            continue
        delta = abs(effective_target - raw_target)
        all_deltas.append(delta)
        if bool(row.metadata.get("family_has_exact_or_terminal", False)):
            exact_family_deltas.append(delta)
        else:
            non_exact_family_deltas.append(delta)
    fraction_rows_in_exact_family = (
        None
        if not rows
        else sum(
            1
            for row in rows
            if bool(row.metadata.get("family_has_exact_or_terminal", False))
        )
        / len(rows)
    )
    return {
        "dataset_family_target_policy": family_target_policy,
        "dataset_family_prediction_blend": family_prediction_blend,
        "fraction_rows_in_exact_family": fraction_rows_in_exact_family,
        "num_exact_families": len(exact_family_representatives),
        "mean_abs_effective_minus_raw_on_exact_families": _mean(
            exact_family_deltas,
        ),
        "mean_abs_effective_minus_raw_on_non_exact_families": _mean(
            non_exact_family_deltas,
        ),
        "effective_minus_raw_mean_abs": _mean(all_deltas),
        "effective_minus_raw_max_abs": max(all_deltas, default=None),
        "num_pv_families": family_targets.num_families,
        "mean_pv_family_size": family_targets.mean_family_size,
        "max_pv_family_size": family_targets.max_family_size,
    }


def _metadata_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = ["apply_dataset_family_target_policy"]
