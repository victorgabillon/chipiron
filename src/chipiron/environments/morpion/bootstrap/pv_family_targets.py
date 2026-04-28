"""Principal-variation family target smoothing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PvFamilyTargetPolicy = Literal[
    "none",
    "pv_mean_prediction",
    "pv_min_prediction",
    "pv_blend_mean_prediction",
    "pv_blend_min_prediction",
    "pv_exact_then_mean_prediction",
    "pv_exact_then_min_prediction",
    "pv_exact_then_blend_mean_prediction",
    "pv_exact_then_blend_min_prediction",
]


@dataclass(frozen=True, slots=True)
class PvFamilyTargets:
    """Effective targets and metadata for PV-family target smoothing."""

    effective_targets: dict[str, float]
    representative_by_node: dict[str, str]
    family_size_by_node: dict[str, int]
    family_has_exact_by_node: dict[str, bool]
    family_exact_target_by_node: dict[str, float | None]
    family_target_rule_by_node: dict[str, str]
    family_num_exact_by_node: dict[str, int]
    num_families: int
    mean_family_size: float | None
    max_family_size: int | None


def principal_variation_families_from_selected_child(
    selected_child_by_node: dict[str, str | None],
) -> dict[str, tuple[str, ...]]:
    """Group nodes by the representative reached by following selected children."""
    known_node_ids = set(selected_child_by_node)
    known_node_ids.update(
        child_id
        for child_id in selected_child_by_node.values()
        if child_id is not None
    )
    families: dict[str, list[str]] = {}
    for node_id in sorted(known_node_ids):
        representative = _principal_variation_representative(
            node_id,
            selected_child_by_node,
        )
        families.setdefault(representative, []).append(node_id)
    return {
        representative: tuple(node_ids)
        for representative, node_ids in families.items()
    }


def family_adjusted_targets(
    *,
    raw_targets: dict[str, float],
    prediction_values: dict[str, float],
    exact_or_terminal_node_ids: set[str],
    selected_child_by_node: dict[str, str | None],
    family_target_policy: PvFamilyTargetPolicy,
    family_prediction_blend: float = 0.25,
) -> PvFamilyTargets:
    """Return effective targets after optional PV-family prediction smoothing."""
    if not 0.0 <= family_prediction_blend <= 1.0:
        raise ValueError("family_prediction_blend must be between 0 and 1.")  # noqa: TRY003
    families = principal_variation_families_from_selected_child(
        selected_child_by_node,
    )
    representative_by_node: dict[str, str] = {}
    family_size_by_node: dict[str, int] = {}
    family_has_exact_by_node: dict[str, bool] = {}
    family_exact_target_by_node: dict[str, float | None] = {}
    family_target_rule_by_node: dict[str, str] = {}
    family_num_exact_by_node: dict[str, int] = {}
    effective_targets: dict[str, float] = {}

    for representative, node_ids in families.items():
        family_size = len(node_ids)
        family_predictions = [
            prediction_values[node_id]
            for node_id in node_ids
            if node_id in prediction_values
        ]
        family_mean = _mean(family_predictions)
        if family_mean is None:
            family_mean = 0.0
        family_min = min(family_predictions, default=0.0)
        exact_node_ids = [
            node_id for node_id in node_ids if node_id in exact_or_terminal_node_ids
        ]
        exact_targets = [raw_targets[node_id] for node_id in exact_node_ids]
        exact_family_target = max(exact_targets, default=None)
        has_exact_family = bool(exact_node_ids)
        exact_family_rule = (
            "pv_exact_family_multi_max"
            if len(set(exact_targets)) > 1
            else "pv_exact_family"
        )
        for node_id in node_ids:
            representative_by_node[node_id] = representative
            family_size_by_node[node_id] = family_size
            family_has_exact_by_node[node_id] = has_exact_family
            family_exact_target_by_node[node_id] = exact_family_target
            family_num_exact_by_node[node_id] = len(exact_node_ids)

            if (
                _uses_exact_family_rule(family_target_policy)
                and exact_family_target is not None
            ):
                effective_targets[node_id] = exact_family_target
                family_target_rule_by_node[node_id] = exact_family_rule
            elif node_id in exact_or_terminal_node_ids:
                effective_targets[node_id] = raw_targets[node_id]
                family_target_rule_by_node[node_id] = "hard_exact_anchor"
            elif family_target_policy == "none":
                effective_targets[node_id] = raw_targets[node_id]
                family_target_rule_by_node[node_id] = "raw_backup"
            elif family_target_policy == "pv_mean_prediction":
                effective_targets[node_id] = family_mean
                family_target_rule_by_node[node_id] = "pv_mean_prediction"
            elif family_target_policy == "pv_min_prediction":
                effective_targets[node_id] = family_min
                family_target_rule_by_node[node_id] = "pv_min_prediction"
            elif family_target_policy == "pv_blend_mean_prediction":
                effective_targets[node_id] = (
                    (1.0 - family_prediction_blend) * raw_targets[node_id]
                    + family_prediction_blend * family_mean
                )
                family_target_rule_by_node[node_id] = "pv_blend_mean_prediction"
            elif family_target_policy == "pv_blend_min_prediction":
                effective_targets[node_id] = (
                    (1.0 - family_prediction_blend) * raw_targets[node_id]
                    + family_prediction_blend * family_min
                )
                family_target_rule_by_node[node_id] = "pv_blend_min_prediction"
            elif family_target_policy == "pv_exact_then_mean_prediction":
                effective_targets[node_id] = family_mean
                family_target_rule_by_node[node_id] = "pv_mean_prediction"
            elif family_target_policy == "pv_exact_then_min_prediction":
                effective_targets[node_id] = family_min
                family_target_rule_by_node[node_id] = "pv_min_prediction"
            elif family_target_policy == "pv_exact_then_blend_mean_prediction":
                effective_targets[node_id] = (
                    (1.0 - family_prediction_blend) * raw_targets[node_id]
                    + family_prediction_blend * family_mean
                )
                family_target_rule_by_node[node_id] = "pv_blend_mean_prediction"
            elif family_target_policy == "pv_exact_then_blend_min_prediction":
                effective_targets[node_id] = (
                    (1.0 - family_prediction_blend) * raw_targets[node_id]
                    + family_prediction_blend * family_min
                )
                family_target_rule_by_node[node_id] = "pv_blend_min_prediction"
            else:
                raise ValueError(  # noqa: TRY003
                    f"Unknown family target policy: {family_target_policy!r}."
                )

    for node_id, raw_target in raw_targets.items():
        if node_id in effective_targets:
            continue
        effective_targets[node_id] = raw_target
        representative_by_node[node_id] = node_id
        family_size_by_node[node_id] = 1
        family_has_exact_by_node[node_id] = node_id in exact_or_terminal_node_ids
        family_exact_target_by_node[node_id] = (
            raw_target if node_id in exact_or_terminal_node_ids else None
        )
        family_target_rule_by_node[node_id] = (
            "hard_exact_anchor" if node_id in exact_or_terminal_node_ids else "raw_backup"
        )
        family_num_exact_by_node[node_id] = int(node_id in exact_or_terminal_node_ids)
    family_sizes = [len(node_ids) for node_ids in families.values()]
    return PvFamilyTargets(
        effective_targets=effective_targets,
        representative_by_node=representative_by_node,
        family_size_by_node=family_size_by_node,
        family_has_exact_by_node=family_has_exact_by_node,
        family_exact_target_by_node=family_exact_target_by_node,
        family_target_rule_by_node=family_target_rule_by_node,
        family_num_exact_by_node=family_num_exact_by_node,
        num_families=len(families),
        mean_family_size=_mean([float(size) for size in family_sizes]),
        max_family_size=max(family_sizes, default=None),
    )


def _principal_variation_representative(
    node_id: str,
    selected_child_by_node: dict[str, str | None],
) -> str:
    """Return the final node reached by following selected-child links."""
    seen: set[str] = set()
    current_id = node_id
    while True:
        if current_id in seen:
            raise ValueError(  # noqa: TRY003
                "Principal-variation family traversal found a cycle."
            )
        seen.add(current_id)
        next_id = selected_child_by_node.get(current_id)
        if next_id is None:
            return current_id
        current_id = next_id


def _uses_exact_family_rule(
    family_target_policy: PvFamilyTargetPolicy,
) -> bool:
    return family_target_policy.startswith("pv_exact_then_")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "PvFamilyTargetPolicy",
    "PvFamilyTargets",
    "family_adjusted_targets",
    "principal_variation_families_from_selected_child",
]
