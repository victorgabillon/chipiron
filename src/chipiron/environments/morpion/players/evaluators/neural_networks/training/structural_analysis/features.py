"""Per-row Morpion state structure, relation usage, and paired errors."""
# pyright: reportMissingImports=false

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import torch

from chipiron.environments.morpion.action_geometry import (
    morpion_action_new_point,
    morpion_action_points,
    morpion_action_segments,
)
from chipiron.environments.morpion.learning import decode_morpion_state_ref_payload
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionEntityRelationType,
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

if TYPE_CHECKING:
    from atomheart.games.morpion.state import Segment

    from chipiron.environments.morpion.learning import MorpionSupervisedRow
    from chipiron.environments.morpion.types import MorpionAction

    from .rows import EvaluatorRoles, PairedPrediction


PAIR_WIN_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StructuralFeatureContext:
    """Reusable canonical converters for sequential selected-row analysis."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)
    token_converter: MorpionEntityTokenConverter = field(
        default_factory=MorpionEntityTokenConverter
    )
    relational_converter: MorpionRelationalEntityTokenConverter = field(
        default_factory=MorpionRelationalEntityTokenConverter
    )


def build_structural_row(
    *,
    row: MorpionSupervisedRow,
    paired: PairedPrediction,
    evaluator_names: tuple[str, ...],
    roles: EvaluatorRoles,
    context: StructuralFeatureContext,
) -> dict[str, object]:
    """Build one stable structural/error record without evaluator inference."""
    atom_state = decode_morpion_state_ref_payload(row.state_ref_payload)
    state = context.dynamics.wrap_atomheart_state(atom_state)
    actions = context.dynamics.all_legal_actions(state)
    token_layout = context.token_converter.state_to_layout(state)
    relational = context.relational_converter.state_to_tensors(state)
    geometry = legal_action_geometry(actions)
    relation_counts = relation_counts_by_name(relational.relation_triples)
    evaluator_errors = {
        name: _error_record(paired.predictions[name], paired.target)
        for name in evaluator_names
    }
    pairs = {
        "ordinary_vs_relational": _pair_record(
            evaluator_a=roles.ordinary,
            evaluator_b=roles.relational,
            evaluator_errors=evaluator_errors,
        ),
        "mlp_vs_relational": _pair_record(
            evaluator_a=roles.mlp,
            evaluator_b=roles.relational,
            evaluator_errors=evaluator_errors,
        ),
        "mlp_vs_ordinary": _pair_record(
            evaluator_a=roles.mlp,
            evaluator_b=roles.ordinary,
            evaluator_errors=evaluator_errors,
        ),
    }
    ensembles = _ensemble_records(
        predictions=paired.predictions,
        target=paired.target,
        evaluator_names=evaluator_names,
        roles=roles,
    )
    return {
        "row_index": paired.row_index,
        "target": paired.target,
        "state": {
            "moves": state.moves,
            "move_count": state.moves,
            "num_points": len(state.points),
            "used_unit_segment_count": len(state.used_unit_segments),
            "legal_action_count": len(actions),
        },
        "tokens": {
            "total": int(token_layout.tensor.shape[0]),
            "entity_token_count": int(token_layout.tensor.shape[0]),
            "dots": len(token_layout.dot_index_by_point),
            "dot_token_count": len(token_layout.dot_index_by_point),
            "edges": len(token_layout.edge_index_by_segment),
            "edge_token_count": len(token_layout.edge_index_by_segment),
            "moves": len(token_layout.move_index_by_action),
            "move_token_count": len(token_layout.move_index_by_action),
        },
        "geometry": geometry,
        "relations": {
            "total": int(relational.relation_triples.shape[0]),
            "relation_count": int(relational.relation_triples.shape[0]),
            "active_type_count": sum(value > 0 for value in relation_counts.values()),
            "counts_by_type": relation_counts,
            "presence_by_type": {
                name: count > 0 for name, count in relation_counts.items()
            },
        },
        "evaluators": evaluator_errors,
        "pairs": pairs,
        "ensembles": ensembles,
    }


def legal_action_geometry(actions: tuple[MorpionAction, ...]) -> dict[str, object]:
    """Compute diagnostic action-window overlap features for one state."""
    action_count = len(actions)
    new_dot_counts = Counter(morpion_action_new_point(action) for action in actions)
    point_sets = [frozenset(morpion_action_points(action)) for action in actions]
    segment_sets = [frozenset(morpion_action_segments(action)) for action in actions]
    segment_multiplicities: Counter[Segment] = Counter(
        segment for segments in segment_sets for segment in segments
    )
    sharing_any_dot = 0
    sharing_two_dots = 0
    sharing_segment = 0
    overlapping_windows = 0
    for first_index, second_index in combinations(range(action_count), 2):
        shared_points = point_sets[first_index] & point_sets[second_index]
        shared_segments = segment_sets[first_index] & segment_sets[second_index]
        if shared_points:
            sharing_any_dot += 1
            overlapping_windows += 1
        if len(shared_points) >= 2:
            sharing_two_dots += 1
        if shared_segments:
            sharing_segment += 1
    unique_new_dot_count = len(new_dot_counts)
    return {
        "unique_new_dot_count": unique_new_dot_count,
        "moves_per_new_dot_mean": (
            0.0 if unique_new_dot_count == 0 else action_count / unique_new_dot_count
        ),
        "moves_per_new_dot_max": max(new_dot_counts.values(), default=0),
        "shared_new_dot_group_count": sum(
            count > 1 for count in new_dot_counts.values()
        ),
        "move_pair_count": action_count * (action_count - 1) // 2,
        "move_pairs_sharing_new_dot": sum(
            count * (count - 1) // 2 for count in new_dot_counts.values()
        ),
        "move_pairs_sharing_any_window_dot": sharing_any_dot,
        "move_pairs_sharing_two_or_more_window_dots": sharing_two_dots,
        "move_pairs_sharing_prospective_segment": sharing_segment,
        "move_pairs_with_overlapping_windows": overlapping_windows,
        "unique_prospective_segment_count": len(segment_multiplicities),
        "prospective_segment_reuse_count": sum(
            count - 1 for count in segment_multiplicities.values() if count > 1
        ),
        "maximum_prospective_segment_multiplicity": max(
            segment_multiplicities.values(), default=0
        ),
    }


def relation_counts_by_name(relation_triples: object) -> dict[str, int]:
    """Count every active declared relation type, including absent zero counts."""
    triples = relation_triples
    if not isinstance(triples, torch.Tensor):
        raise TypeError
    counts = {
        relation_type.name: 0
        for relation_type in MorpionEntityRelationType
        if relation_type is not MorpionEntityRelationType.NO_RELATION
    }
    if triples.numel() == 0:
        return counts
    type_counts = Counter(int(value) for value in triples[:, 2].tolist())
    for relation_type in MorpionEntityRelationType:
        if relation_type is MorpionEntityRelationType.NO_RELATION:
            continue
        counts[relation_type.name] = type_counts.get(int(relation_type), 0)
    return counts


def _error_record(prediction: float, target: float) -> dict[str, float]:
    """Return prediction and prediction-minus-target errors."""
    residual = prediction - target
    return {
        "prediction": prediction,
        "residual": residual,
        "absolute_error": abs(residual),
        "squared_error": residual * residual,
    }


def _pair_record(
    *,
    evaluator_a: str,
    evaluator_b: str,
    evaluator_errors: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Return signed A-over-B improvements and deterministic paired winner."""
    first = evaluator_errors[evaluator_a]
    second = evaluator_errors[evaluator_b]
    absolute_delta = second["absolute_error"] - first["absolute_error"]
    if abs(absolute_delta) <= PAIR_WIN_TOLERANCE:
        winner = "tie"
    elif absolute_delta > 0.0:
        winner = evaluator_a
    else:
        winner = evaluator_b
    return {
        "evaluator_a": evaluator_a,
        "evaluator_b": evaluator_b,
        "squared_error_improvement_a_over_b": (
            second["squared_error"] - first["squared_error"]
        ),
        "absolute_error_improvement_a_over_b": absolute_delta,
        "winner": winner,
        "prediction_disagreement": abs(first["prediction"] - second["prediction"]),
    }


def _ensemble_records(
    *,
    predictions: dict[str, float],
    target: float,
    evaluator_names: tuple[str, ...],
    roles: EvaluatorRoles,
) -> dict[str, dict[str, float]]:
    """Return fixed equal-weight ensemble predictions and errors."""
    definitions = {
        "ordinary_plus_relational": (roles.ordinary, roles.relational),
        "mlp_plus_relational": (roles.mlp, roles.relational),
        "all_evaluators": evaluator_names,
    }
    records: dict[str, dict[str, float]] = {}
    for name, members in definitions.items():
        prediction = math.fsum(predictions[member] for member in members) / len(members)
        records[name] = _error_record(prediction, target)
    return records


def structural_numeric_value(row: dict[str, object], path: str) -> float:
    """Read one numeric dotted path from a structural row."""
    current: object = row
    for component in path.split("."):
        if not isinstance(current, dict):
            raise TypeError
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TypeError
    return float(current)


__all__ = [
    "PAIR_WIN_TOLERANCE",
    "StructuralFeatureContext",
    "build_structural_row",
    "legal_action_geometry",
    "relation_counts_by_name",
    "structural_numeric_value",
]
