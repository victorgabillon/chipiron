"""Deterministic relation-intervention definitions and tensor transforms."""
# ruff: noqa: TRY003

from __future__ import annotations

from dataclasses import dataclass

import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionEntityRelationType,
)

from .args import InvalidMorpionRelationInterventionInputError


@dataclass(frozen=True, slots=True)
class RelationInterventionDefinition:
    """A stable named set of active relation IDs to disable."""

    name: str
    relation_type_ids: tuple[int, ...]
    kind: str


def individual_relation_interventions() -> tuple[RelationInterventionDefinition, ...]:
    """Return one intervention for every active relation type."""
    return tuple(
        RelationInterventionDefinition(
            name=f"disable_{relation_type.name}",
            relation_type_ids=(int(relation_type),),
            kind="individual",
        )
        for relation_type in MorpionEntityRelationType
        if relation_type is not MorpionEntityRelationType.NO_RELATION
    )


def group_relation_interventions() -> tuple[RelationInterventionDefinition, ...]:
    """Return the required stable relation groups."""
    names = {
        "move_to_dot_slots": tuple(f"MOVE_TO_DOT_SLOT_{slot}" for slot in range(5)),
        "dot_to_move_slots": tuple(f"DOT_SLOT_{slot}_OF_MOVE" for slot in range(5)),
        "edge_dot_connectivity": ("EDGE_TOUCHES_DOT", "DOT_TOUCHES_EDGE"),
        "existing_move_window_edges": (
            "MOVE_WINDOW_CONTAINS_EDGE",
            "EDGE_IN_MOVE_WINDOW",
        ),
        "move_to_move": ("MOVES_SHARE_NEW_DOT",),
        "all_bidirectional_move_dot": (
            *(f"MOVE_TO_DOT_SLOT_{slot}" for slot in range(5)),
            *(f"DOT_SLOT_{slot}_OF_MOVE" for slot in range(5)),
        ),
    }
    return tuple(
        validate_intervention_definition(
            RelationInterventionDefinition(
                name=f"disable_group_{name}",
                relation_type_ids=tuple(
                    int(MorpionEntityRelationType[relation_name])
                    for relation_name in relation_names
                ),
                kind="group",
            )
        )
        for name, relation_names in names.items()
    )


def all_relation_interventions() -> tuple[RelationInterventionDefinition, ...]:
    """Return all-disabled, individual, and group interventions in stable order."""
    active_ids = tuple(
        int(relation_type)
        for relation_type in MorpionEntityRelationType
        if relation_type is not MorpionEntityRelationType.NO_RELATION
    )
    return (
        RelationInterventionDefinition(
            name="disable_all_relations",
            relation_type_ids=active_ids,
            kind="all",
        ),
        *individual_relation_interventions(),
        *group_relation_interventions(),
    )


def validate_intervention_definition(
    definition: RelationInterventionDefinition,
) -> RelationInterventionDefinition:
    """Reject empty, duplicate, padding, or unknown relation IDs."""
    ids = definition.relation_type_ids
    active_ids = {
        int(relation_type)
        for relation_type in MorpionEntityRelationType
        if relation_type is not MorpionEntityRelationType.NO_RELATION
    }
    if not definition.name or not ids:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "intervention definitions require a name and at least one relation ID"
        )
    if len(set(ids)) != len(ids):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"intervention {definition.name!r} contains duplicate relation IDs"
        )
    unknown = sorted(set(ids) - active_ids)
    if unknown:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"intervention {definition.name!r} contains unknown relation IDs {unknown}"
        )
    return definition


def disable_relation_types(
    relation_triples: torch.Tensor,
    relation_type_ids: tuple[int, ...],
) -> torch.Tensor:
    """Clone triples and replace only selected active type IDs with padding zero."""
    definition = validate_intervention_definition(
        RelationInterventionDefinition(
            name="tensor_intervention",
            relation_type_ids=relation_type_ids,
            kind="internal",
        )
    )
    intervened = relation_triples.clone()
    types = intervened[..., 2]
    mask = torch.zeros_like(types, dtype=torch.bool)
    for relation_type_id in definition.relation_type_ids:
        mask |= types == relation_type_id
    types[mask] = 0
    return intervened


__all__ = [
    "RelationInterventionDefinition",
    "all_relation_interventions",
    "disable_relation_types",
    "group_relation_interventions",
    "individual_relation_interventions",
    "validate_intervention_definition",
]
