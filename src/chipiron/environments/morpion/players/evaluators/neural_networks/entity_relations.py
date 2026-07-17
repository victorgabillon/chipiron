"""Sparse structural relations for clean Morpion entity-token inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Final, cast

import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
    canonical_segment,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    DIRECTIONS,
)
from chipiron.environments.morpion.types import (
    MorpionAction,
    MorpionDynamics,
    MorpionState,
)

if TYPE_CHECKING:
    from atomheart.games.morpion.state import Point, Segment

MORPION_ENTITY_RELATION_SCHEMA: Final[str] = "morpion_entity_relations_v1"
MORPION_ENTITY_RELATION_TYPE_COUNT: Final[int] = 16


class MorpionEntityRelationType(IntEnum):
    """Directed relation types for Morpion entity-token inputs."""

    NO_RELATION = 0

    MOVE_TO_DOT_SLOT_0 = 1
    MOVE_TO_DOT_SLOT_1 = 2
    MOVE_TO_DOT_SLOT_2 = 3
    MOVE_TO_DOT_SLOT_3 = 4
    MOVE_TO_DOT_SLOT_4 = 5

    DOT_SLOT_0_OF_MOVE = 6
    DOT_SLOT_1_OF_MOVE = 7
    DOT_SLOT_2_OF_MOVE = 8
    DOT_SLOT_3_OF_MOVE = 9
    DOT_SLOT_4_OF_MOVE = 10

    EDGE_TOUCHES_DOT = 11
    DOT_TOUCHES_EDGE = 12

    MOVE_WINDOW_CONTAINS_EDGE = 13
    EDGE_IN_MOVE_WINDOW = 14

    MOVES_SHARE_NEW_DOT = 15


assert len(MorpionEntityRelationType) == MORPION_ENTITY_RELATION_TYPE_COUNT

_MOVE_TO_DOT_RELATIONS: Final[tuple[MorpionEntityRelationType, ...]] = (
    MorpionEntityRelationType.MOVE_TO_DOT_SLOT_0,
    MorpionEntityRelationType.MOVE_TO_DOT_SLOT_1,
    MorpionEntityRelationType.MOVE_TO_DOT_SLOT_2,
    MorpionEntityRelationType.MOVE_TO_DOT_SLOT_3,
    MorpionEntityRelationType.MOVE_TO_DOT_SLOT_4,
)
_DOT_TO_MOVE_RELATIONS: Final[tuple[MorpionEntityRelationType, ...]] = (
    MorpionEntityRelationType.DOT_SLOT_0_OF_MOVE,
    MorpionEntityRelationType.DOT_SLOT_1_OF_MOVE,
    MorpionEntityRelationType.DOT_SLOT_2_OF_MOVE,
    MorpionEntityRelationType.DOT_SLOT_3_OF_MOVE,
    MorpionEntityRelationType.DOT_SLOT_4_OF_MOVE,
)


class _InvalidGeneratedMorpionRelationsError(ValueError):
    """Raised when generated relation triples violate internal invariants."""

    @classmethod
    def invalid_shape_or_dtype(cls) -> _InvalidGeneratedMorpionRelationsError:
        """Return an invalid relation tensor layout error."""
        return cls("Morpion relation triples must have shape [R, 3] and dtype long.")

    @classmethod
    def missing_entity(cls) -> _InvalidGeneratedMorpionRelationsError:
        """Return an invalid entity index error."""
        return cls("Morpion relation triples reference a missing entity token.")

    @classmethod
    def invalid_relation_type(cls) -> _InvalidGeneratedMorpionRelationsError:
        """Return an invalid active relation type error."""
        return cls("Morpion active relation types must be in the range [1, 15].")


@dataclass(frozen=True, slots=True)
class MorpionRelationalEntityTokens:
    """Morpion entity tokens and directed sparse relation triples."""

    token_tensor: torch.Tensor
    relation_triples: torch.Tensor


@dataclass(frozen=True, slots=True)
class MorpionRelationalEntityTokenConverter:
    """Build clean Morpion entity tokens and sparse structural relations."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)
    max_tokens: int = 1536

    def state_to_tensors(
        self,
        state: MorpionState,
    ) -> MorpionRelationalEntityTokens:
        """Return clean entity tokens and deterministic sparse relations."""
        entity_layout = MorpionEntityTokenConverter(
            dynamics=self.dynamics,
            max_tokens=self.max_tokens,
        ).state_to_layout(state)
        relations: set[tuple[int, int, int]] = set()
        move_indices_by_new_dot: dict[Point, list[int]] = {}

        for action, move_index in entity_layout.move_index_by_action.items():
            points = _points_for_action(action)
            for slot, point in enumerate(points):
                dot_index = entity_layout.dot_index_by_point.get(point)
                if dot_index is None:
                    continue
                _add_relation(
                    relations,
                    move_index,
                    dot_index,
                    _MOVE_TO_DOT_RELATIONS[slot],
                )
                _add_relation(
                    relations,
                    dot_index,
                    move_index,
                    _DOT_TO_MOVE_RELATIONS[slot],
                )

            for segment in _segments_for_action(action):
                edge_index = entity_layout.edge_index_by_segment.get(
                    canonical_segment(segment)
                )
                if edge_index is None:
                    continue
                _add_relation(
                    relations,
                    move_index,
                    edge_index,
                    MorpionEntityRelationType.MOVE_WINDOW_CONTAINS_EDGE,
                )
                _add_relation(
                    relations,
                    edge_index,
                    move_index,
                    MorpionEntityRelationType.EDGE_IN_MOVE_WINDOW,
                )

            missing_point = _missing_point_for_action(action)
            move_indices_by_new_dot.setdefault(missing_point, []).append(move_index)

        for segment, edge_index in entity_layout.edge_index_by_segment.items():
            for point in segment:
                dot_index = entity_layout.dot_index_by_point.get(point)
                if dot_index is None:
                    continue
                _add_relation(
                    relations,
                    edge_index,
                    dot_index,
                    MorpionEntityRelationType.EDGE_TOUCHES_DOT,
                )
                _add_relation(
                    relations,
                    dot_index,
                    edge_index,
                    MorpionEntityRelationType.DOT_TOUCHES_EDGE,
                )

        for move_indices in move_indices_by_new_dot.values():
            for source_index in move_indices:
                for destination_index in move_indices:
                    if source_index == destination_index:
                        continue
                    _add_relation(
                        relations,
                        source_index,
                        destination_index,
                        MorpionEntityRelationType.MOVES_SHARE_NEW_DOT,
                    )

        ordered_relations = sorted(relations)
        if ordered_relations:
            relation_triples = torch.tensor(ordered_relations, dtype=torch.long)
        else:
            relation_triples = torch.empty((0, 3), dtype=torch.long)
        _validate_generated_relation_triples(
            relation_triples,
            token_count=int(entity_layout.tensor.shape[0]),
        )
        return MorpionRelationalEntityTokens(
            token_tensor=entity_layout.tensor,
            relation_triples=relation_triples,
        )

    def state_to_model_input_tensors(
        self,
        state: MorpionState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two positional tensors consumed by a relational model."""
        relational = self.state_to_tensors(state)
        return relational.token_tensor, relational.relation_triples


def _add_relation(
    relations: set[tuple[int, int, int]],
    source_index: int,
    destination_index: int,
    relation_type: MorpionEntityRelationType,
) -> None:
    """Add one active directed relation triple."""
    relations.add((source_index, destination_index, int(relation_type)))


def _points_for_action(
    action: MorpionAction,
) -> tuple[Point, Point, Point, Point, Point]:
    """Return the five ordered lattice points in one action window."""
    direction_index, x0, y0, _missing_index = action
    dx, dy = DIRECTIONS[direction_index]
    return cast(
        "tuple[Point, Point, Point, Point, Point]",
        tuple((x0 + slot * dx, y0 + slot * dy) for slot in range(5)),
    )


def _segments_for_action(
    action: MorpionAction,
) -> tuple[Segment, Segment, Segment, Segment]:
    """Return the four canonical unit segments in one action window."""
    points = _points_for_action(action)
    return cast(
        "tuple[Segment, Segment, Segment, Segment]",
        tuple(
            canonical_segment((points[index], points[index + 1]))
            for index in range(4)
        ),
    )


def _missing_point_for_action(action: MorpionAction) -> Point:
    """Return the absent or new point represented by one action."""
    return _points_for_action(action)[action[3]]


def _validate_generated_relation_triples(
    relation_triples: torch.Tensor,
    *,
    token_count: int,
) -> None:
    """Raise when generated sparse relation triples violate their contract."""
    if (
        relation_triples.ndim != 2
        or relation_triples.shape[1] != 3
        or relation_triples.dtype != torch.long
    ):
        raise _InvalidGeneratedMorpionRelationsError.invalid_shape_or_dtype()
    if relation_triples.numel() == 0:
        return
    entity_indices = relation_triples[:, :2]
    relation_types = relation_triples[:, 2]
    if (
        int(entity_indices.min().item()) < 0
        or int(entity_indices.max().item()) >= token_count
    ):
        raise _InvalidGeneratedMorpionRelationsError.missing_entity()
    if (
        int(relation_types.min().item()) < 1
        or int(relation_types.max().item()) >= MORPION_ENTITY_RELATION_TYPE_COUNT
    ):
        raise _InvalidGeneratedMorpionRelationsError.invalid_relation_type()


__all__ = [
    "MORPION_ENTITY_RELATION_SCHEMA",
    "MORPION_ENTITY_RELATION_TYPE_COUNT",
    "MorpionEntityRelationType",
    "MorpionRelationalEntityTokenConverter",
    "MorpionRelationalEntityTokens",
]
