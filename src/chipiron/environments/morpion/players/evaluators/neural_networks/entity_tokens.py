"""Entity-token tensor conversion for Morpion neural evaluators."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_extractor import (
    DIRECTIONS,
)
from chipiron.environments.morpion.types import (
    MorpionAction,
    MorpionDynamics,
    MorpionState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atomheart.games.morpion.state import Point, Segment

MORPION_ENTITY_TOKEN_MODEL_KIND: Final[str] = "entity_token_transformer_value_net"
MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION: Final[str] = "morpion_entity_tokens_v1"


def is_morpion_entity_token_model_kind(model_kind: str) -> bool:
    """Return whether a model kind consumes Morpion entity-token tensors."""
    return model_kind == MORPION_ENTITY_TOKEN_MODEL_KIND


class MorpionEntityTokenType(IntEnum):
    """Entity types used by the first clean Morpion token representation."""

    GLOBAL = 0
    DOT = 1
    EDGE = 2
    MOVE = 3


MORPION_ENTITY_TOKEN_DIRECTIONS: Final[tuple[str, ...]] = (
    "horizontal",
    "vertical",
    "diag_up",
    "diag_down",
)

MORPION_ENTITY_TOKEN_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "type_global",
    "type_dot",
    "type_edge",
    "type_move",
    "x_rel",
    "y_rel",
    "dir_horizontal",
    "dir_vertical",
    "dir_diag_up",
    "dir_diag_down",
    "occupied",
    "candidate",
    "drawn",
    "degree_horizontal",
    "degree_vertical",
    "degree_diag_up",
    "degree_diag_down",
    "is_new_dot_for_move",
    "missing_index_in_5_window",
    "path_position",
    "num_points",
    "moves",
    "legal_action_count",
    "used_unit_segment_count",
    "validity",
)
MORPION_ENTITY_TOKEN_FEATURE_DIM: Final[int] = len(MORPION_ENTITY_TOKEN_FEATURE_NAMES)

_VALIDITY_INDEX: Final[int] = MORPION_ENTITY_TOKEN_FEATURE_DIM - 1


class InvalidMorpionEntityTokenConverterError(ValueError):
    """Raised when entity-token conversion is configured invalidly."""

    @classmethod
    def invalid_max_tokens(cls) -> InvalidMorpionEntityTokenConverterError:
        """Return the invalid max-token-count error."""
        return cls("Morpion entity-token conversion requires max_tokens >= 1.")


@dataclass(frozen=True, slots=True)
class MorpionEntityTokenLayout:
    """One entity tensor and stable indices for every surviving entity."""

    tensor: Tensor
    dot_index_by_point: Mapping[Point, int]
    edge_index_by_segment: Mapping[Segment, int]
    move_index_by_action: Mapping[MorpionAction, int]


@dataclass(frozen=True, slots=True)
class _MorpionEntityTokenRecord:
    """A token row and its optional game-entity identity."""

    row: list[float]
    point: Point | None = None
    segment: Segment | None = None
    action: MorpionAction | None = None


@dataclass(frozen=True, slots=True)
class _CoordinateNormalizer:
    """Normalize lattice coordinates into a state-local bounded scale."""

    center_x: float
    center_y: float
    scale: float

    def point(self, point: Point) -> tuple[float, float]:
        """Return normalized coordinates for one lattice point."""
        return (
            (float(point[0]) - self.center_x) / self.scale,
            (float(point[1]) - self.center_y) / self.scale,
        )

    def segment_center(self, segment: Segment) -> tuple[float, float]:
        """Return normalized coordinates for the midpoint of one segment."""
        (x0, y0), (x1, y1) = segment
        return (
            ((float(x0) + float(x1)) * 0.5 - self.center_x) / self.scale,
            ((float(y0) + float(y1)) * 0.5 - self.center_y) / self.scale,
        )


@dataclass(frozen=True, slots=True)
class MorpionEntityTokenConverter:
    """Convert Morpion states into variable-length entity-token tensors."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)
    max_tokens: int = 1536

    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered entity-token feature names."""
        return MORPION_ENTITY_TOKEN_FEATURE_NAMES

    @property
    def input_dim(self) -> int:
        """Return the entity-token feature width."""
        return MORPION_ENTITY_TOKEN_FEATURE_DIM

    def state_to_tensor(self, state: MorpionState) -> Tensor:
        """Return a ``T x F`` float32 tensor of real entity tokens for ``state``."""
        return self.state_to_layout(state).tensor

    def state_to_model_input_tensors(
        self,
        state: MorpionState,
    ) -> tuple[Tensor, ...]:
        """Return the ordinary model's single positional input tensor."""
        return (self.state_to_tensor(state),)

    def state_to_layout(self, state: MorpionState) -> MorpionEntityTokenLayout:
        """Return tokens and stable tensor indices for every surviving entity."""
        if self.max_tokens < 1:
            raise InvalidMorpionEntityTokenConverterError.invalid_max_tokens()

        normalizer = _normalizer_for_state(state)
        actions = self.dynamics.all_legal_actions(state)
        candidate_points = _candidate_points_from_actions(actions)
        records: list[_MorpionEntityTokenRecord] = [
            _MorpionEntityTokenRecord(
                row=_global_token(state=state, legal_action_count=len(actions))
            )
        ]
        records.extend(
            _MorpionEntityTokenRecord(
                row=_dot_token(
                    point=point,
                    state=state,
                    candidate=point in candidate_points,
                    normalizer=normalizer,
                ),
                point=point,
            )
            for point in sorted(state.points)
        )
        records.extend(
            _MorpionEntityTokenRecord(
                row=_dot_token(
                    point=point,
                    state=state,
                    candidate=True,
                    normalizer=normalizer,
                ),
                point=point,
            )
            for point in sorted(candidate_points - state.points)
        )
        edge_records: list[_MorpionEntityTokenRecord] = []
        for raw_segment in sorted(state.used_unit_segments, key=_segment_sort_key):
            segment = canonical_segment(raw_segment)
            edge_records.append(
                _MorpionEntityTokenRecord(
                    row=_edge_token(segment=segment, normalizer=normalizer),
                    segment=segment,
                )
            )
        records.extend(edge_records)
        records.extend(
            _MorpionEntityTokenRecord(
                row=_move_token(action=action, normalizer=normalizer),
                action=action,
            )
            for action in sorted(actions, key=_action_sort_key)
        )

        # TODO: replace deterministic tail truncation with priority-aware truncation.
        surviving_records = records[: self.max_tokens]
        dot_index_by_point: dict[Point, int] = {}
        edge_index_by_segment: dict[Segment, int] = {}
        move_index_by_action: dict[MorpionAction, int] = {}
        for index, record in enumerate(surviving_records):
            if record.point is not None:
                dot_index_by_point[record.point] = index
            if record.segment is not None:
                edge_index_by_segment[record.segment] = index
            if record.action is not None:
                move_index_by_action[record.action] = index

        return MorpionEntityTokenLayout(
            tensor=torch.tensor(
                [record.row for record in surviving_records], dtype=torch.float32
            ),
            dot_index_by_point=MappingProxyType(dot_index_by_point),
            edge_index_by_segment=MappingProxyType(edge_index_by_segment),
            move_index_by_action=MappingProxyType(move_index_by_action),
        )


def _blank_token(token_type: MorpionEntityTokenType) -> list[float]:
    """Return one zero-filled token row with type and validity set."""
    row = [0.0] * MORPION_ENTITY_TOKEN_FEATURE_DIM
    row[token_type.value] = 1.0
    row[_VALIDITY_INDEX] = 1.0
    return row


def _global_token(*, state: MorpionState, legal_action_count: int) -> list[float]:
    """Return one global-context token."""
    row = _blank_token(MorpionEntityTokenType.GLOBAL)
    row[_feature_index("num_points")] = float(len(state.points))
    row[_feature_index("moves")] = float(state.moves)
    row[_feature_index("legal_action_count")] = float(legal_action_count)
    row[_feature_index("used_unit_segment_count")] = float(
        len(state.used_unit_segments)
    )
    return row


def _dot_token(
    *,
    point: Point,
    state: MorpionState,
    candidate: bool,
    normalizer: _CoordinateNormalizer,
) -> list[float]:
    """Return one occupied or candidate dot token."""
    row = _blank_token(MorpionEntityTokenType.DOT)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index("occupied")] = 1.0 if point in state.points else 0.0
    row[_feature_index("candidate")] = 1.0 if candidate else 0.0
    dir_usage = state.dir_usage
    for dir_index, name in enumerate(MORPION_ENTITY_TOKEN_DIRECTIONS):
        row[_feature_index(f"degree_{name}")] = float(
            dir_usage.get((point, dir_index), 0)
        )
    return row


def _edge_token(
    *,
    segment: Segment,
    normalizer: _CoordinateNormalizer,
) -> list[float]:
    """Return one drawn unit-edge token."""
    row = _blank_token(MorpionEntityTokenType.EDGE)
    x_rel, y_rel = normalizer.segment_center(segment)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index("drawn")] = 1.0
    dir_index = _direction_index_for_segment(segment)
    if dir_index is not None:
        row[_feature_index(f"dir_{MORPION_ENTITY_TOKEN_DIRECTIONS[dir_index]}")] = 1.0
    return row


def _move_token(
    *,
    action: MorpionAction,
    normalizer: _CoordinateNormalizer,
) -> list[float]:
    """Return one legal-action token."""
    # TODO: enrich MOVE tokens with explicit 5-dot/4-edge path features or
    # relation-aware encoding.
    dir_index, _x0, _y0, missing_index = action
    point = _missing_point_from_action(action)
    row = _blank_token(MorpionEntityTokenType.MOVE)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index(f"dir_{MORPION_ENTITY_TOKEN_DIRECTIONS[dir_index]}")] = 1.0
    row[_feature_index("candidate")] = 1.0
    row[_feature_index("is_new_dot_for_move")] = 1.0
    row[_feature_index("missing_index_in_5_window")] = float(missing_index)
    row[_feature_index("path_position")] = float(missing_index) / 4.0
    return row


def _candidate_points_from_actions(
    actions: tuple[MorpionAction, ...],
) -> frozenset[Point]:
    """Return the set of new-dot positions represented by legal actions."""
    return frozenset(_missing_point_from_action(action) for action in actions)


def _action_sort_key(action: MorpionAction) -> tuple[int, int, int, int]:
    """Return a stable key for raw Morpion actions."""
    dir_index, x0, y0, missing_index = action
    return (int(dir_index), int(x0), int(y0), int(missing_index))


def _segment_sort_key(
    segment: Segment,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a stable key for unit segments independent of endpoint order."""
    point_a, point_b = segment
    ordered = tuple(sorted((point_a, point_b)))
    return (
        (int(ordered[0][0]), int(ordered[0][1])),
        (int(ordered[1][0]), int(ordered[1][1])),
    )


def canonical_segment(segment: Segment) -> Segment:
    """Return a unit segment with its endpoints in canonical point order."""
    point_a, point_b = segment
    ordered_a, ordered_b = sorted((point_a, point_b))
    return ordered_a, ordered_b


def _missing_point_from_action(action: MorpionAction) -> Point:
    """Return the absent point represented by one raw Morpion action."""
    dir_index, x0, y0, missing_index = action
    dx, dy = DIRECTIONS[dir_index]
    return (x0 + missing_index * dx, y0 + missing_index * dy)


def _direction_index_for_segment(segment: Segment) -> int | None:
    """Return the Morpion direction index for one unit segment, if recognized."""
    (x0, y0), (x1, y1) = segment
    delta = (abs(x1 - x0), y1 - y0)
    if delta == (1, 0):
        return 0
    if delta in ((0, 1), (0, -1)):
        return 1
    if delta == (1, 1):
        return 2
    if delta == (1, -1):
        return 3
    return None


def _normalizer_for_state(state: MorpionState) -> _CoordinateNormalizer:
    """Build a coordinate normalizer covering occupied points and used segments."""
    points = set(state.points)
    for segment in state.used_unit_segments:
        points.update(segment)
    if not points:
        return _CoordinateNormalizer(center_x=0.0, center_y=0.0, scale=1.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    center_x = (float(min(xs)) + float(max(xs))) * 0.5
    center_y = (float(min(ys)) + float(max(ys))) * 0.5
    scale = max(float(max(xs) - min(xs)), float(max(ys) - min(ys)), 1.0)
    return _CoordinateNormalizer(center_x=center_x, center_y=center_y, scale=scale)


def _feature_index(name: str) -> int:
    """Return the index of one entity-token feature name."""
    return MORPION_ENTITY_TOKEN_FEATURE_NAMES.index(name)


__all__ = [
    "MORPION_ENTITY_TOKEN_DIRECTIONS",
    "MORPION_ENTITY_TOKEN_FEATURE_DIM",
    "MORPION_ENTITY_TOKEN_FEATURE_NAMES",
    "MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION",
    "MORPION_ENTITY_TOKEN_MODEL_KIND",
    "MorpionEntityTokenConverter",
    "MorpionEntityTokenLayout",
    "MorpionEntityTokenType",
    "canonical_segment",
    "is_morpion_entity_token_model_kind",
]
