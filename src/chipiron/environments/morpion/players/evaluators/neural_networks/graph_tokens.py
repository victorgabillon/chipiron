"""Graph-token tensor conversion for Morpion neural evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
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
    from atomheart.games.morpion.state import Point, Segment

MORPION_GRAPH_MODEL_KIND: Final[str] = "graph_transformer"
MORPION_GRAPH_INPUT_REPRESENTATION: Final[str] = "graph_tokens_v1"


class MorpionGraphTokenType(IntEnum):
    """Token types used by the first Morpion graph-token representation."""

    VALUE = 0
    GLOBAL = 1
    DOT = 2
    EDGE = 3
    MOVE = 4


MORPION_GRAPH_DIRECTIONS: Final[tuple[str, ...]] = (
    "horizontal",
    "vertical",
    "diag_up",
    "diag_down",
)

MORPION_GRAPH_TOKEN_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "type_value",
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
MORPION_GRAPH_TOKEN_FEATURE_DIM: Final[int] = len(MORPION_GRAPH_TOKEN_FEATURE_NAMES)

_VALIDITY_INDEX: Final[int] = MORPION_GRAPH_TOKEN_FEATURE_DIM - 1


class InvalidMorpionGraphTokenConverterError(ValueError):
    """Raised when graph-token conversion is configured invalidly."""

    @classmethod
    def invalid_max_tokens(cls) -> InvalidMorpionGraphTokenConverterError:
        """Return the invalid max-token-count error."""
        return cls("Morpion graph token conversion requires max_tokens >= 2.")


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
class MorpionGraphTokenConverter:
    """Convert Morpion states into variable-length graph-token tensors."""

    dynamics: MorpionDynamics = field(default_factory=MorpionDynamics)
    max_tokens: int = 1536

    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered graph-token feature names."""
        return MORPION_GRAPH_TOKEN_FEATURE_NAMES

    @property
    def input_dim(self) -> int:
        """Return the graph-token feature width."""
        return MORPION_GRAPH_TOKEN_FEATURE_DIM

    def state_to_tensor(self, state: MorpionState) -> Tensor:
        """Return a ``T x F`` float32 tensor of real graph tokens for ``state``."""
        if self.max_tokens < 2:
            raise InvalidMorpionGraphTokenConverterError.invalid_max_tokens()

        normalizer = _normalizer_for_state(state)
        actions = self.dynamics.all_legal_actions(state)
        candidate_points = _candidate_points_from_actions(actions)
        rows: list[list[float]] = [
            _value_token(),
            _global_token(state=state, legal_action_count=len(actions)),
        ]
        rows.extend(
            _dot_token(
                point=point,
                state=state,
                candidate=point in candidate_points,
                normalizer=normalizer,
            )
            for point in sorted(state.points)
        )
        rows.extend(
            _dot_token(
                point=point,
                state=state,
                candidate=True,
                normalizer=normalizer,
            )
            for point in sorted(candidate_points - state.points)
        )
        rows.extend(
            _edge_token(segment=segment, normalizer=normalizer)
            for segment in sorted(state.used_unit_segments, key=_segment_sort_key)
        )
        rows.extend(
            _move_token(action=action, normalizer=normalizer)
            for action in sorted(actions, key=_action_sort_key)
        )

        # TODO: replace deterministic tail truncation with priority-aware truncation.
        if len(rows) > self.max_tokens:
            rows = rows[: self.max_tokens]
        return torch.tensor(rows, dtype=torch.float32)


def _blank_token(token_type: MorpionGraphTokenType) -> list[float]:
    """Return one zero-filled token row with type and validity set."""
    row = [0.0] * MORPION_GRAPH_TOKEN_FEATURE_DIM
    row[token_type.value] = 1.0
    row[_VALIDITY_INDEX] = 1.0
    return row


def _value_token() -> list[float]:
    """Return the value token consumed by value-token pooling."""
    return _blank_token(MorpionGraphTokenType.VALUE)


def _global_token(*, state: MorpionState, legal_action_count: int) -> list[float]:
    """Return one global-context token."""
    row = _blank_token(MorpionGraphTokenType.GLOBAL)
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
    row = _blank_token(MorpionGraphTokenType.DOT)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index("occupied")] = 1.0 if point in state.points else 0.0
    row[_feature_index("candidate")] = 1.0 if candidate else 0.0
    dir_usage = state.dir_usage
    for dir_index, name in enumerate(MORPION_GRAPH_DIRECTIONS):
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
    row = _blank_token(MorpionGraphTokenType.EDGE)
    x_rel, y_rel = normalizer.segment_center(segment)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index("drawn")] = 1.0
    dir_index = _direction_index_for_segment(segment)
    if dir_index is not None:
        row[_feature_index(f"dir_{MORPION_GRAPH_DIRECTIONS[dir_index]}")] = 1.0
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
    row = _blank_token(MorpionGraphTokenType.MOVE)
    x_rel, y_rel = normalizer.point(point)
    row[_feature_index("x_rel")] = x_rel
    row[_feature_index("y_rel")] = y_rel
    row[_feature_index(f"dir_{MORPION_GRAPH_DIRECTIONS[dir_index]}")] = 1.0
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
    """Return the index of one graph-token feature name."""
    return MORPION_GRAPH_TOKEN_FEATURE_NAMES.index(name)


__all__ = [
    "MORPION_GRAPH_DIRECTIONS",
    "MORPION_GRAPH_INPUT_REPRESENTATION",
    "MORPION_GRAPH_MODEL_KIND",
    "MORPION_GRAPH_TOKEN_FEATURE_DIM",
    "MORPION_GRAPH_TOKEN_FEATURE_NAMES",
    "MorpionGraphTokenConverter",
    "MorpionGraphTokenType",
]
