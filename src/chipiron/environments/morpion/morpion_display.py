"""Pure Morpion display payload helpers shared by GUI and dashboard renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomheart.games.morpion import (
    MorpionDynamics as AtomMorpionDynamics,
)
from atomheart.games.morpion import (
    Move as AtomMorpionMove,
)
from atomheart.games.morpion import (
    Variant as AtomMorpionVariant,
)
from atomheart.games.morpion import (
    action_to_played_move,
)
from atomheart.games.morpion import (
    initial_state as initial_atom_morpion_state,
)

if TYPE_CHECKING:
    from atomheart.games.morpion import MorpionState as AtomMorpionState
    from atomheart.games.morpion.state import Point, Segment

    from chipiron.environments.morpion.types import (
        MorpionAction,
        MorpionDynamics,
        MorpionState,
    )

type DisplayPoint = tuple[int, int]
type DisplaySegment = tuple[DisplayPoint, DisplayPoint]


class MorpionPreviewSegmentError(ValueError):
    """Raised when a Morpion preview segment cannot be reconstructed."""

    @classmethod
    def empty_delta(cls) -> MorpionPreviewSegmentError:
        """Build the empty-preview error."""
        return cls("Morpion preview segment requires at least one new segment.")

    @classmethod
    def null_edge(cls) -> MorpionPreviewSegmentError:
        """Build the degenerate-preview error."""
        return cls("Morpion preview segment cannot be inferred from a null edge.")


class MorpionPreviewPointError(ValueError):
    """Raised when a Morpion preview delta does not expose exactly one new point."""

    def __init__(self) -> None:
        """Build the invalid preview-point error."""
        super().__init__(
            "Morpion move preview expects exactly one new point in the delta."
        )


class MorpionNumberedPointReplayError(ValueError):
    """Raised when numbered added points cannot be replayed from a payload."""

    @classmethod
    def payload_must_be_mapping(cls) -> MorpionNumberedPointReplayError:
        """Build the payload-shape error."""
        return cls("Morpion state_ref payload must be a mapping.")

    @classmethod
    def missing_variant(cls) -> MorpionNumberedPointReplayError:
        """Build the missing-variant error."""
        return cls("Morpion state_ref payload is missing `variant`.")

    @classmethod
    def invalid_variant(cls, raw_variant: object) -> MorpionNumberedPointReplayError:
        """Build the invalid-variant error."""
        return cls(f"Invalid Morpion variant in state_ref payload: {raw_variant!r}")

    @classmethod
    def missing_played_moves(cls) -> MorpionNumberedPointReplayError:
        """Build the missing-move-sequence error."""
        return cls("Morpion state_ref payload is missing `played_moves`.")

    @classmethod
    def played_moves_must_be_sequence(cls) -> MorpionNumberedPointReplayError:
        """Build the move-sequence-type error."""
        return cls("Morpion state_ref payload field `played_moves` must be a sequence.")

    @classmethod
    def invalid_move_payload(
        cls, index: int, payload: object
    ) -> MorpionNumberedPointReplayError:
        """Build the malformed-move error."""
        return cls(
            "Morpion state_ref payload move at index "
            f"{index} must be a four-integer sequence: {payload!r}"
        )

    @classmethod
    def illegal_move(
        cls, index: int, move: AtomMorpionMove
    ) -> MorpionNumberedPointReplayError:
        """Build the illegal-move error."""
        return cls(
            f"Morpion state_ref payload move at index {index} is not legal: {move!r}"
        )

    @classmethod
    def invalid_new_point_delta(
        cls, index: int
    ) -> MorpionNumberedPointReplayError:
        """Build the invalid-point-delta error."""
        return cls(
            "Morpion numbered-point replay expected exactly one new point for "
            f"move index {index}."
        )


def _normalize_point(point: Point) -> DisplayPoint:
    """Convert one atomheart point to a plain integer tuple."""
    return (int(point[0]), int(point[1]))


def _normalize_segment(segment: Segment) -> DisplaySegment:
    """Convert one atomheart segment to plain integer tuples."""
    start, end = segment
    return (_normalize_point(start), _normalize_point(end))


def _sign(value: int) -> int:
    """Return the sign of an integer as -1, 0, or 1."""
    return (value > 0) - (value < 0)


def _preview_segment_from_new_segments(
    new_segments: frozenset[Segment],
) -> DisplaySegment:
    """Collapse four new unit segments into the preview line endpoints."""
    if not new_segments:
        raise MorpionPreviewSegmentError.empty_delta()

    first_start, first_end = next(iter(new_segments))
    dx = _sign(first_end[0] - first_start[0])
    dy = _sign(first_end[1] - first_start[1])
    if dx == 0 and dy == 0:
        raise MorpionPreviewSegmentError.null_edge()

    line_points = {point for segment in new_segments for point in segment}
    start = min(line_points, key=lambda point: point[0] * dx + point[1] * dy)
    end = max(line_points, key=lambda point: point[0] * dx + point[1] * dy)
    return (_normalize_point(start), _normalize_point(end))


@dataclass(frozen=True, slots=True)
class MorpionMoveDisplay:
    """Drawable one-move preview for the Morpion board UI."""

    action_name: str
    new_point: DisplayPoint
    segment: DisplaySegment


@dataclass(frozen=True, slots=True)
class MorpionNumberedPointDisplay:
    """Drawable numbered point overlay for one replayed Morpion move."""

    move_index: int
    point: DisplayPoint


@dataclass(frozen=True, slots=True)
class MorpionDisplayPayload:
    """Structured Morpion board payload usable by multiple renderers."""

    variant: str
    moves: int
    point_count: int
    points: tuple[DisplayPoint, ...]
    segments: tuple[DisplaySegment, ...]
    unique_legal_moves: tuple[MorpionMoveDisplay, ...]
    all_legal_moves: tuple[MorpionMoveDisplay, ...]
    is_terminal: bool
    numbered_added_points: tuple[MorpionNumberedPointDisplay, ...] = ()

    @property
    def legal_moves(self) -> tuple[MorpionMoveDisplay, ...]:
        """Return the default unique legal-move previews."""
        return self.unique_legal_moves


def build_morpion_display_payload(
    *,
    state: MorpionState,
    dynamics: MorpionDynamics,
    state_ref_payload: Mapping[str, object] | None = None,
) -> MorpionDisplayPayload:
    """Build the canonical Morpion board payload for one state."""
    atom_state = state.to_atomheart_state()
    unique_actions = tuple(action for action in dynamics.legal_actions(state).get_all())
    all_actions = dynamics.all_legal_actions(state)
    all_legal_moves = tuple(
        _build_move_display(
            state=state,
            atom_state=atom_state,
            action=action,
            dynamics=dynamics,
        )
        for action in all_actions
    )
    move_display_by_action = dict(zip(all_actions, all_legal_moves, strict=True))
    unique_legal_moves = tuple(move_display_by_action[action] for action in unique_actions)
    return MorpionDisplayPayload(
        variant=state.variant.value,
        moves=state.moves,
        point_count=len(state.points),
        points=tuple(sorted(_normalize_point(point) for point in state.points)),
        segments=tuple(
            sorted(_normalize_segment(segment) for segment in state.used_unit_segments)
        ),
        numbered_added_points=(
            ()
            if state_ref_payload is None
            else build_numbered_added_points_from_state_ref_payload(state_ref_payload)
        ),
        unique_legal_moves=unique_legal_moves,
        all_legal_moves=all_legal_moves,
        is_terminal=state.is_game_over(),
    )


def build_numbered_added_points_from_state_ref_payload(
    state_ref_payload: Mapping[str, object],
) -> tuple[MorpionNumberedPointDisplay, ...]:
    """Replay ordered checkpoint moves and recover the newly added point for each move."""
    payload = _normalized_state_ref_payload_mapping(state_ref_payload)
    variant = _state_ref_payload_variant(payload)
    serialized_moves = _state_ref_payload_played_moves(payload)

    dynamics = AtomMorpionDynamics()
    replay_state = initial_atom_morpion_state(variant)
    numbered_points: list[MorpionNumberedPointDisplay] = []

    for move_index, serialized_move in enumerate(serialized_moves, start=1):
        target_move = _state_ref_payload_move(serialized_move, move_index - 1)
        matching_action = next(
            (
                action
                for action in dynamics.all_legal_actions(replay_state)
                if action_to_played_move(action) == target_move
            ),
            None,
        )
        if matching_action is None:
            raise MorpionNumberedPointReplayError.illegal_move(
                move_index - 1, target_move
            )
        next_state = dynamics.step(replay_state, matching_action).next_state
        new_points = next_state.points - replay_state.points
        if len(new_points) != 1:
            raise MorpionNumberedPointReplayError.invalid_new_point_delta(move_index)
        numbered_points.append(
            MorpionNumberedPointDisplay(
                move_index=move_index,
                point=_normalize_point(next(iter(new_points))),
            )
        )
        replay_state = next_state

    return tuple(numbered_points)


def _build_move_display(
    *,
    state: MorpionState,
    atom_state: AtomMorpionState,
    action: MorpionAction,
    dynamics: MorpionDynamics,
) -> MorpionMoveDisplay:
    """Build the drawable delta for one legal Morpion move."""
    next_transition = dynamics.inner.step(atom_state, action)
    next_state = next_transition.next_state

    new_points = next_state.points - atom_state.points
    if len(new_points) != 1:
        raise MorpionPreviewPointError
    new_point = _normalize_point(next(iter(new_points)))

    new_segments = next_state.used_unit_segments - atom_state.used_unit_segments
    preview_segment = _preview_segment_from_new_segments(new_segments)

    return MorpionMoveDisplay(
        action_name=dynamics.action_name(state, action),
        new_point=new_point,
        segment=preview_segment,
    )


def _normalized_state_ref_payload_mapping(
    payload: Mapping[str, object] | object,
) -> dict[str, object]:
    """Return a normalized string-keyed payload mapping."""
    if not isinstance(payload, Mapping):
        raise MorpionNumberedPointReplayError.payload_must_be_mapping()
    normalized_payload: dict[str, object] = {}
    for key_obj, value_obj in payload.items():
        if not isinstance(key_obj, str):
            raise MorpionNumberedPointReplayError.payload_must_be_mapping()
        normalized_payload[key_obj] = value_obj
    return normalized_payload


def _state_ref_payload_variant(payload: Mapping[str, object]) -> AtomMorpionVariant:
    """Return the validated Morpion variant from one checkpoint payload."""
    raw_variant = payload.get("variant")
    if raw_variant is None:
        raise MorpionNumberedPointReplayError.missing_variant()
    if not isinstance(raw_variant, str):
        raise MorpionNumberedPointReplayError.invalid_variant(raw_variant)
    try:
        return AtomMorpionVariant(raw_variant)
    except ValueError as exc:
        raise MorpionNumberedPointReplayError.invalid_variant(raw_variant) from exc


def _state_ref_payload_played_moves(
    payload: Mapping[str, object],
) -> Sequence[object]:
    """Return the raw ordered move sequence from one checkpoint payload."""
    raw_played_moves = payload.get("played_moves")
    if raw_played_moves is None:
        raise MorpionNumberedPointReplayError.missing_played_moves()
    if not isinstance(raw_played_moves, Sequence) or isinstance(
        raw_played_moves, str | bytes | bytearray
    ):
        raise MorpionNumberedPointReplayError.played_moves_must_be_sequence()
    return raw_played_moves


def _state_ref_payload_move(payload: object, index: int) -> AtomMorpionMove:
    """Return one validated move tuple from serialized checkpoint data."""
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes | bytearray):
        raise MorpionNumberedPointReplayError.invalid_move_payload(index, payload)
    values = list(payload)
    if len(values) != 4 or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise MorpionNumberedPointReplayError.invalid_move_payload(index, payload)
    x1, y1, x2, y2 = (int(value) for value in values)
    move = (x1, y1, x2, y2)
    return move if (x1, y1) <= (x2, y2) else (x2, y2, x1, y1)


__all__ = [
    "MorpionDisplayPayload",
    "MorpionMoveDisplay",
    "MorpionNumberedPointDisplay",
    "MorpionNumberedPointReplayError",
    "MorpionPreviewPointError",
    "MorpionPreviewSegmentError",
    "build_morpion_display_payload",
    "build_numbered_added_points_from_state_ref_payload",
]
