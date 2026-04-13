"""Pure Morpion display payload helpers shared by GUI and dashboard renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chipiron.environments.morpion.types import (
    MorpionAction,
    MorpionDynamics,
    MorpionState,
)

if TYPE_CHECKING:
    from atomheart.games.morpion import MorpionState as AtomMorpionState
    from atomheart.games.morpion.state import Point, Segment

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

    @property
    def legal_moves(self) -> tuple[MorpionMoveDisplay, ...]:
        """Return the default unique legal-move previews."""
        return self.unique_legal_moves


def build_morpion_display_payload(
    *,
    state: MorpionState,
    dynamics: MorpionDynamics,
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
        unique_legal_moves=unique_legal_moves,
        all_legal_moves=all_legal_moves,
        is_terminal=state.is_game_over(),
    )


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


__all__ = [
    "MorpionDisplayPayload",
    "MorpionMoveDisplay",
    "MorpionPreviewPointError",
    "MorpionPreviewSegmentError",
    "build_morpion_display_payload",
]