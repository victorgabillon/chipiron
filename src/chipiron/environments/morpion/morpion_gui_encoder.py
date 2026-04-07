"""Morpion GUI encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from atomheart.games.morpion import MorpionState as AtomMorpionState
from atomheart.games.morpion.state import Point, Segment
from valanga.game import Seed

from chipiron.displays.gui_protocol import UpdatePayload, UpdGameStatus, UpdStateGeneric
from chipiron.environments.morpion.types import MorpionAction, MorpionDynamics, MorpionState
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.gui_encoder import GuiEncoder

type DisplayPoint = tuple[int, int]
type DisplaySegment = tuple[DisplayPoint, DisplayPoint]


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


def _preview_segment_from_new_segments(new_segments: frozenset[Segment]) -> DisplaySegment:
    """Collapse four new unit segments into the preview line endpoints."""
    if not new_segments:
        raise ValueError("Morpion preview segment requires at least one new segment.")

    first_start, first_end = next(iter(new_segments))
    dx = _sign(first_end[0] - first_start[0])
    dy = _sign(first_end[1] - first_start[1])
    if dx == 0 and dy == 0:
        raise ValueError("Morpion preview segment cannot be inferred from a null edge.")

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
    """Structured GUI payload for Morpion board rendering."""

    variant: str
    moves: int
    point_count: int
    points: tuple[DisplayPoint, ...]
    segments: tuple[DisplaySegment, ...]
    legal_moves: tuple[MorpionMoveDisplay, ...]
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class MorpionGuiEncoder(GuiEncoder[MorpionState]):
    """Encode Morpion state updates for the generic GUI."""

    dynamics: MorpionDynamics
    game_kind: GameKind = GameKind.MORPION

    def _build_move_display(
        self,
        *,
        state: MorpionState,
        atom_state: AtomMorpionState,
        action: MorpionAction,
    ) -> MorpionMoveDisplay:
        """Build the drawable delta for one legal Morpion move."""
        next_transition = self.dynamics.inner.step(atom_state, action)
        next_state = cast("AtomMorpionState", next_transition.next_state)

        new_points = next_state.points - atom_state.points
        if len(new_points) != 1:
            raise ValueError(
                "Morpion move preview expects exactly one new point in the delta."
            )
        new_point = _normalize_point(next(iter(new_points)))

        new_segments = next_state.used_unit_segments - atom_state.used_unit_segments
        preview_segment = _preview_segment_from_new_segments(new_segments)

        return MorpionMoveDisplay(
            action_name=self.dynamics.action_name(state, action),
            new_point=new_point,
            segment=preview_segment,
        )

    def make_state_payload(
        self,
        *,
        state: MorpionState,
        seed: Seed | None,
    ) -> UpdatePayload:
        """Create state payload."""
        atom_state = state.to_atomheart_state()
        legal_actions = tuple(
            cast("MorpionAction", action)
            for action in self.dynamics.legal_actions(state).get_all()
        )
        legal_moves = tuple(
            self._build_move_display(state=state, atom_state=atom_state, action=action)
            for action in legal_actions
        )

        return UpdStateGeneric(
            state_tag=state.tag,
            action_name_history=[],
            adapter_payload=MorpionDisplayPayload(
                variant=state.variant.value,
                moves=state.moves,
                point_count=len(state.points),
                points=tuple(sorted(_normalize_point(point) for point in state.points)),
                segments=tuple(
                    sorted(_normalize_segment(segment) for segment in state.used_unit_segments)
                ),
                legal_moves=legal_moves,
                is_terminal=state.is_game_over(),
            ),
            seed=seed,
        )

    def make_status_payload(
        self,
        *,
        status: PlayingStatus,
    ) -> UpdatePayload:
        """Create status payload."""
        return UpdGameStatus(status=status)
