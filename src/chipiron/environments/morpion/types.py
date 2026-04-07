"""Morpion Chipiron-facing state and dynamics adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import valanga
from atomheart.games.morpion import MorpionDynamics as AtomMorpionDynamics
from atomheart.games.morpion import MorpionState as AtomMorpionState
from atomheart.games.morpion.dynamics import Action as AtomMorpionAction
from atomheart.games.morpion.state import Point, Segment, Variant as MorpionVariant
from valanga import SOLO, SoloRole

type MorpionAction = AtomMorpionAction
type DirUsageEntry = tuple[tuple[Point, int], int]


@dataclass(frozen=True, slots=True)
class MorpionState:
    """Turn-carrying Morpion state for Chipiron runtime code."""

    points: frozenset[Point]
    used_unit_segments: frozenset[Segment]
    dir_usage_entries: tuple[DirUsageEntry, ...]
    moves: int = 0
    variant: MorpionVariant = MorpionVariant.TOUCHING_5T
    turn: SoloRole = SOLO
    is_terminal: bool = False

    @property
    def tag(self) -> valanga.StateTag:
        """Return a stable tag suitable for caching."""
        return hash(
            (
                self.variant,
                self.moves,
                tuple(sorted(self.points)),
                tuple(sorted(self.used_unit_segments)),
                self.dir_usage_entries,
            )
        )

    @property
    def dir_usage(self) -> dict[tuple[Point, int], int]:
        """Return same-direction usage data as a fresh mapping."""
        return dict(self.dir_usage_entries)

    def is_game_over(self) -> bool:
        """Return whether no legal moves remain."""
        return self.is_terminal

    def pprint(self) -> str:
        """Return the underlying Morpion board text."""
        return self.to_atomheart_state().pprint()

    def __str__(self) -> str:
        """Return a concise summary of the current Morpion position."""
        return (
            f"variant={self.variant.value}, moves={self.moves}, "
            f"points={len(self.points)}"
        )

    def to_atomheart_state(self) -> AtomMorpionState:
        """Convert to the atomheart Morpion state."""
        return AtomMorpionState(
            points=self.points,
            used_unit_segments=self.used_unit_segments,
            dir_usage=self.dir_usage,
            moves=self.moves,
            variant=self.variant,
        )

    @classmethod
    def from_atomheart_state(
        cls,
        state: AtomMorpionState,
        *,
        is_terminal: bool = False,
    ) -> MorpionState:
        """Build a Chipiron-facing state from an atomheart Morpion state."""
        return cls(
            points=state.points,
            used_unit_segments=state.used_unit_segments,
            dir_usage_entries=tuple(sorted(state.dir_usage.items())),
            moves=state.moves,
            variant=state.variant,
            is_terminal=is_terminal,
        )


@dataclass(slots=True)
class MorpionDynamics(valanga.Dynamics[MorpionState]):
    """Chipiron-facing adapter over atomheart Morpion dynamics."""

    inner: AtomMorpionDynamics = field(default_factory=AtomMorpionDynamics)

    def _as_atomheart_state(
        self,
        state: MorpionState | AtomMorpionState,
    ) -> AtomMorpionState:
        """Normalize a Chipiron or atomheart Morpion state to atomheart form."""
        if isinstance(state, MorpionState):
            return state.to_atomheart_state()
        return state

    def legal_action_count(
        self,
        state: MorpionState | AtomMorpionState,
    ) -> int:
        """Return the current number of legal Morpion moves."""
        return len(self.inner.legal_actions(self._as_atomheart_state(state)).get_all())

    def is_terminal_state(
        self,
        state: MorpionState | AtomMorpionState,
    ) -> bool:
        """Return whether the provided Morpion state has no legal action."""
        return self.legal_action_count(state) == 0

    def wrap_atomheart_state(
        self,
        state: AtomMorpionState,
        *,
        is_terminal: bool | None = None,
    ) -> MorpionState:
        """Build a Chipiron Morpion state with terminality kept consistent."""
        terminal = self.is_terminal_state(state) if is_terminal is None else is_terminal
        return MorpionState.from_atomheart_state(state, is_terminal=terminal)

    def legal_actions(
        self,
        state: MorpionState,
    ) -> valanga.BranchKeyGeneratorP[MorpionAction]:
        """Return legal actions for the provided state."""
        return self.inner.legal_actions(self._as_atomheart_state(state))

    def step(
        self,
        state: MorpionState,
        action: valanga.BranchKey,
    ) -> valanga.Transition[MorpionState]:
        """Apply an action and convert the resulting transition."""
        transition = self.inner.step(self._as_atomheart_state(state), action)
        return valanga.Transition(
            next_state=self.wrap_atomheart_state(
                cast("AtomMorpionState", transition.next_state),
                is_terminal=transition.is_over,
            ),
            modifications=transition.modifications,
            info=transition.info,
            is_over=transition.is_over,
            over_event=transition.over_event,
        )

    def action_name(self, state: MorpionState, action: valanga.BranchKey) -> str:
        """Return the canonical string name for an action key."""
        return self.inner.action_name(self._as_atomheart_state(state), action)

    def action_from_name(
        self,
        state: MorpionState,
        name: str,
    ) -> MorpionAction:
        """Parse a canonical action name and validate it for the given state."""
        return cast(
            "MorpionAction",
            self.inner.action_from_name(self._as_atomheart_state(state), name),
        )
