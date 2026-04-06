"""Integer reduction Chipiron-facing state and dynamics adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import valanga
from atomheart.games.integer_reduction import (
    IntegerReductionDynamics as AtomIntegerReductionDynamics,
)
from atomheart.games.integer_reduction.dynamics import (
    IntegerReductionAction as AtomIntegerReductionAction,
)
from atomheart.games.integer_reduction.state import (
    IntegerReductionState as AtomIntegerReductionState,
)
from valanga import SOLO, SoloRole

type IntegerReductionAction = AtomIntegerReductionAction


@dataclass(frozen=True, slots=True)
class IntegerReductionState:
    """Turn-carrying integer-reduction state for Chipiron runtime code."""

    value: int
    turn: SoloRole = SOLO

    def __post_init__(self) -> None:
        """Validate using atomheart's canonical integer-reduction state."""
        AtomIntegerReductionState(self.value)

    @property
    def tag(self) -> valanga.StateTag:
        """Return a stable tag suitable for caching."""
        return self.value

    def is_game_over(self) -> bool:
        """Return whether the terminal goal has been reached."""
        return self.value == 1

    def pprint(self) -> str:
        """Return a concise human-readable state string."""
        return f"n={self.value}"

    def __str__(self) -> str:
        """Return the compact canonical state form."""
        return self.pprint()

    def to_atomheart_state(self) -> AtomIntegerReductionState:
        """Convert to the atomheart integer-reduction state."""
        return AtomIntegerReductionState(self.value)

    @classmethod
    def from_atomheart_state(
        cls, state: AtomIntegerReductionState
    ) -> IntegerReductionState:
        """Build a Chipiron-facing state from an atomheart state."""
        return cls(value=state.value)


@dataclass(slots=True)
class IntegerReductionDynamics(valanga.Dynamics[IntegerReductionState]):
    """Chipiron-facing adapter over atomheart integer-reduction dynamics."""

    inner: AtomIntegerReductionDynamics = field(
        default_factory=AtomIntegerReductionDynamics
    )

    def legal_actions(
        self,
        state: IntegerReductionState,
    ) -> valanga.BranchKeyGeneratorP[IntegerReductionAction]:
        """Return legal actions for the provided state."""
        return self.inner.legal_actions(state.to_atomheart_state())

    def step(
        self,
        state: IntegerReductionState,
        action: valanga.BranchKey,
    ) -> valanga.Transition[IntegerReductionState]:
        """Apply an action and convert the resulting transition."""
        transition = self.inner.step(state.to_atomheart_state(), action)
        return valanga.Transition(
            next_state=IntegerReductionState.from_atomheart_state(
                transition.next_state
            ),
            modifications=transition.modifications,
            info=transition.info,
            is_over=transition.is_over,
            over_event=transition.over_event,
        )

    def action_name(self, state: IntegerReductionState, action: valanga.BranchKey) -> str:
        """Return the canonical string name for an action key."""
        return self.inner.action_name(state.to_atomheart_state(), action)

    def action_from_name(
        self,
        state: IntegerReductionState,
        name: str,
    ) -> IntegerReductionAction:
        """Parse a canonical action name and validate it for the given state."""
        return self.inner.action_from_name(state.to_atomheart_state(), name)
