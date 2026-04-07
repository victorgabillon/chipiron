"""Integer reduction Chipiron-facing state and dynamics adapters."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import lru_cache

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


class IntegerReductionStateStepsTypeError(TypeError):
    """Raised when the state step counter is not a valid integer."""

    def __init__(self) -> None:
        """Build a consistent type validation error."""
        super().__init__("Integer reduction state steps must be an int.")


class IntegerReductionStateStepsValueError(ValueError):
    """Raised when the state step counter is outside the valid range."""

    def __init__(self) -> None:
        """Build a consistent range validation error."""
        super().__init__("Integer reduction state steps must be >= 0.")


@lru_cache(maxsize=1)
def _atomheart_state_accepts_steps() -> bool:
    """Return whether the installed atomheart state already supports ``steps``."""
    return "steps" in inspect.signature(AtomIntegerReductionState).parameters


def _build_atomheart_state(*, value: int, steps: int) -> AtomIntegerReductionState:
    """Build the atomheart state while remaining compatible with older releases."""
    if _atomheart_state_accepts_steps():
        return AtomIntegerReductionState(value=value, steps=steps)
    return AtomIntegerReductionState(value)


@dataclass(frozen=True, slots=True)
class IntegerReductionState:
    """Turn-carrying integer-reduction state for Chipiron runtime code."""

    value: int
    steps: int = 0
    turn: SoloRole = SOLO

    def __post_init__(self) -> None:
        """Validate using atomheart's canonical integer-reduction state."""
        _build_atomheart_state(value=self.value, steps=self.steps)
        if self.steps.__class__ is not int:
            raise IntegerReductionStateStepsTypeError
        if self.steps < 0:
            raise IntegerReductionStateStepsValueError

    @property
    def tag(self) -> valanga.StateTag:
        """Return a stable tag suitable for caching."""
        return (self.value, self.steps)

    def is_game_over(self) -> bool:
        """Return whether the terminal goal has been reached."""
        return self.value == 1

    def pprint(self) -> str:
        """Return a concise human-readable state string."""
        return f"n={self.value}, steps={self.steps}"

    def __str__(self) -> str:
        """Return the compact canonical state form."""
        return self.pprint()

    def to_atomheart_state(self) -> AtomIntegerReductionState:
        """Convert to the atomheart integer-reduction state."""
        return _build_atomheart_state(value=self.value, steps=self.steps)

    @classmethod
    def from_atomheart_state(
        cls,
        state: AtomIntegerReductionState,
        *,
        fallback_steps: int | None = None,
    ) -> IntegerReductionState:
        """Build a Chipiron-facing state from an atomheart state."""
        steps = getattr(
            state,
            "steps",
            fallback_steps if fallback_steps is not None else 0,
        )
        return cls(value=state.value, steps=steps)


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
        next_steps = getattr(transition.next_state, "steps", state.steps + 1)
        return valanga.Transition(
            next_state=IntegerReductionState.from_atomheart_state(
                transition.next_state,
                fallback_steps=next_steps,
            ),
            modifications=transition.modifications,
            info=transition.info,
            is_over=transition.is_over,
            over_event=transition.over_event,
        )

    def action_name(
        self, state: IntegerReductionState, action: valanga.BranchKey
    ) -> str:
        """Return the canonical string name for an action key."""
        return self.inner.action_name(state.to_atomheart_state(), action)

    def action_from_name(
        self,
        state: IntegerReductionState,
        name: str,
    ) -> IntegerReductionAction:
        """Parse a canonical action name and validate it for the given state."""
        return self.inner.action_from_name(state.to_atomheart_state(), name)
