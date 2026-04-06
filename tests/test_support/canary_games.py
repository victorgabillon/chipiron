"""Tiny deterministic canary games for future orchestration refactors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type CanaryAction = Literal["advance"]


@dataclass(frozen=True, slots=True)
class CanaryState:
    """Tiny immutable state carrying only the actor cycle and progress."""

    active_role: str | None
    remaining_steps: int
    step_index: int = 0
    history: tuple[str, ...] = ()

    def is_terminal(self) -> bool:
        """Return whether the tiny game has reached its deterministic end."""
        return self.remaining_steps == 0


class RoleCycleCanaryGame:
    """Minimal game-like fixture with a fixed actor cycle and one action."""

    def __init__(self, *, roles: tuple[str, ...], total_steps: int) -> None:
        self.roles = roles
        self.total_steps = total_steps

    @property
    def initial_state(self) -> CanaryState:
        """Return the deterministic initial state."""
        return CanaryState(
            active_role=self.roles[0],
            remaining_steps=self.total_steps,
        )

    def current_actor(self, state: CanaryState) -> str | None:
        """Return the role expected to act now."""
        return state.active_role

    def legal_actions(self, state: CanaryState) -> tuple[CanaryAction, ...]:
        """Return the closed action set for the current state."""
        if state.is_terminal():
            return ()
        return ("advance",)

    def step(self, state: CanaryState, action: CanaryAction) -> CanaryState:
        """Advance the deterministic actor cycle by one step."""
        if action != "advance":
            raise ValueError(f"Unsupported canary action: {action!r}")
        if state.is_terminal():
            raise ValueError("Cannot advance a terminal canary state.")

        actor = self.current_actor(state)
        next_remaining = state.remaining_steps - 1
        next_step_index = state.step_index + 1
        next_role = None
        if next_remaining > 0:
            next_role = self.roles[next_step_index % len(self.roles)]

        return CanaryState(
            active_role=next_role,
            remaining_steps=next_remaining,
            step_index=next_step_index,
            history=state.history + (() if actor is None else (actor,)),
        )


def make_solo_canary_game(*, total_steps: int = 3) -> RoleCycleCanaryGame:
    """Build the one-role deterministic canary game."""
    return RoleCycleCanaryGame(roles=("solo",), total_steps=total_steps)


def make_two_role_canary_game(*, total_steps: int = 4) -> RoleCycleCanaryGame:
    """Build the two-role alternating deterministic canary game."""
    return RoleCycleCanaryGame(roles=("white", "black"), total_steps=total_steps)


def make_three_role_canary_game(*, total_steps: int = 4) -> RoleCycleCanaryGame:
    """Build the three-role cyclic deterministic canary game."""
    return RoleCycleCanaryGame(
        roles=("alpha", "beta", "gamma"), total_steps=total_steps
    )
