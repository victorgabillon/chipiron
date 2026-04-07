"""Morpion adapter."""

from __future__ import annotations

from typing import cast

from atomheart.games.morpion import MorpionState as AtomMorpionState
from valanga.game import BranchName, Seed
from valanga.policy import BranchSelector, NotifyProgressCallable, Recommendation

from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.players.player import PlayerMoveSelectionError


class MorpionAdapter:
    """Morpion-specific adapter used by the game-agnostic `Player`."""

    def __init__(
        self,
        *,
        dynamics: MorpionDynamics,
        main_move_selector: BranchSelector[MorpionState],
    ) -> None:
        """Initialize the instance."""
        self._dynamics = dynamics
        self._main = main_move_selector

    def build_runtime_state(
        self,
        snapshot: MorpionState | AtomMorpionState,
    ) -> MorpionState:
        """Build runtime state."""
        if isinstance(snapshot, MorpionState):
            return snapshot
        return MorpionState.from_atomheart_state(
            cast("AtomMorpionState", snapshot),
            is_terminal=False,
        )

    def legal_action_count(self, runtime_state: MorpionState) -> int:
        """Legal action count."""
        return len(self._dynamics.legal_actions(runtime_state).get_all())

    def only_action_name(self, runtime_state: MorpionState) -> BranchName:
        """Only action name."""
        actions = self._dynamics.legal_actions(runtime_state).get_all()
        if len(actions) != 1:
            raise PlayerMoveSelectionError
        return self._dynamics.action_name(runtime_state, actions[0])

    def oracle_action_name(  # pylint: disable=useless-return
        self,
        runtime_state: MorpionState,
    ) -> BranchName | None:
        """Oracle action name."""
        del runtime_state
        return None

    def recommend(
        self,
        runtime_state: MorpionState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend."""
        return self._main.recommend(
            state=runtime_state,
            seed=seed,
            notify_progress=notify_progress,
        )
