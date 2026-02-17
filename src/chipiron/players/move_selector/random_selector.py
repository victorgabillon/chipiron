"""Runtime random move selector implementation."""

import random as py_random

from valanga import Dynamics, TurnState
from valanga.game import Seed
from valanga.policy import NotifyProgressCallable, Recommendation


class RandomSelector[StateT: TurnState]:
    """Runtime selector that samples uniformly from legal actions."""

    def __init__(
        self,
        *,
        dynamics: Dynamics[StateT],
        rng: py_random.Random,
    ) -> None:
        self._dynamics = dynamics
        self._rng = rng

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Recommend one random legal move."""
        _ = notify_progress
        self._rng.seed(seed)
        action = self._rng.choice(self._dynamics.legal_actions(state).get_all())
        return Recommendation(
            recommended_name=self._dynamics.action_name(state, action),
            evaluation=None,
        )


def create_random_selector[StateT: TurnState](
    *,
    dynamics: Dynamics[StateT],
    rng: py_random.Random,
) -> RandomSelector[StateT]:
    """Build a runtime random selector from dependencies."""
    return RandomSelector(dynamics=dynamics, rng=rng)
