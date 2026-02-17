"""Document the module provides a random move selector for a chess game.

The Random class in this module implements a move selector that randomly selects a legal move from the given chess board.
It uses a random number generator to make the selection.

Example usage:
    random_selector = create_random(random_generator)
    move = random_selector.select_move(board, move_seed)
"""

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from valanga import Dynamics, TurnState
from valanga.game import Seed
from valanga.policy import NotifyProgressCallable, Recommendation

from .move_selector_types import MoveSelectorTypes

if TYPE_CHECKING:
    from atomheart.move import MoveUci
    from valanga import BranchKey


class MissingRandomDynamicsError(ValueError):
    """Raised when a random selector is used without runtime dynamics."""

    DEFAULT_MESSAGE = (
        "Random move selector requires `dynamics`. "
        "Use create_random(...) to build a runtime selector instance."
    )

    def __init__(self) -> None:
        """Initialize the error with a fixed guidance message."""
        super().__init__(self.DEFAULT_MESSAGE)


@dataclass
class Random[StateT: TurnState]:
    """Random move selector class.

    This class implements a move selector that randomly selects a legal move from the given chess board.

    Attributes:
        type (Literal[MoveSelectorTypes.Random]): The type of move selector (for serialization).
        random_generator (random.Random): The random number generator used for making the selection.

    """

    type: Literal[MoveSelectorTypes.RANDOM]  # for serialization
    dynamics: Dynamics[StateT] | None = None
    random_generator: random.Random = field(default_factory=random.Random)

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Select a random move from the given chess board.

        Args:
            state (State): The current state of the chess game.
            _seed: The seed for the random number generator.
            _notify_progress: Optional callback for progress notification.

        Returns:
            Recommendation: The selected move recommendation.

        """
        _ = notify_progress  # Unused in this implementation
        if self.dynamics is None:
            raise MissingRandomDynamicsError
        self.random_generator.seed(seed)
        random_move_key: BranchKey = self.random_generator.choice(
            self.dynamics.legal_actions(state).get_all()
        )
        random_move_uci: MoveUci = self.dynamics.action_name(state, random_move_key)
        return Recommendation(recommended_name=random_move_uci, evaluation=None)


def create_random[StateT: TurnState](
    *,
    dynamics: Dynamics[StateT],
    random_generator: random.Random,
) -> Random[StateT]:
    """Create a random move selector.

    Args:
        random_generator (random.Random): The random number generator to use.

    Returns:
        Random: The created random move selector.

    """
    return Random(
        type=MoveSelectorTypes.RANDOM,
        dynamics=dynamics,
        random_generator=random_generator,
    )
