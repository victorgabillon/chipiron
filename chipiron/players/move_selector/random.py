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

from valanga.game import Seed, State
from valanga.policy import Recommendation

from .move_selector_types import MoveSelectorTypes

if TYPE_CHECKING:
    from atomheart.move import MoveUci
    from valanga import BranchKey

from valanga.policy import NotifyProgressCallable


@dataclass
class Random:
    """Random move selector class.

    This class implements a move selector that randomly selects a legal move from the given chess board.

    Attributes:
        type (Literal[MoveSelectorTypes.Random]): The type of move selector (for serialization).
        random_generator (random.Random): The random number generator used for making the selection.

    """

    type: Literal[MoveSelectorTypes.Random]  # for serialization
    random_generator: random.Random = field(default_factory=random.Random)

    def recommend(
        self,
        state: State,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Select a random move from the given chess board.

        Args:
            board (boards.IBoard): The chess board.
            move_seed (Seed): The seed for the random number generator.
            _notify_progress (NotifyProgressCallable | None): Optional callback for progress notification.

        Returns:
            Recommendation: The selected move recommendation.

        """
        self.random_generator.seed(seed)
        random_move_key: BranchKey = self.random_generator.choice(
            state.branch_keys.get_all()
        )
        random_move_uci: MoveUci = state.branch_name_from_key(key=random_move_key)
        return Recommendation(recommended_name=random_move_uci, evaluation=None)


def create_random(random_generator: random.Random) -> Random:
    """Create a random move selector.

    Args:
        random_generator (random.Random): The random number generator to use.

    Returns:
        Random: The created random move selector.

    """
    return Random(type=MoveSelectorTypes.Random, random_generator=random_generator)
