"""
This module provides a random move selector for a chess game.

The Random class in this module implements a move selector that randomly selects a legal move from the given chess board.
It uses a random number generator to make the selection.

Example usage:
    random_selector = create_random(random_generator)
    move = random_selector.select_move(board, move_seed)

"""

import random
from dataclasses import dataclass, field
from typing import Literal

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.utils import seed

from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes


@dataclass
class Random:
    """
    Random move selector class.

    This class implements a move selector that randomly selects a legal move from the given chess board.

    Attributes:
        type (Literal[MoveSelectorTypes.Random]): The type of move selector (for serialization).
        random_generator (random.Random): The random number generator used for making the selection.

    """

    type: Literal[MoveSelectorTypes.Random]  # for serialization
    random_generator: random.Random = field(default_factory=random.Random)

    def select_move(self, board: boards.IBoard, move_seed: seed) -> MoveRecommendation:
        """
        Selects a random move from the given chess board.

        Args:
            board (boards.BoardChi): The chess board.
            move_seed (seed): The seed for the random number generator.

        Returns:
            MoveRecommendation: The selected move recommendation.

        """
        self.random_generator.seed(move_seed)
        random_move_key: moveKey = self.random_generator.choice(
            board.legal_moves.get_all()
        )
        random_move_uci: moveUci = board.get_uci_from_move_key(move_key=random_move_key)
        return MoveRecommendation(move=random_move_uci)


def create_random(random_generator: random.Random) -> Random:
    """
    Creates a random move selector.

    Args:
        random_generator (random.Random): The random number generator to use.

    Returns:
        Random: The created random move selector.

    """
    return Random(type=MoveSelectorTypes.Random, random_generator=random_generator)
