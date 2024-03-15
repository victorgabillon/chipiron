import chipiron.environments.chess.board as boards
from .move_selector import MoveRecommendation
import chess
from typing import Literal
from dataclasses import dataclass, field
import random
from chipiron.utils import seed
from .move_selector_types import MoveSelectorTypes



@dataclass
class Random:
    type: Literal[MoveSelectorTypes.Random]  # for serialization
    random_generator: random.Random = field(default_factory=random.Random)

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed: seed
    ) -> MoveRecommendation:
        self.random_generator.seed(move_seed)
        random_move: chess.Move = self.random_generator.choice(list(board.legal_moves))
        return MoveRecommendation(move=random_move)


def create_random(
        random_generator: random.Random
) -> Random:
    return Random(
        type=MoveSelectorTypes.Random,
        random_generator=random_generator
    )
