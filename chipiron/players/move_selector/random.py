import chipiron.environments.chess.board as boards
from .move_selector import MoveRecommendation
import chess
from typing import Literal
from dataclasses import dataclass, field
import random

Random_Name_Literal: str = 'Random'


@dataclass
class Random:
    type: Literal[Random_Name_Literal]  # for serialization
    random_generator: random.Random = field(default_factory=random.Random)

    def select_move(
            self, board: boards.BoardChi
    ) -> MoveRecommendation:
        random_move: chess.Move = self.random_generator.choice(list(board.legal_moves))
        return MoveRecommendation(move=random_move)


def create_random(
        random_generator: random.Random
) -> Random:
    return Random(
        type=Random_Name_Literal,
        random_generator=random_generator
    )
