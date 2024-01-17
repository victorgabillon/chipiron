from random import choice
import chipiron.environments.chess.board as boards
from .move_selector import MoveRecommendation
import chess
from typing import Literal
from dataclasses import dataclass

Random_Name_Literal: str = 'Random'


def create_random():
    return Random(type=Random_Name_Literal)


@dataclass
class Random:
    type: Literal[Random_Name_Literal]  # for serialization

    def select_move(
            self, board: boards.BoardChi
    ) -> MoveRecommendation:
        random_move: chess.Move = choice(list(board.legal_moves))
        return MoveRecommendation(move=random_move)
