from dataclasses import dataclass
from typing import Protocol

import chess

import chipiron.environments.chess.board as boards
from chipiron.utils import seed


@dataclass
class MoveRecommendation:
    move: chess.Move
    evaluation: float | None = None


class MoveSelector(Protocol):

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed: seed

    ) -> MoveRecommendation:
        ...
