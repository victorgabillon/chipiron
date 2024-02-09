from typing import Protocol
import chipiron.environments.chess.board as boards
import chess
from dataclasses import dataclass
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
