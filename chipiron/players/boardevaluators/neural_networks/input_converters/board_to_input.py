from typing import Callable
from chessenvironment.board import BoardChi
import torch
import chess
from .board_representation import Representation364
from .factory import Representation364Factory
from typing import Optional, Protocol


class BoardToInput(Protocol):

    def convert(self, board: BoardChi) -> torch.Tensor:
        ...


class Representation364BTI:

    def __init__(
            self,
            representation_factory: Representation364Factory
    ):
        self.representation_factory = representation_factory

    def convert(
            self,
            board: BoardChi
    ) -> torch.Tensor:
        representation: Representation364 = self.representation_factory.create_from_board(board=board)
        tensor: torch.Tensor = representation.get_evaluator_input(color_to_play=board.turn)
        return tensor
