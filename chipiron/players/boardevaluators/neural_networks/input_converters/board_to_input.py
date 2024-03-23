from typing import Protocol

import torch

from chipiron.environments.chess.board import BoardChi


class BoardToInput(Protocol):

    def convert(self, board: BoardChi) -> torch.Tensor:
        ...
