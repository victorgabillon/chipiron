from typing import Protocol, runtime_checkable

import torch

from chipiron.environments.chess.board import BoardChi


class BoardToInput(Protocol):

    def convert(self, board: BoardChi) -> torch.Tensor:
        ...


@runtime_checkable
class BoardToInputFunction(Protocol):

    def __call__(self, board: BoardChi) -> torch.Tensor:
        ...
