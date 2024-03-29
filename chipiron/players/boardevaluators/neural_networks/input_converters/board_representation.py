from dataclasses import dataclass
from typing import Protocol

import chess
import torch


class BoardRepresentation(Protocol):

    def get_evaluator_input(
            self,
            color_to_play: chess.Color
    ) -> torch.Tensor:
        ...


@dataclass(slots=True)
class Representation364:
    tensor_white: torch.Tensor
    tensor_black: torch.Tensor
    tensor_castling_white: torch.Tensor
    tensor_castling_black: torch.Tensor

    def get_evaluator_input(
            self,
            color_to_play: chess.Color
    ) -> torch.Tensor:

        if color_to_play == chess.WHITE:
            tensor = torch.cat((self.tensor_white, self.tensor_black), 0)
        else:
            tensor = torch.cat((self.tensor_black, self.tensor_white), 0)

        if color_to_play == chess.WHITE:
            tensor_castling = torch.cat((self.tensor_castling_white, self.tensor_castling_black), 0)
        else:
            tensor_castling = torch.cat((self.tensor_castling_black, self.tensor_castling_white), 0)

        tensor_2 = torch.cat((tensor, tensor_castling), 0)

        return tensor_2
