import chess

from chipiron.environments.chess.board.iboard import PieceMap
from chipiron.players.boardevaluators.neural_networks.models.tranformer_one import (
    TransformerArgs,
)
import torch


def build_transformer_input(
    piece_map: PieceMap, transformer_args: TransformerArgs
) -> torch.Tensor:
    # initialing to the empty square and the output embedding
    indices: list[int] = [
        square * transformer_args.number_occupancy_types for square in range(64)
    ] + [transformer_args.len_square_tensor]
    square: chess.Square
    piece: tuple[chess.PieceType, chess.Color]
    for square, piece in piece_map.items():
        piece_type = piece[0]
        color = piece[1]

        indices[square] = (
            square * transformer_args.number_occupancy_types
            + 1  # +1 for the empty square
            + color * transformer_args.number_pieces_types
            + piece_type
        )
    return torch.tensor(indices).int()
