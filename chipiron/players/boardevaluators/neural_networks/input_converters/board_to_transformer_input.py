import chess
import torch

from chipiron.environments.chess.board import create_board
from chipiron.environments.chess.board.iboard import PieceMap
from chipiron.environments.chess.board.utils import FenPlusHistory, square_rotate
from chipiron.players.boardevaluators.neural_networks.models.transformer_one import (
    TransformerArgs,
)


def build_transformer_input(
    piece_map: PieceMap, board_turn: chess.Color, transformer_args: TransformerArgs
) -> torch.Tensor:
    # initialing to the empty square and the output embedding
    indices: list[int] = [
        square * transformer_args.number_occupancy_types for square in range(64)
    ] + [transformer_args.len_square_tensor]
    # print('deb indi', indices)
    square: chess.Square
    piece: tuple[chess.PieceType, chess.Color]
    for square, piece in piece_map.items():
        piece_type = piece[0] - 1  # pieces are from 1 to 6 in chess and we want 0 to 5
        color = piece[1]

        color = color != board_turn

        if board_turn == chess.BLACK:
            square = square_rotate(square)

        indices[square] = (
            square * transformer_args.number_occupancy_types
            + 1  # +1 for the empty square
            + color * transformer_args.number_pieces_types
            + piece_type
        )
    return torch.tensor(indices).int()


if __name__ == "__main__":

    board = create_board(
        fen_with_history=FenPlusHistory(
            current_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        )
    )
    a = build_transformer_input(
        piece_map=board.piece_map(),
        board_turn=board.turn,
        transformer_args=TransformerArgs(),
    )
    print("trans inouyt", a, a.shape)
