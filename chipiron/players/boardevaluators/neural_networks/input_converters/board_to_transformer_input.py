"""Module for board to transformer input."""
import chess
import torch
from atomheart.board import create_board
from atomheart.board.iboard import PieceMap
from atomheart.board.utils import FenPlusHistory, square_rotate
from atomheart.utils.color import valanga_color_to_chess
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
)
from valanga import Color


def build_transformer_input(
    piece_map: PieceMap, board_turn: Color, transformer_args: TransformerArgs
) -> torch.Tensor:
    """Build the transformer input tensor from the piece map and board turn.

    Args:
        piece_map (PieceMap): The piece map of the board.
        board_turn (Color): The color of the player to move.
        transformer_args (TransformerArgs): The transformer arguments.

    Returns:
        torch.Tensor: The transformer input tensor.

    """
    board_turn_chess = valanga_color_to_chess(board_turn)

    # initialing to the empty square and the output embedding
    indices: list[int] = [
        square * transformer_args.number_occupancy_types for square in range(64)
    ] + [transformer_args.len_square_tensor]
    square: chess.Square
    piece: tuple[chess.PieceType, chess.Color]
    for square, piece in piece_map.items():
        piece_type = piece[0] - 1  # pieces are from 1 to 6 in chess and we want 0 to 5
        color = piece[1]

        color = color != board_turn_chess

        if board_turn_chess == chess.BLACK:
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
