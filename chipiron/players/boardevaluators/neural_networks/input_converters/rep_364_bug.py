import chess
import torch

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board.iboard import IBoard

from .board_representation import Representation364


def create_from_board_and_from_parent(
    board: IBoard,
    board_modifications: boards.BoardModificationP,
    parent_node_board_representation: Representation364,
) -> Representation364:
    """
    Converts the node, board modifications, and parent node into a tensor representation.

    Args:
        board (IBoard): The current board in the tree.
        board_modifications (board_mod.BoardModification): The modifications made to the board.
        parent_node (AlgorithmNode): The parent node of the current node.

    Returns:
        Representation364: The tensor representation of the node, board modifications, and parent node.
    """

    assert isinstance(parent_node_board_representation, Representation364)
    tensor_white = torch.empty_like(
        parent_node_board_representation.tensor_white
    ).copy_(parent_node_board_representation.tensor_white)
    tensor_black = torch.empty_like(
        parent_node_board_representation.tensor_black
    ).copy_(parent_node_board_representation.tensor_black)

    for removal in board_modifications.removals:
        piece_type = removal.piece
        piece_color = removal.color
        square = removal.square
        piece_code = piece_type - 1
        if piece_color == chess.BLACK:
            square_index = chess.square_mirror(square)
            index = 64 * piece_code + square_index
            tensor_black[index] = 0
        else:
            square_index = square
            index = 64 * piece_code + square_index
            tensor_white[index] = 0

    for appearance in board_modifications.appearances:
        # print('app',appearance)
        piece_type = appearance.piece
        piece_color = appearance.color
        square = appearance.square
        piece_code = piece_type - 1
        if piece_color == chess.BLACK:
            square_index = chess.square_mirror(square)
            index = 64 * piece_code + square_index
            tensor_black[index] = 1
        else:
            square_index = square
            index = 64 * piece_code + square_index
            tensor_white[index] = 1

    tensor_castling_white = torch.zeros(2, requires_grad=False, dtype=torch.float)
    tensor_castling_black = torch.zeros(2, requires_grad=False, dtype=torch.float)

    tensor_castling_white[0] = bool(board.castling_rights & chess.BB_A1)
    tensor_castling_white[1] = bool(board.castling_rights & chess.BB_H1)
    tensor_castling_black[0] = bool(board.castling_rights & chess.BB_A8)
    tensor_castling_black[1] = bool(board.castling_rights & chess.BB_H8)

    representation = Representation364(
        tensor_white=tensor_white,
        tensor_black=tensor_black,
        tensor_castling_black=tensor_castling_black,
        tensor_castling_white=tensor_castling_white,
    )

    return representation


def create_from_board(board: boards.IBoard) -> Representation364:
    """
    Create a Representation364 object from a board.

    Args:
        board (BoardChi): The chess board.

    Returns:
        Representation364: The created Representation364 object.
    """

    white: torch.Tensor
    black: torch.Tensor
    castling_white: torch.Tensor
    castling_black: torch.Tensor
    white, black, castling_black, castling_white = d()

    p: dict[chess.Square, tuple[int, bool]] = e(board)
    a(black, white, p)
    b(board, castling_black, castling_white)
    representation = c(castling_black, castling_white, black, white)

    return representation


def e(board: boards.IBoard) -> dict[chess.Square, tuple[int, bool]]:
    piece_map: dict[chess.Square, tuple[int, bool]] = board.piece_map()
    return piece_map


def d() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    white: torch.Tensor = torch.zeros(384, dtype=torch.float)
    black: torch.Tensor = torch.zeros(384, dtype=torch.float)
    castling_white: torch.Tensor = torch.zeros(2, dtype=torch.float)
    castling_black: torch.Tensor = torch.zeros(2, dtype=torch.float)
    return white, black, castling_black, castling_white


def c(
    castling_black: torch.Tensor,
    castling_white: torch.Tensor,
    black: torch.Tensor,
    white: torch.Tensor,
) -> Representation364:
    representation: Representation364 = Representation364(
        tensor_white=white,
        tensor_black=black,
        tensor_castling_black=castling_black,
        tensor_castling_white=castling_white,
    )
    return representation


def b(
    board: boards.IBoard, castling_black: torch.Tensor, castling_white: torch.Tensor
) -> None:
    castling_white[0] = board.has_queenside_castling_rights(chess.WHITE)
    castling_white[1] = board.has_kingside_castling_rights(chess.WHITE)
    castling_black[0] = board.has_queenside_castling_rights(chess.BLACK)
    castling_black[1] = board.has_kingside_castling_rights(chess.BLACK)


def a(
    black: torch.Tensor,
    white: torch.Tensor,
    piece_map: dict[chess.Square, tuple[int, bool]],
) -> None:
    square: chess.Square
    piece: tuple[int, bool]
    for square, piece in piece_map.items():
        piece_code: int = piece[0] - 1
        piece_color: bool = piece[1]
        square_index: chess.Square
        index: int
        if piece_color == chess.BLACK:
            square_index = chess.square_mirror(square)
            index = 64 * piece_code + square_index
            black[index] = 1.0
        else:
            square_index = square
            index = 64 * piece_code + square_index
            white[index] = 1.0
