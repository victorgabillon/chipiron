"""
Module to transform the board into a tensor representation.
"""

import chess
import torch

from chipiron.environments.chess.board import IBoard


def transform_board_pieces_one_side(
    board: IBoard, requires_grad_bool: bool
) -> torch.Tensor:
    """
    Transform the board pieces for one side into a tensor representation.

    Args:
        board (BoardChi): The chess board.
        requires_grad_bool (bool): Whether the tensor requires gradient.

    Returns:
        torch.Tensor: The transformed board pieces tensor.
    """
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.turn == chess.BLACK:
        color_turn = board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(5)

    transform[0] = bin(board.pawns & board.occupied_color(color_turn)).count("1") - bin(
        board.pawns & board.occupied_color(color_not_turn)
    ).count("1")
    transform[1] = bin(board.knights & board.occupied_color(color_turn)).count(
        "1"
    ) - bin(board.knights & board.occupied_color(color_not_turn)).count("1")
    transform[2] = bin(board.bishops & board.occupied_color(color_turn)).count(
        "1"
    ) - bin(board.bishops & board.occupied_color(color_not_turn)).count("1")
    transform[3] = bin(board.rooks & board.occupied_color(color_turn)).count("1") - bin(
        board.rooks & board.occupied_color(color_not_turn)
    ).count("1")
    transform[4] = bin(board.queens & board.occupied_color(color_turn)).count(
        "1"
    ) - bin(board.queens & board.occupied_color(color_not_turn)).count("1")

    if requires_grad_bool:
        transform.requires_grad_(True)

    return transform


def transform_board_pieces_two_sides(
    board: IBoard, requires_grad_bool: bool
) -> torch.Tensor:
    """
    Transform the board pieces for both sides into a tensor representation.

    Args:
        board (BoardChi): The chess board.
        requires_grad_bool (bool): Whether the tensor requires gradient.

    Returns:
        torch.Tensor: The transformed board pieces tensor.
    """
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.turn == chess.BLACK:
        color_turn = board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(10, requires_grad=requires_grad_bool)

    transform[0] = bin(board.pawns & board.occupied_color(color_turn)).count("1")
    transform[1] = bin(board.knights & board.occupied_color(color_turn)).count("1")
    transform[2] = bin(board.bishops & board.occupied_color(color_turn)).count("1")
    transform[3] = bin(board.rooks & board.occupied_color(color_turn)).count("1")
    transform[4] = bin(board.queens & board.occupied_color(color_turn)).count("1")
    transform[5] = -bin(board.pawns & board.occupied_color(color_not_turn)).count("1")
    transform[6] = -bin(board.knights & board.occupied_color(color_not_turn)).count("1")
    transform[7] = -bin(board.bishops & board.occupied_color(color_not_turn)).count("1")
    transform[8] = -bin(board.rooks & board.occupied_color(color_not_turn)).count("1")
    transform[9] = -bin(board.queens & board.occupied_color(color_not_turn)).count("1")
    return transform


def get_tensor_from_tensors(
    tensor_white: torch.Tensor,
    tensor_black: torch.Tensor,
    tensor_castling_white: torch.Tensor,
    tensor_castling_black: torch.Tensor,
    color_to_play: chess.Color,
) -> torch.Tensor:
    """
    Get the final tensor representation from individual tensors.

    Args:
        tensor_white (torch.Tensor): The tensor representation for white pieces.
        tensor_black (torch.Tensor): The tensor representation for black pieces.
        tensor_castling_white (torch.Tensor): The tensor representation for white castling.
        tensor_castling_black (torch.Tensor): The tensor representation for black castling.
        color_to_play (chess.Color): The color to play.

    Returns:
        torch.Tensor: The final tensor representation.
    """
    if color_to_play == chess.WHITE:
        tensor = tensor_white - tensor_black
    else:
        tensor = tensor_black - tensor_white

    if color_to_play == chess.WHITE:
        tensor_castling = tensor_castling_white - tensor_castling_black
    else:
        tensor_castling = tensor_castling_black - tensor_castling_white

    tensor_2 = torch.cat((tensor, tensor_castling), 0)
    return tensor_2
