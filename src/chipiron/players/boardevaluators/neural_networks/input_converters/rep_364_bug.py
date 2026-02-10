# pylint: disable=duplicate-code
"""Module for rep 364 bug."""

from typing import Protocol, cast

import atomheart.board as boards
import chess
import torch
from valanga import ContentRepresentation

from chipiron.environments.chess.types import ChessState

from .board_representation import Representation364


class _Rep364Like(ContentRepresentation[ChessState, torch.Tensor], Protocol):
    """Protocol for tensor-backed Representation364-like objects."""

    tensor_white: torch.Tensor
    tensor_black: torch.Tensor
    tensor_castling_white: torch.Tensor
    tensor_castling_black: torch.Tensor


def create_from_state_and_modifications(
    state: ChessState,
    state_modifications: boards.BoardModificationP,
    previous_state_representation: ContentRepresentation[ChessState, torch.Tensor],
) -> Representation364:
    """Convert the node, board modifications, and parent node into a tensor representation.

    Args:
        state (ChessState): The current state in the tree.
        state_modifications (boards.BoardModificationP): The modifications made to the board.
        previous_state_representation: The previous state representation.

    Returns:
        Representation364: The tensor representation of the node, board modifications, and parent node.

    """
    prev364 = cast("_Rep364Like", previous_state_representation)

    tensor_white = torch.empty_like(prev364.tensor_white).copy_(prev364.tensor_white)
    tensor_black = torch.empty_like(prev364.tensor_black).copy_(prev364.tensor_black)

    for removal in state_modifications.removals:
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

    for appearance in state_modifications.appearances:
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

    board = state.board
    tensor_castling_white[0] = bool(board.castling_rights & chess.BB_A1)
    tensor_castling_white[1] = bool(board.castling_rights & chess.BB_H1)
    tensor_castling_black[0] = bool(board.castling_rights & chess.BB_A8)
    tensor_castling_black[1] = bool(board.castling_rights & chess.BB_H8)

    return Representation364(
        tensor_white=tensor_white,
        tensor_black=tensor_black,
        tensor_castling_black=tensor_castling_black,
        tensor_castling_white=tensor_castling_white,
    )


def create_from_board(state: ChessState) -> Representation364:
    """Create a Representation364 object from a board.

    Args:
        state (ChessState): The chess state.

    Returns:
        Representation364: The created Representation364 object.

    """
    white: torch.Tensor
    black: torch.Tensor
    castling_white: torch.Tensor
    castling_black: torch.Tensor
    white, black, castling_black, castling_white = d()

    board: boards.IBoard = state.board
    p: dict[chess.Square, tuple[int, bool]] = e(board)
    a(black, white, p)
    b(board, castling_black, castling_white)
    return c(castling_black, castling_white, black, white)


def e(board: boards.IBoard) -> dict[chess.Square, tuple[int, bool]]:
    """E."""
    piece_map: dict[chess.Square, tuple[int, bool]] = board.piece_map()
    return piece_map


def d() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """D."""
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
    """C."""
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
    """B."""
    castling_white[0] = board.has_queenside_castling_rights(chess.WHITE)
    castling_white[1] = board.has_kingside_castling_rights(chess.WHITE)
    castling_black[0] = board.has_queenside_castling_rights(chess.BLACK)
    castling_black[1] = board.has_kingside_castling_rights(chess.BLACK)


def a(
    black: torch.Tensor,
    white: torch.Tensor,
    piece_map: dict[chess.Square, tuple[int, bool]],
) -> None:
    """Populate piece tensors."""
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
