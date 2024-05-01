"""
This module provides functions and classes for creating board representations used in neural networks.

The main classes in this module are:
- `Representation364`: Represents the board state using a tensor of size 364.
- `Representation364Factory`: Factory class for creating `Representation364` objects.

The module also includes helper functions for converting board states to tensors and vice versa.

Note: This module is part of the `chipiron` package.
"""

from typing import Any
from typing import Optional

import chess
import torch

import chipiron.environments.chess.board as board_mod
from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from .board_representation import Representation364


def node_to_tensors_pieces_square_from_parent(
        node: ITreeNode[Any],
        board_modifications: board_mod.BoardModification,
        parent_node: AlgorithmNode
) -> Representation364:
    """
    Converts the node, board modifications, and parent node into a tensor representation.

    Args:
        node (ITreeNode): The current node in the tree.
        board_modifications (board_mod.BoardModification): The modifications made to the board.
        parent_node (AlgorithmNode): The parent node of the current node.

    Returns:
        Representation364: The tensor representation of the node, board modifications, and parent node.
    """

    board_representation = parent_node.board_representation
    assert isinstance(board_representation, Representation364)
    tensor_white = torch.empty_like(board_representation.tensor_white).copy_(board_representation.tensor_white)
    tensor_black = torch.empty_like(board_representation.tensor_black).copy_(board_representation.tensor_black)

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

    board = node.board
    tensor_castling_white[0] = bool(board.board.castling_rights & chess.BB_A1)
    tensor_castling_white[1] = bool(board.board.castling_rights & chess.BB_H1)
    tensor_castling_black[0] = bool(board.board.castling_rights & chess.BB_A8)
    tensor_castling_black[1] = bool(board.board.castling_rights & chess.BB_H8)

    representation = Representation364(
        tensor_white=tensor_white,
        tensor_black=tensor_black,
        tensor_castling_black=tensor_castling_black,
        tensor_castling_white=tensor_castling_white
    )

    return representation


class Representation364Factory:
    """
    Factory class for creating instances of Representation364.

    This factory provides methods to create a Representation364 object either from a transition or from a board.
    """

    def create_from_transition(
            self,
            tree_node: TreeNode[Any],
            parent_node: AlgorithmNode | None,
            modifications: board_mod.BoardModification | None
    ) -> Representation364:
        """
        Create a Representation364 object from a transition.

        Args:
            tree_node (TreeNode): The current tree node.
            parent_node (AlgorithmNode | None): The parent node of the current tree node.
            modifications (board_mod.BoardModification | None): The board modifications.

        Returns:
            Representation364: The created Representation364 object.
        """
        """  this version is supposed to be faster as it only modifies the parent
        representation with the last move and does not scan fully the new board"""
        if parent_node is None:  # this is the root_node
            representation = self.create_from_board(board=tree_node.board)
        else:
            assert modifications is not None
            representation = node_to_tensors_pieces_square_from_parent(
                node=tree_node,
                board_modifications=modifications,
                parent_node=parent_node
            )

        return representation

    def create_from_board(
            self,
            board: BoardChi
    ) -> Representation364:
        """
        Create a Representation364 object from a board.

        Args:
            board (BoardChi): The chess board.

        Returns:
            Representation364: The created Representation364 object.
        """
        white: torch.Tensor = torch.zeros(384, dtype=torch.float)
        black: torch.Tensor = torch.zeros(384, dtype=torch.float)
        castling_white: torch.Tensor = torch.zeros(2, dtype=torch.float)
        castling_black: torch.Tensor = torch.zeros(2, dtype=torch.float)

        square: int
        for square in range(64):
            piece: Optional[chess.Piece] = board.piece_at(square)
            if piece:
                piece_code: int = piece.piece_type - 1
                square_index: chess.Square
                index: int
                if piece.color == chess.BLACK:
                    square_index = chess.square_mirror(square)
                    index = 64 * piece_code + square_index
                    black[index] = 1.0
                else:
                    square_index = square
                    index = 64 * piece_code + square_index
                    white[index] = 1.0

        castling_white[0] = board.has_queenside_castling_rights(chess.WHITE)
        castling_white[1] = board.has_kingside_castling_rights(chess.WHITE)
        castling_black[0] = board.has_queenside_castling_rights(chess.BLACK)
        castling_black[1] = board.has_kingside_castling_rights(chess.BLACK)

        representation: Representation364 = Representation364(
            tensor_white=white,
            tensor_black=black,
            tensor_castling_black=castling_black,
            tensor_castling_white=castling_white
        )

        return representation


def create_board_representation(
        board_representation_str: str
) -> Representation364Factory | None:
    """
    Create a board representation based on the given string.

    Args:
        board_representation_str (str): The string representing the board representation.

    Returns:
        Representation364Factory | None: The created board representation object, or None if the string is 'no'.

    Raises:
        Exception: If the string is not '364'.

    """
    board_representation: Representation364Factory | None
    match board_representation_str:
        case '364':
            board_representation = Representation364Factory()
        case 'no':
            board_representation = None
        case other:
            raise Exception(f'trying to create {other} in file {__name__}')

    return board_representation
