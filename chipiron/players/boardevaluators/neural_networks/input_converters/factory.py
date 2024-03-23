from typing import Optional

import chess
import torch

import chipiron.environments.chess.board as board_mod
from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from .board_representation import Representation364


def node_to_tensors_pieces_square_from_parent(node, board_modifications, parent_node):
    tensor_white = torch.empty_like(parent_node.board_representation.tensor_white).copy_(
        parent_node.board_representation.tensor_white)
    tensor_black = torch.empty_like(parent_node.board_representation.tensor_black).copy_(
        parent_node.board_representation.tensor_black)

    for removal in board_modifications.removals:
        piece_type = removal[1]
        piece_color = removal[2]
        square = removal[0]
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
        piece_type = appearance[1]
        piece_color = appearance[2]
        square = appearance[0]
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
    def create_from_transition(
            self,
            tree_node: TreeNode,
            parent_node: AlgorithmNode | None,
            modifications: board_mod.BoardModification | None
    ) -> Representation364:

        """  this version is supposed to be faster as it only modifies the parent
        representation with the last move and does not scan fully the new board"""
        if parent_node is None:  # this is the root_node
            representation = self.create_from_board(board=tree_node.board)
        else:
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
    board_representation: Representation364Factory | None
    match board_representation_str:
        case '364':
            board_representation = Representation364Factory()
        case 'no':
            board_representation = None
        case other:
            raise Exception(f'trying to create {other} in file {__name__}')

    return board_representation
