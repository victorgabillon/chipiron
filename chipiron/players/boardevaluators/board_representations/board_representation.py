import torch
import chess
from typing import Protocol


class BoardRepresentation(Protocol):

    def get_evaluator_input(self, color_to_play):
        ...


class Representation364:
    tensor_white: torch.Tensor
    tensor_black: torch.Tensor
    tensor_castling_white: torch.Tensor
    tensor_castling_black: torch.Tensor

    def __init__(self,
                 tensor_white: torch.Tensor,
                 tensor_black: torch.Tensor,
                 tensor_castling_white: torch.Tensor,
                 tensor_castling_black: torch.Tensor
                 ) -> None:
        self.tensor_white = tensor_white
        self.tensor_black = tensor_black
        self.tensor_castling_white = tensor_castling_white
        self.tensor_castling_black = tensor_castling_black

    def get_evaluator_input(self, color_to_play: chess.Color):

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
    tensor_castling_white[0] = bool(board.castling_rights & chess.BB_A1)
    tensor_castling_white[1] = bool(board.castling_rights & chess.BB_H1)
    tensor_castling_black[0] = bool(board.castling_rights & chess.BB_A8)
    tensor_castling_black[1] = bool(board.castling_rights & chess.BB_H8)

    representation = Representation364(tensor_white=tensor_white,
                                       tensor_black=tensor_black,
                                       tensor_castling_black=tensor_castling_black,
                                       tensor_castling_white=tensor_castling_white)

    return representation


def board_to_tensors_pieces_square(board):
    tensor_white = torch.zeros(384, dtype=torch.float)
    tensor_black = torch.zeros(384, dtype=torch.float)
    tensor_castling_white = torch.zeros(2, dtype=torch.float)
    tensor_castling_black = torch.zeros(2, dtype=torch.float)

    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            piece_code = piece.piece_type - 1
            if piece.color == chess.BLACK:
                square_index = chess.square_mirror(square)
                index = 64 * piece_code + square_index
                tensor_black[index] = 1.0
            else:
                square_index = square
                index = 64 * piece_code + square_index
                tensor_white[index] = 1.0

    tensor_castling_white[0] = board.has_queenside_castling_rights(chess.WHITE)
    tensor_castling_white[1] = board.has_kingside_castling_rights(chess.WHITE)
    tensor_castling_black[0] = board.has_queenside_castling_rights(chess.BLACK)
    tensor_castling_black[1] = board.has_kingside_castling_rights(chess.BLACK)

    return tensor_white, tensor_black, tensor_castling_white, tensor_castling_black


def node_to_tensors_pieces_square(node):
    board = node.board

    tensor_white, tensor_black, tensor_castling_white, tensor_castling_black \
        = board_to_tensors_pieces_square(board)

    representation = Representation364(tensor_white=tensor_white,
                                       tensor_black=tensor_black,
                                       tensor_castling_black=tensor_castling_black,
                                       tensor_castling_white=tensor_castling_white)

    return representation


def node_to_tensors_pieces_square_fast(node, parent_node, board_modifications):
    """  this version is supposed to be faster as it only modifies the parent
    representation with the last move and does not scan fully the new board"""
    if parent_node is None:  # this is the root_node
        representation = node_to_tensors_pieces_square(node)
    else:
        representation = node_to_tensors_pieces_square_from_parent(node, board_modifications, parent_node)
    return representation


class Representation364Factory:
    def create(self,
               tree_node,
               parent_node,
               modifications
               ) -> Representation364:
        representation = node_to_tensors_pieces_square_fast(node=tree_node,
                                                            parent_node=parent_node,
                                                            board_modifications=modifications)
        return representation
