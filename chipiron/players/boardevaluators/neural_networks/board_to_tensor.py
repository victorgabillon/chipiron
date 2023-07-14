import chess
import torch

#This code is supposed to slowly be turnred into the cmasses fro board and node represenatition
def transform_board_pieces_one_side(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.turn == chess.BLACK:
        color_turn = board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(5)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.pawns & board.occupied_co[color_turn]).count('1') \
                   - bin(board.pawns & board.occupied_co[color_not_turn]).count('1')
    transform[1] = bin(board.knights & board.occupied_co[color_turn]).count('1') \
                   - bin(board.knights & board.occupied_co[color_not_turn]).count('1')
    transform[2] = bin(board.bishops & board.occupied_co[color_turn]).count('1') \
                   - bin(board.bishops & board.occupied_co[color_not_turn]).count('1')
    transform[3] = bin(board.rooks & board.occupied_co[color_turn]).count('1') \
                   - bin(board.rooks & board.occupied_co[color_not_turn]).count('1')
    transform[4] = bin(board.queens & board.occupied_co[color_turn]).count('1') \
                   - bin(board.queens & board.occupied_co[color_not_turn]).count('1')

    if requires_grad_:
        transform.requires_grad_(True)

    return transform


def transform_board_pieces_two_sides(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.turn == chess.BLACK:
        color_turn = board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(10, requires_grad=requires_grad_)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.pawns & board.occupied_co[color_turn]).count('1')
    transform[1] = bin(board.knights & board.occupied_co[color_turn]).count('1')
    transform[2] = bin(board.bishops & board.occupied_co[color_turn]).count('1')
    transform[3] = bin(board.rooks & board.occupied_co[color_turn]).count('1')
    transform[4] = bin(board.queens & board.occupied_co[color_turn]).count('1')
    transform[5] = -bin(board.pawns & board.occupied_co[color_not_turn]).count('1')
    transform[6] = -bin(board.knights & board.occupied_co[color_not_turn]).count('1')
    transform[7] = -bin(board.bishops & board.occupied_co[color_not_turn]).count('1')
    transform[8] = -bin(board.rooks & board.occupied_co[color_not_turn]).count('1')
    transform[9] = -bin(board.queens & board.occupied_co[color_not_turn]).count('1')
    return transform



def board_to_tensor_pieces_square(board, requires_grad_):
    tensor_white, tensor_black, tensor_castling_white, tensor_castling_black \
        = board_to_tensors_pieces_square(board, requires_grad_)
    side_to_move = board.turn
    tensor = get_tensor_from_tensors(tensor_white, tensor_black, tensor_castling_white, tensor_castling_black,
                                     side_to_move)
    return tensor



def get_tensor_from_tensors(tensor_white, tensor_black, tensor_castling_white, tensor_castling_black, color_to_play):
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



def transform_board_pieces_square_old(node, requires_grad_):
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    board = node.board
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.turn == chess.BLACK:
        inversion = -1

    transform = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece_type = board.piece_type_at(square)
        piece_color = board.color_at(square)
        if piece_type is not None:
            # print('p', square, piece.color, type(piece.piece_type))
            piece_code = (piece_type - 1)
            # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
            if piece_color == chess.BLACK:
                square_index = chess.square_mirror(square)
            else:
                square_index = square
            index = 64 * piece_code + square_index
            transform[index] += (2 * piece_color - 1) * inversion

        # transform[64 * piece_code + square] = 2 * piece.color - 1
    return transform


def transform_board_pieces_square_old2(node, requires_grad_):
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    board = node.board
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.turn == chess.BLACK:
        inversion = -1

    transform = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            # print('p', square, piece.color, type(piece.piece_type))
            piece_code = (piece.piece_type - 1)
            # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
            if piece.color == chess.BLACK:
                square_index = chess.square_mirror(square)
            else:
                square_index = square
            index = 64 * piece_code + square_index
            transform[index] += (2 * piece.color - 1) * inversion

        # transform[64 * piece_code + square] = 2 * piece.color - 1
    return transform
