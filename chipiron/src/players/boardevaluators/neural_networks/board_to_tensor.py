import chess
import torch
from chess import BB_SQUARES, PAWN, square_rank


def transform_board_pieces_one_side(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.chess_board.turn == chess.BLACK:
        color_turn = board.chess_board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(5, requires_grad=requires_grad_)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.chess_board.pawns & board.chess_board.occupied_co[color_turn]).count('1') \
                   - bin(board.chess_board.pawns & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[1] = bin(board.chess_board.knights & board.chess_board.occupied_co[color_turn]).count('1') \
                   - bin(board.chess_board.knights & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[2] = bin(board.chess_board.bishops & board.chess_board.occupied_co[color_turn]).count('1') \
                   - bin(board.chess_board.bishops & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[3] = bin(board.chess_board.rooks & board.chess_board.occupied_co[color_turn]).count('1') \
                   - bin(board.chess_board.rooks & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[4] = bin(board.chess_board.queens & board.chess_board.occupied_co[color_turn]).count('1') \
                   - bin(board.chess_board.queens & board.chess_board.occupied_co[color_not_turn]).count('1')
    return transform


def transform_board_pieces_two_sides(board, requires_grad_):
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)

    if board.chess_board.turn == chess.BLACK:
        color_turn = board.chess_board.turn
        color_not_turn = chess.WHITE
    else:
        color_turn = chess.WHITE
        color_not_turn = chess.BLACK

    transform = torch.zeros(10, requires_grad=requires_grad_)

    # print('ol', board.chessBoard)
    transform[0] = bin(board.chess_board.pawns & board.chess_board.occupied_co[color_turn]).count('1')
    transform[1] = bin(board.chess_board.knights & board.chess_board.occupied_co[color_turn]).count('1')
    transform[2] = bin(board.chess_board.bishops & board.chess_board.occupied_co[color_turn]).count('1')
    transform[3] = bin(board.chess_board.rooks & board.chess_board.occupied_co[color_turn]).count('1')
    transform[4] = bin(board.chess_board.queens & board.chess_board.occupied_co[color_turn]).count('1')
    transform[5] = -bin(board.chess_board.pawns & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[6] = -bin(board.chess_board.knights & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[7] = -bin(board.chess_board.bishops & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[8] = -bin(board.chess_board.rooks & board.chess_board.occupied_co[color_not_turn]).count('1')
    transform[9] = -bin(board.chess_board.queens & board.chess_board.occupied_co[color_not_turn]).count('1')
    return transform


def board_to_tensor_pieces_square_old(node, requires_grad_):
    board = node.board
    tensor_white = torch.zeros(384, requires_grad=requires_grad_)
    tensor_black = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece = board.chess_board.piece_at(square)
        if piece:
            piece_code = piece.piece_type - 1
            if piece.color == chess.BLACK:
                square_index = chess.square_mirror(square)
                index = 64 * piece_code + square_index
                tensor_black[index] += 1
            else:
                square_index = square
                index = 64 * piece_code + square_index
                tensor_white[index] += 1

    board.tensor_representation = (tensor_white, tensor_black)
    result = (tensor_white - tensor_black) * (2 * (node.player_to_move == chess.WHITE) - 1)

    # transform = transform_board_pieces_square_old(node, requires_grad_)
    # assert (torch.eq(transform, result).all())
    return result


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def bits2(n):
    a = [int(x) for x in bin(n)[2:]]
    a.reverse()
    return a + [0] * (64 - len(a))


def binary_board(chess_b, color):
    pawns_white = torch.tensor(chess_b.pawns & chess_b.occupied_co[color])
    # bin_p = binary(pawns_white, 64)
    bin_p = bits2(pawns_white)
    knights_white = chess_b.knights & chess_b.occupied_co[color]
    bin_n = bits2(knights_white)
    bishops_white = chess_b.bishops & chess_b.occupied_co[color]
    bin_b = bits2(bishops_white)
    rooks_white = chess_b.rooks & chess_b.occupied_co[color]
    bin_r = bits2(rooks_white)
    queens_white = chess_b.queens & chess_b.occupied_co[color]
    bin_q = bits2(queens_white)
    kings_white = chess_b.kings & chess_b.occupied_co[color]
    bin_k = bits2(kings_white)
    # print(bin_p + bin_n + bin_b + bin_r + bin_q + bin_k)
    return torch.tensor(bin_p + bin_n + bin_b + bin_r + bin_q + bin_k)


def binary_board_flipped_verti(chess_b, color):
    pawns_white = chess.flip_vertical(chess_b.pawns & chess_b.occupied_co[color])
    # bin_p = binary(pawns_white, 64)
    bin_p = bits2(pawns_white)
    knights_white = chess.flip_vertical(chess_b.knights & chess_b.occupied_co[color])
    bin_n = bits2(knights_white)
    bishops_white = chess.flip_vertical(chess_b.bishops & chess_b.occupied_co[color])
    bin_b = bits2(bishops_white)
    rooks_white = chess.flip_vertical(chess_b.rooks & chess_b.occupied_co[color])
    bin_r = bits2(rooks_white)
    queens_white = chess.flip_vertical(chess_b.queens & chess_b.occupied_co[color])
    bin_q = bits2(queens_white)
    kings_white = chess.flip_vertical(chess_b.kings & chess_b.occupied_co[color])
    bin_k = bits2(kings_white)
    # print(bin_p + bin_n + bin_b + bin_r + bin_q + bin_k)
    return torch.tensor(bin_p + bin_n + bin_b + bin_r + bin_q + bin_k)


def board_to_tensor_pieces_square_2(node, requires_grad_):
    board = node.board
    chess_b = board.chess_board

    tensor_white = binary_board(chess_b, chess.WHITE)
    tensor_black = binary_board_flipped_verti(chess_b, chess.BLACK)

    result = (tensor_white - tensor_black) * (2 * (node.player_to_move == chess.WHITE) - 1)

    # transform = transform_board_pieces_square_old(node, requires_grad_)
    # print(result)
    # print(transform)
    # assert (torch.eq(transform, result).all())

    return result.float()


def board_to_tensor_pieces_square(node, requires_grad_):
    board = node.board
    tensor_white = torch.zeros(384, requires_grad=requires_grad_)
    tensor_black = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece = board.chess_board.piece_at(square)
        if piece:
            piece_code = piece.piece_type - 1
            if piece.color == chess.BLACK:
                square_index = chess.square_mirror(square)
                index = 64 * piece_code + square_index
                tensor_black[index] += 1
            else:
                square_index = square
                index = 64 * piece_code + square_index
                tensor_white[index] += 1

    node.tensor_white = tensor_white
    node.tensor_black = tensor_black
    result = (tensor_white - tensor_black) * (2 * (node.player_to_move == chess.WHITE) - 1)

    transform = transform_board_pieces_square_old(node, requires_grad_)
    assert (torch.eq(transform, result).all())
    return result


def board_to_tensor_pieces_square_fast(node, parent_node, board_modifications, requires_grad_):
    '''  this version is supposed to be faster as it only modifies the parent
    representation with the last move and does not scan fully the new board'''
    board = node.board
    if parent_node == None:  # this is the root_node
        return board_to_tensor_pieces_square(node, requires_grad_)
    else:
        tensor_white = parent_node.tensor_white.detach().clone()
        tensor_black = parent_node.tensor_black.detach().clone()

        for removal in board_modifications.removals:
            #print('rem',removal)
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
            piece_color =appearance[2]
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

    node.tensor_white = tensor_white
    node.tensor_black = tensor_black


# def transform_board_pieces_square_two_sides(node, requires_grad_):
#     # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
#     board = node.board
#     inversion = 1
#     if board.chess_board.turn == chess.BLACK:
#         inversion = -1
#
#     transform = torch.zeros(768, requires_grad=requires_grad_)
#
#     for square in range(64):
#         piece = board.chess_board.piece_at(square)
#         if piece:
#             # print('p', square, piece.color, type(piece.piece_type))
#             piece_code = 6 * (piece.color != board.chess_board.turn) + (piece.piece_type - 1)
#             # print('dp', 64 * piece_code + square, 2 * piece.color - 1)
#             if piece.color == chess.BLACK:
#                 square_index = chess.square_mirror(square)
#             else:
#                 square_index = square
#             index = 64 * piece_code + square_index
#             transform[index] = (2 * piece.color - 1) * inversion
#         # transform[64 * piece_code + square] = 2 * piece.color - 1
#     print('::ssss@', node.id)
#
#     node.board.tensor_representation = transform
#     return transform


def transform_board_pieces_square_old(node, requires_grad_):
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    board = node.board
    # normalisation of the board so that it is white turn (possible color inversion if it was black's turn)
    inversion = 1
    if board.chess_board.turn == chess.BLACK:
        inversion = -1

    transform = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece_type = board.chess_board.piece_type_at(square)
        piece_color = board.chess_board.color_at(square)
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
    if board.chess_board.turn == chess.BLACK:
        inversion = -1

    transform = torch.zeros(384, requires_grad=requires_grad_)

    for square in range(64):
        piece = board.chess_board.piece_at(square)
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
