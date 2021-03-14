
import random

piece_list = ["R", "N", "B", "Q", "P"]


def place_kings(brd):
    while True:
        rank_white, file_white, rank_black, file_black = random.randint(0 ,7), random.randint(0 ,7), random.randint(0
                                                                                                                    ,7), random.randint \
            (0 ,7)
        diff_list = [abs(rank_white - rank_black),  abs(file_white - file_black)]
        if sum(diff_list) > 2 or set(diff_list) == set([0, 2]):
            brd[rank_white][file_white], brd[rank_black][file_black] = "K", "k"
            break

def populate_board(brd, wp, bp):
    for x in range(2):
        if x == 0:
            piece_amount = wp
            pieces = piece_list
        else:
            piece_amount = bp
            pieces = [s.lower() for s in piece_list]
        while piece_amount != 0:
            # print('piece_amount',piece_amount)
            # print('pbrd',brd)

            piece_rank, piece_file = random.randint(0, 7), random.randint(0, 7)
            piece = random.choice(pieces)
            if brd[piece_rank][piece_file] == " " and pawn_on_bad_rows(piece, piece_rank) == False:
                brd[piece_rank][piece_file] = piece
                piece_amount -= 1


def fen_from_board(brd):
    fen = ""
    for x in brd:
        n = 0
        for y in x:
            if y == " ":
                n += 1
            else:
                if n != 0:
                    fen += str(n)
                fen += y
                n = 0
        if n != 0:
            fen += str(n)
        fen += "/" if fen.count("/") < 7 else ""
    if random.random() > .5:
        fen += " w"
    else:
        fen += " b"
    fen += " - - 0 1"
    return fen


def pawn_on_bad_rows(pc, pr):
    if (pc == "P" or pc == "p") and (pr == 0 or pr == 7):
        return True
    return False


def start():
    piece_amount = random.randint(1, 3)
    piece_amount_white = random.randint(0, piece_amount)
    piece_amount_black = piece_amount - piece_amount_white

    # print('#d#',piece_amount_white, piece_amount_black)

    place_kings(board)

    populate_board(board, piece_amount_white, piece_amount_black)
    fen = fen_from_board(board)
    # print(fen)
    # for x in board:
    #    print(x)
    return fen


def start_2():
    diff = random.randint(0, 2) - 1
    bl_wh = random.randint(0, 1)
    if bl_wh == 0:
        piece_amount_white = random.randint(3, 14)
        piece_amount_black = piece_amount_white + diff
    else:
        piece_amount_black = random.randint(3, 14)
        piece_amount_white = piece_amount_black + diff

    # print('#d#',piece_amount_white, piece_amount_black)

    place_kings(board)

    populate_board(board, piece_amount_white, piece_amount_black)
    fen = fen_from_board(board)
    # print(fen)
    # for x in board:
    #    print(x)
    return fen


import chess.pgn
import chess
# entry point
import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
from players.boardevaluators.syzygy import Syzygy
from chessenvironment.chess_environment import ChessEnvironment
from chessenvironment.boards.board import MyBoard

data_frame_file_name = '/home/victor/random_classify'
the_dic = []

count = 0
while True:

    count += 1
    if count % 1000 == 0:
        print('count', count)

    if count % 10000 == 0:  # save
        new_data_frame_states = pd.DataFrame.from_dict(the_dic)
        print('%%', len(new_data_frame_states.index))
        new_data_frame_states.to_pickle(data_frame_file_name)

    # Iterate through all moves and play them on a board.

    #     try:
    board = [[" " for x in range(8)] for y in range(8)]
    fen = start_2()
    fen = '2B5/R1QRb3/1Qp3pp/2qnR2P/k6r/R2Pnrb1/1R1K4/4B1bQ b - - 0 1'
    my_board = MyBoard(fen=fen)
    print(fen,my_board.chess_board.is_valid())
    if my_board.chess_board.is_valid():
        print(fen)
        print('Excepts')

        engine = chess.engine.SimpleEngine.popen_uci("/home/victor/stockfish_13_linux_x64/stockfish_13_linux_x64")

        print('Exceptdw')
        try:
            print('Exceptqwd')

            info = engine.analyse(my_board.chess_board, chess.engine.Limit(depth=20))
            print('Exceptsad')
            problemos = False
            print('Exceptxsd')

            engine.quit()
        except:
            print('Except')
            problemos = True
            pass

        if problemos:
            continue

        print("Score:", info["score"], fen)

        print(my_board.chess_board)
        # input("Press Enter to continue...")

        tag = None
        if info["score"].pov(chess.WHITE) > chess.engine.Cp(700):
            tag = 'Win-Wh'
        if info["score"].pov(chess.BLACK) > chess.engine.Cp(700):
            tag = 'Win-Bl'
        if info["score"].pov(chess.BLACK) < chess.engine.Cp(2) and info["score"].pov(chess.WHITE) < chess.engine.Cp(2):
            tag = 'Draw'
            # print(chess_board,{'fen': chess_board.fen(),'explored':'engine','final_value':tag})
            # input("Press Enter to continue...")
        # print('ssd@',centi_score_white)

        if tag is not None:
            the_dic.append({'fen': my_board.chess_board.fen(), 'explored': 'engine', 'final_value': tag})
            # print(chess_board,{'fen': chess_board.fen(),'explored':'engine','final_value':tag})
            # input("Press Enter to continue...")


