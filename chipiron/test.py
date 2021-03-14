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

fen = '2B5/R1QRb3/1Qp3pp/2qnR2P/k6r/R2Pnrb1/1R1K4/4B1bQ b - - 0 1'
my_board = MyBoard(fen=fen)
print('^^', fen, my_board.chess_board.is_valid())
print('^^', fen, my_board.chess_board.is_game_over())

print(my_board.chess_board)