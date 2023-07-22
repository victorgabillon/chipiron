from random import choice
import environments.chess.board as boards
import chess

class RandomPlayer:

    def __init__(self):
        pass

    def select_move(self, board: boards.BoardChi) -> chess.Move:
        return choice(list(board.legal_moves))
