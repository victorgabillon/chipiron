from random import choice
import chipiron as ch


class RandomPlayer:

    def __init__(self):
        pass

    def select_move(self, board: ch.chess.board.BoardChi):
        return choice(list(board.legal_moves))
