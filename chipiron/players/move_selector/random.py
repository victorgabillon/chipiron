from random import choice
import chipiron.environments.chess.board as boards
import chess


class Random:

    def __init__(self):
        pass

    def select_move(self, board: boards.BoardChi) -> chess.Move:
        return choice(list(board.legal_moves))
