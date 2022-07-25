from random import choice
from chipiron.chessenvironment.board.iboard import IBoard


class RandomPlayer:

    def __init__(self):
        pass

    def select_move(self, board: IBoard):
        return choice(list(board.chess_board.legal_moves))
