from random import choice


class RandomPlayer:

    def __init__(self):
        pass

    def get_move_from_player(self, board, color):
        return choice(list(board.chess_board.legal_moves))
