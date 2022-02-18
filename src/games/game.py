from src.chessenvironment.boards.board import BoardChi


class Game:
    """
    recording the game. it could be a doubling of the object Board but i keep it like this for now
    """

    STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def __init__(self, fen=STARTING_FEN):
        self.board = BoardChi(fen=fen)
        self.board_sequence = [self.board.copy()]


    def play(self, move):
        self.board.chess_board.push(move)
        self.board_sequence.append(self.board.copy())



    def is_legal(self, move):
        return self.board.is_legal(move)



    def who_plays(self):
         return self.board.chess_board.turn

    # def is_finished(self):
    #     return self.board.chess_board.is_game_over()


