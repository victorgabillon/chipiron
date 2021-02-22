from chessenvironment.boards.board import MyBoard
from displays.display_boards import DisplayBoards
import chess


class Game:
    """
    recording the game. it could be a doubling of the object Board but i keep it like this for now
    """

    GAME_RESULTS = [WIN_FOR_WHITE, WIN_FOR_BLACK, DRAW] = range(3)

    def __init__(self, chess_simulator, starting_position,syzygy):
        self.moves = []

        self.board = MyBoard(starting_position)
        self.chessSimulator = chess_simulator
        self.syzygy = syzygy
        self.chess_board_sequence  = [self.board.chess_board.copy()]

    def display_last_position(self):
        display = DisplayBoards()
        display.display(self.board)

    def play(self, move1):

        self.chessSimulator.stepModify(self.board, move1)
        self.moves.append(move1)
        if self.syzygy.fast_in_table(self.board):
            print('Theoretically finished with value for white: ', self.syzygy.sting_result(self.board))
        self.chess_board_sequence.append(self.board.chess_board.copy())


    def last_move(self):
        if len(self.moves) == 0:
            return None
        else:
            return self.moves[-1]

    def is_legal(self, move):
        return self.board.is_legal(move)

    def get_current_round(self):
        return self.board.chess_board.fullmove_number

    def who_plays(self):
        return self.board.chess_board.turn

    def is_finished(self):
        return self.board.chess_board.is_game_over()

    def tell_results(self):
        if self.syzygy.fast_in_table(self.board):
            print('Syzygy: Theoretical value for white', self.syzygy.sting_result(self.board))
        if self.board.chess_board.is_fivefold_repetition():
            print('is_fivefold_repetition')
        if self.board.chess_board.is_seventyfive_moves():
            print('is seventy five  moves')
        if self.board.chess_board.is_insufficient_material():
            print('is_insufficient_material')
        if self.board.chess_board.is_stalemate():
            print('is_stalemate')
        if self.board.chess_board.is_checkmate():
            print('is_checkmate')
        print(self.board.chess_board.result())

    def simple_results(self):
        if self.board.chess_board.result() == '*':
            if self.syzygy is None or not self.syzygy.fast_in_table(self.board):  # for debug i guess
                return (-10000, -10000, -10000)
            else: #todo what is happening here?
                val = self.syzygy.value_white(self.board, chess.WHITE)
                print('rf', val)
                if val == 0:
                    return self.DRAW
                if val == -1000:
                    return self.WIN_FOR_BLACK
                if val == 1000:
                    return self.WIN_FOR_WHITE
        else:
            result = self.board.chess_board.result()
            if result == '1/2-1/2':
                return self.DRAW
            if result == '0-1':
                return self.WIN_FOR_BLACK
            if result == '1-0':
                return self.WIN_FOR_WHITE
