import chess
from chessenvironment.boards.board_tools import convertToFen
import time

COLORS = [WHITE, BLACK] = [True, False]


class MyBoard:
    """
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness
    """

    def __init__(self, starting_position_arg=None, chess_board=None, fen=None):

        if starting_position_arg is not None:
            if starting_position_arg['type'] == 'classic':
                self.chess_board = chess.Board()
            elif starting_position_arg['type'] == 'fromFile':
                file_name = starting_position_arg['options']['file_name']
                fen = self.load_from_file(file_name)
                self.chess_board = chess.Board(fen)
            elif starting_position_arg['type'] == 'fen':
                fen = starting_position_arg['fen']
                print(';fen', fen)
                self.chess_board = chess.Board(fen)

        if fen is not None:
            self.chess_board = chess.Board(fen)

        if chess_board is not None:
            self.chess_board = chess_board

    def is_legal(self, move):
        return move in self.chess_board.legal_moves

    def get_legal_moves(self):
        return self.chess_board.legal_moves

    def play(self, move1):
        self.chess_board.push(move1)

    def load_from_file(self, file_name):
        with  open('chipiron/runs/StartingBoards/' + file_name, "r") as f:
            asciiBoard = f.read()
            fen = convertToFen(asciiBoard)
        return fen

    def compute_key(self):
        string = str(self.chess_board.pawns) + str(self.chess_board.knights) \
                 + str(self.chess_board.bishops) + str(self.chess_board.rooks) \
                 + str(self.chess_board.queens) + str(self.chess_board.kings) \
                 + str(self.chess_board.turn) + str(self.chess_board.castling_rights) \
                 + str(self.chess_board.ep_square) + str(self.chess_board.halfmove_clock) \
                 + str(self.chess_board.occupied_co[WHITE]) + str(self.chess_board.occupied_co[BLACK]) \
                 + str(self.chess_board.promoted) \
                 + str(self.chess_board.fullmove_number)
        return string

    def fast_representation(self):
        # fastRep = board.chessBoard.fen()
        return self.compute_key()

    def print_chess_board(self):
        print(self.chess_board)
        print(self.chess_board.fen())

    def __str__(self):
        return self.chess_board.__str__()


    def number_of_pieces_on_the_board(self):
        return bin(self.chess_board.occupied).count('1')

    def copy(self):
        return type(self)(None,self.chess_board.copy())
