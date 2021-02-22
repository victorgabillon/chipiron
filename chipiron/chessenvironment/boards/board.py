import chess
from chessenvironment.boards.board_tools import convertToFen
import time

COLORS = [WHITE, BLACK] = [True, False]


class MyBoard:
    """
    object that describes the current board. it wraps the chess Board from the chess package so it has a lot more in it
    """

    def __init__(self, starting_position, chess_board=None):
        if chess_board is None:
            if starting_position['type'] == 'classic':
                self.chess_board = chess.Board()
            elif starting_position['type'] == 'fromFile':

                fileName = starting_position['options']['fileName']
                fen = self.load_from_file(fileName)
                self.chess_board = chess.Board(fen)
        else:
            self.chess_board = chess_board

    def is_legal(self, move):
        return move in self.chess_board.legal_moves

    def get_legal_moves(self):
        return self.chess_board.legal_moves

    def play(self, move1):
        self.chess_board.push(move1)

    def load_from_file(self, fileName):
        with  open('runs/StartingBoards/' + fileName, "r") as f:
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

    def number_of_pieces_on_the_board(self):
        return bin(self.chess_board.occupied).count('1')
