from src.chessenvironment.board.board import BoardChi
from src.chessenvironment.board.board_observable import BoardObservable
import chess


def create_board(subscribers):
    board = BoardChi()

    if subscribers:
        board = BoardObservable(board)
        for subscriber in subscribers:
            board.register_mailbox(subscriber ,'board_to_display')

    return board
