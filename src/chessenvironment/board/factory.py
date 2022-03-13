from src.chessenvironment.board.board import BoardChi
from src.chessenvironment.board.board_observable import BoardObservable


def create_board(subscribers):
    board = BoardChi()

    print('subsccribers',subscribers)
    if subscribers:
        board = BoardObservable(board)
        for subscriber in subscribers:
            board.register_mailbox(subscriber)

    return board
