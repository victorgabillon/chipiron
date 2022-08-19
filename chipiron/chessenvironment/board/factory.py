from chipiron.chessenvironment.board.board import BoardChi
from chipiron.chessenvironment.board.board_observable import BoardObservable


def create_board(subscribers) -> BoardChi:
    board: BoardChi = BoardChi()

    # if subscribers:
    #     board = BoardObservable(board)
    #     for subscriber in subscribers:
    #
    # board.register_mailbox(subscriber, 'board_to_display')

    return board
