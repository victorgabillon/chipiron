import chipiron as ch


class BoardHistory:
    """
    object that records the changes the board
    """

    def __init__(self, first_board: ch.chess.board.BoardChi | None = None):
        if first_board is not None:
            self.list_of_board = [first_board]
        else:
            self.list_of_board = []

    def play_move(self, ):
        self.appearances.add(appearance)

    def rewind_move(self):
        self.removals.add(removal)
