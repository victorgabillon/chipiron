import copy


class BoardObservable:

    def __init__(self, board):
        self.board = board
        self.mailboxes = []

    def play_move(self, move):
        self.board.play_move(move)
        self.notify_new_board()

    def back_one_move(self):
        self.board.rewind_move()
        self.notify_new_board()

    def rewind_one_move(self, move):
        self.board.rewind_one_move(move)
        self.notify_new_board()

    def copy_board(self):
        board_copy = copy.deepcopy(self.board)
        return board_copy

    def register_mailbox(self, mailbox):
        self.mailboxes.append(mailbox)

    def notify_new_board(self):
        for mailbox in self.mailboxes:
            board_copy = self.copy_board()
            mailbox.put({'type': 'board', 'board': board_copy})

    # forwarding
    def set_starting_position(self, starting_position_arg=None, fen=None):
        self.board.set_starting_position(starting_position_arg, fen)
        self.notify_new_board()

    def ply(self):
        self.board.ply()

    def print_chess_board(self):
        self.board.print_chess_board()
