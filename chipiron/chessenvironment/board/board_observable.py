import copy
import chess
import chipiron.chessenvironment.board as board_mod


class BoardObservable:
    """
    This class makes an object of the class board_mod.BoardChi observable by subscribers,
     notified whenever the board is modified.
    """

    def __init__(self,
                 board: board_mod.BoardChi) -> None:
        """

        Args:
            board: the board that is observed
        """
        self.board = board
        self.mailboxes_display = []  # mailboxes for board to be displayed
        self.mailboxes_play = []  # mailboxes for board to be played
        # the difference between the two is that board can be modified without asking the player to play
        # (for instance when using the button back)

    def play_move(self, move: chess.Move):
        self.board.play_move(move)
        self.notify_board_display()
        self.notify_board_play()

    def rewind_one_move(self):
        self.board.rewind_one_move()
        self.notify_board_display()

    def register_mailbox(self, mailbox, type):
        if type == 'board_to_display':
            self.mailboxes_display.append(mailbox)
        if type == 'board_to_play':
            self.mailboxes_play.append(mailbox)

    def notify_board_display(self):
        for mailbox in self.mailboxes_display:
            board_copy = copy.deepcopy(self.board)
            mailbox.put({'type': 'board', 'board': board_copy})

    def notify_board_play(self):
        for mailbox in self.mailboxes_play:
            board_copy = copy.deepcopy(self.board)
            mailbox.put({'type': 'board', 'board': board_copy})

    # forwarding
    def set_starting_position(self, starting_position_arg=None, fen=None):
        self.board.set_starting_position(starting_position_arg, fen)
        self.notify_board_display()
        self.notify_board_play()

    def ply(self) -> int:
        return self.board.ply()

    def fen(self):
        return self.board.fen()

    def number_of_pieces_on_the_board(self):
        return self.board.number_of_pieces_on_the_board()

    def is_game_over(self):
        return self.board.is_game_over()

    @property
    def turn(self):
        return self.board.turn

    @property
    def legal_moves(self):
        return self.board.legal_moves

    def print_chess_board(self):
        self.board.print_chess_board()
