import chess
import queue
import copy

from .game_playing_status import GamePlayingStatus, PlayingStatus

from typing import Callable
from chipiron.environments.chess.board import BoardChi

from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.communication.gui_messages import GameStatusMessage

from chipiron.utils import seed, unique_int_from_list
from chipiron.utils.is_dataclass import IsDataclass


class Game:
    """
    Objet
    """
    _playing_status: GamePlayingStatus
    _board: BoardChi
    _seed: seed | None

    def __init__(
            self,
            board: BoardChi,
            playing_status: GamePlayingStatus,
            seed_: seed
    ):
        self._board = board
        self._playing_status = playing_status
        self._seed = seed_

    def play_move(self, move: chess.Move):
        assert (self._board.board.is_valid())
        if self._playing_status.is_play():
            assert (move in self._board.legal_moves)
            self._board.play_move(move)
        else:
            print(f'Cannot play move if the game status is PAUSE {self._playing_status.status}')
        assert (self._board.board.is_valid())

    def rewind_one_move(self):
        if self._playing_status.is_paused():
            self._board.rewind_one_move()
        else:
            print(f'Cannot rewind move if the game status is PLAY')

    @property
    def playing_status(self):
        return self._playing_status

    @playing_status.setter
    def playing_status(self, value):
        self._playing_status = value

    def play(self):
        self._playing_status.play()

    def pause(self):
        self._playing_status.pause()

    def is_paused(self):
        return self._playing_status.is_paused()

    def is_play(self):
        return self._playing_status.is_play()

    @property
    def board(self):
        return self._board


# function that will be called by the observable game when the board is updated, which should query at least one player
# to compute a move
MoveFunction = Callable[[BoardChi, seed], None]


class ObservableGame:
    """
    observable version of Game
    """

    game: Game
    mailboxes_display: list[queue.Queue[IsDataclass]]

    # function that will be called by the observable game when the board is updated, which should query
    # at least one player to compute a move
    move_functions: list[MoveFunction]

    def __init__(self, game: Game):
        self.game = game
        self.mailboxes_display = []  # mailboxes for board to be displayed
        self.move_functions = []  # mailboxes for board to be played
        # the difference between the two is that board can be modified without asking the player to play
        # (for instance when using the button back)

    def register_display(self, mailbox: queue.Queue[IsDataclass]):
        self.mailboxes_display.append(mailbox)

    def register_player(self, move_function: MoveFunction):
        self.move_functions.append(move_function)

    def set_starting_position(self, starting_position_arg=None, fen=None):
        self.board.set_starting_position(starting_position_arg, fen)
        self.notify_display()
        self.notify_players()

    def play_move(self, move: chess.Move):
        self.game.play_move(move)
        self.notify_display()
        self.notify_players()

    def rewind_one_move(self):
        self.game.rewind_one_move()
        self.notify_display()

    @property
    def playing_status(self):
        return self.game.playing_status

    @playing_status.setter
    def playing_status(
            self,
            new_status: PlayingStatus
    ) -> None:
        self.game.playing_status = new_status
        raise Exception('problem no notificaiton implemented. Maybe this function is deadcode?')

    def play(self):
        self.game.play()
        self.notify_players()
        self.notify_display()
        self.notify_status()

    def pause(self):
        self.game.pause()
        self.notify_status()

    def is_paused(self):
        return self.game.is_paused()

    def is_play(self):
        return self.game.is_play()

    def notify_display(self):
        for mailbox in self.mailboxes_display:
            board_copy = copy.deepcopy(self.game.board)
            message: BoardMessage = BoardMessage(board=board_copy)
            mailbox.put(item=message)

    def notify_players(self) -> None:
        """ Notify the players to ask for a move"""
        if not self.game.board.is_game_over():
            move_function: MoveFunction
            for move_function in self.move_functions:
                board_copy: BoardChi = copy.deepcopy(self.game.board)
                merged_seed: int | None = unique_int_from_list([self.game._seed, board_copy.ply()])
                if merged_seed is not None:
                    move_function(board_copy, merged_seed)


    def notify_status(self):
        print('notify game', self.game.playing_status.status)

        observable_copy = copy.copy(self.game.playing_status.status)
        message: GameStatusMessage = GameStatusMessage(status=observable_copy)
        for mailbox in self.mailboxes_display:
            mailbox.put(message)

    @property
    def board(self):
        return self.game.board
