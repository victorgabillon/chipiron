"""
Module for the Game class.
"""
import copy
import queue
from typing import Protocol

import chess

from chipiron.environments.chess.board import BoardChi
from chipiron.environments.chess.board.starting_position import AllStartingPositionArgs
from chipiron.utils import seed, unique_int_from_list
from chipiron.utils.communication.gui_messages import GameStatusMessage
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.is_dataclass import IsDataclass
from .game_playing_status import GamePlayingStatus


class Game:
    """ 
    Class representing a game of chess.
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
        """
        Initializes the Game object.

        Args:
            board (BoardChi): The chess board.
            playing_status (GamePlayingStatus): The playing status of the game.
            seed_ (seed): The seed for random number generation.
        """
        self._board = board
        self._playing_status = playing_status
        self._seed = seed_

    def play_move(
            self,
            move: chess.Move
    ) -> None:
        """
        Plays a move on the chess board.

        Args:
            move (chess.Move): The move to be played.

        Raises:
            AssertionError: If the move is not valid or the game status is not play.
        """
        assert (self._board.board.is_valid())
        if self._playing_status.is_play():
            assert (move in self._board.legal_moves)
            self._board.play_move(move)
        else:
            print(f'Cannot play move if the game status is PAUSE {self._playing_status.status}')
        assert (self._board.board.is_valid())

    def rewind_one_move(self) -> None:
        """
        Rewinds the last move on the chess board.

        Raises:
            AssertionError: If the game status is not paused.
        """
        if self._playing_status.is_paused():
            self._board.rewind_one_move()
        else:
            print('Cannot rewind move if the game status is PLAY')

    @property
    def playing_status(self) -> GamePlayingStatus:
        """
        Gets or sets the playing status of the game.

        Returns:
            GamePlayingStatus: The playing status of the game.
        """
        return self._playing_status

    @playing_status.setter
    def playing_status(
            self,
            value: GamePlayingStatus
    ) -> None:
        """
        Sets the playing status of the game.

        Args:
            value (GamePlayingStatus): The new playing status of the game.
        """
        self._playing_status = value

    def play(self) -> None:
        """
        Starts playing the game.
        """
        self._playing_status.play()

    def pause(self) -> None:
        """
        Pauses the game.
        """
        self._playing_status.pause()

    def is_paused(self) -> bool:
        """
        Checks if the game is paused.

        Returns:
            bool: True if the game is paused, False otherwise.
        """
        return self._playing_status.is_paused()

    def is_play(self) -> bool:
        """
        Checks if the game is being played.

        Returns:
            bool: True if the game is being played, False otherwise.
        """
        return self._playing_status.is_play()

    @property
    def board(self) -> BoardChi:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self._board


class MoveFunction(Protocol):
    def __call__(
            self,
            board: BoardChi,
            seed_: seed
    ) -> None: ...


class ObservableGame:
    """
    Represents an observable version of the Game object.
    """

    game: Game
    mailboxes_display: list[queue.Queue[IsDataclass]]
    move_functions: list[MoveFunction]

    def __init__(
            self,
            game: Game
    ) -> None:
        """
        Initializes the ObservableGame object.

        Args:
            game (Game): The underlying Game object.
        """
        self.game = game
        self.mailboxes_display = []  # mailboxes for board to be displayed
        self.move_functions = []  # mailboxes for board to be played

    def register_display(
            self,
            mailbox: queue.Queue[IsDataclass]
    ) -> None:
        """
        Registers a mailbox for displaying the board.

        Args:
            mailbox (queue.Queue[IsDataclass]): The mailbox for board to be displayed.
        """
        self.mailboxes_display.append(mailbox)

    def register_player(
            self,
            move_function: MoveFunction
    ) -> None:
        """
        Registers a player to compute a move.

        Args:
            move_function (MoveFunction): The function to be called to compute a move.
        """
        self.move_functions.append(move_function)

    def set_starting_position(
            self,
            starting_position_arg: AllStartingPositionArgs | None = None,
            fen: str | None = None
    ) -> None:
        """
        Sets the starting position of the chess board.

        Args:
            starting_position_arg (AllStartingPositionArgs | None): The starting position arguments.
            fen (str | None): The FEN string representing the starting position.
        """
        self.board.set_starting_position(starting_position_arg, fen)
        self.notify_display()
        self.notify_players()

    def play_move(
            self,
            move: chess.Move
    ) -> None:
        """
        Plays a move on the chess board.

        Args:
            move (chess.Move): The move to be played.
        """
        self.game.play_move(move)
        self.notify_display()
        self.notify_players()

    def rewind_one_move(self) -> None:
        """
        Rewinds the last move on the chess board.
        """
        self.game.rewind_one_move()
        self.notify_display()

    @property
    def playing_status(self) -> GamePlayingStatus:
        """
        Gets the playing status of the game.

        Returns:
            GamePlayingStatus: The playing status of the game.
        """
        return self.game.playing_status

    @playing_status.setter
    def playing_status(
            self,
            new_status: GamePlayingStatus
    ) -> None:
        """
        Sets the playing status of the game.

        Args:
            new_status (GamePlayingStatus): The new playing status of the game.
        """
        self.game.playing_status = new_status
        raise Exception('problem no notificaiton implemented. Maybe this function is deadcode?')

    def play(self) -> None:
        """
        Starts playing the game.
        """
        self.game.play()
        self.notify_players()
        self.notify_display()
        self.notify_status()

    def pause(self) -> None:
        """
        Pauses the game.
        """
        self.game.pause()
        self.notify_status()

    def is_paused(self) -> bool:
        """
        Checks if the game is paused.

        Returns:
            bool: True if the game is paused, False otherwise.
        """
        return self.game.is_paused()

    def is_play(self) -> bool:
        """
        Checks if the game is being played.

        Returns:
            bool: True if the game is being played, False otherwise.
        """
        return self.game.is_play()

    def notify_display(self) -> None:
        """
        Notifies the display mailboxes with the updated board.
        """
        for mailbox in self.mailboxes_display:
            board_copy = copy.deepcopy(self.game.board)
            message: BoardMessage = BoardMessage(board=board_copy)
            mailbox.put(item=message)

    def notify_players(self) -> None:
        """
        Notifies the players to ask for a move.
        """
        if not self.game.board.is_game_over():
            move_function: MoveFunction
            for move_function in self.move_functions:
                board_copy: BoardChi = copy.deepcopy(self.game.board)
                merged_seed: int | None = unique_int_from_list([self.game._seed, board_copy.ply()])
                if merged_seed is not None:
                    move_function(board=board_copy, seed_=merged_seed)

    def notify_status(self) -> None:
        """
        Notifies the status mailboxes with the updated game status.
        """
        print('notify game', self.game.playing_status.status)

        observable_copy = copy.copy(self.game.playing_status.status)
        message: GameStatusMessage = GameStatusMessage(status=observable_copy)
        for mailbox in self.mailboxes_display:
            mailbox.put(message)

    @property
    def board(self) -> BoardChi:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self.game.board
