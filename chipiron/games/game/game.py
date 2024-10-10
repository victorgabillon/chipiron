"""
Module for the Game class.
"""
import copy
import queue

import chess

from chipiron.environments.chess.board import IBoard
from chipiron.players.factory_higher_level import MoveFunction
from chipiron.utils import seed, unique_int_from_list
from chipiron.utils.communication.gui_messages import GameStatusMessage
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.is_dataclass import IsDataclass
from .game_playing_status import GamePlayingStatus
from ...environments.chess.board.utils import FenPlusMoveHistory


class Game:
    """
    Class representing a game of chess.
    """
    _playing_status: GamePlayingStatus
    _board: IBoard
    _seed: seed | None

    def __init__(
            self,
            board: IBoard,
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
        if self._playing_status.is_play():
            assert (move in self._board.legal_moves)
            self._board.play_move(move)
        else:
            print(f'Cannot play move if the game status is PAUSE {self._playing_status.status}')

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
    def board(self) -> IBoard:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self._board


class ObservableGame:
    """
    Represents an observable version of the Game object.
    """

    game: Game
    mailboxes_display: list[queue.Queue[IsDataclass]]

    # function that will be called by the observable game when the board is updated, which should query
    # at least one player to compute a move
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
        # the difference between the two is that board can be modified without asking the player to play
        # (for instance when using the button back)

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
        print('start playing')
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
            print('sending board', self.game.board.fen, self.game.board.move_history_stack)

            message: BoardMessage = BoardMessage(
                fen_plus_moves=FenPlusMoveHistory(current_fen=self.game.board.fen,
                                                  historical_moves=self.game.board.move_history_stack)
            )
            mailbox.put(item=message)

    def notify_players(self) -> None:
        """
        Notifies the players to ask for a move.
        """
        if not self.game.board.is_game_over():
            move_function: MoveFunction
            for move_function in self.move_functions:
                board_copy: IBoard = self.game.board.copy(stack=True)
                merged_seed: int | None = unique_int_from_list([self.game._seed, board_copy.ply()])
                if merged_seed is not None:
                    move_function(board=board_copy, seed_int=merged_seed)

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
    def board(self) -> IBoard:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self.game.board
