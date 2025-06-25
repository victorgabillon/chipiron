"""
Module for the Game class.
"""

import copy
import queue

from chipiron.environments.chess.board.iboard import IBoard
from chipiron.environments.chess.board.utils import fen
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.factory_higher_level import MoveFunction
from chipiron.utils import seed, unique_int_from_list
from chipiron.utils.communication.gui_messages import GameStatusMessage
from chipiron.utils.communication.player_game_messages import BoardMessage
from chipiron.utils.dataclass import IsDataclass

from .game_playing_status import GamePlayingStatus


class Game:
    """
    Class representing a game of chess.
    Note that a Game and a Board can look similar classes as a Board can (but not necessarily) include a record
    of previous moves. Board should be more lightweight that Board. Boards are more specific to a special position
     (but can include history) while Game are more related to the entire Game and the process generating it.
    Game needs the original fen, all the moves, the seed to generate and maybe more.
    """

    _playing_status: GamePlayingStatus  # todo should this be here? looks related to gui
    _current_board: IBoard
    _seed: seed | None
    _fen_history: list[fen]
    _move_history: list[moveUci]

    # list of boards object to implement rewind function without having to necessarily code it in the Board object.
    # this let the board object a bit more lightweight to speed up the Monte Carlo tree search
    _board_history: list[IBoard]

    def __init__(
        self, board: IBoard, playing_status: GamePlayingStatus, seed_: seed = 0
    ):
        """
        Initializes the Game object.

        Args:
            board (BoardChi): The chess board.
            playing_status (GamePlayingStatus): The playing status of the game.
            seed_ (seed): The seed for random number generation.
        """
        self._current_board = board
        self._playing_status = playing_status
        self._seed = seed_
        self._fen_history = [board.fen]
        self._move_history = []
        board_copy: IBoard = board.copy(stack=True)
        self._board_history = [board_copy]

    def play_move(self, move: moveKey) -> None:
        """
        Plays a move on the chess board.

        Args:
            move (chess.Move): The move to be played.

        Raises:
            AssertionError: If the move is not valid or the game status is not play.
        """
        if self._playing_status.is_play():
            assert move in [i for i in self._current_board.legal_moves]

            self.move_history.append(
                self._current_board.get_uci_from_move_key(move_key=move)
            )

            self._current_board.play_move_key(move)

            self.fen_history.append(self._current_board.fen)
            current_board_copy: IBoard = self._current_board.copy(
                stack=True, deep_copy_legal_moves=True
            )
            self._board_history.append(current_board_copy)
        else:
            print(
                f"Cannot play move if the game status is PAUSE {self._playing_status.status}"
            )

    def rewind_one_move(self) -> None:
        """
        Rewinds the last move on the chess board.

        Raises:
            AssertionError: If the game status is not paused.
        """

        if self._playing_status.is_paused():
            if len(self._board_history) > 1:
                del self._board_history[-1]
                self._current_board = self._board_history[-1].copy(
                    stack=True, deep_copy_legal_moves=True
                )
        else:
            print(f"Cannot rewind move if the game status is {self._playing_status}")

    @property
    def playing_status(self) -> GamePlayingStatus:
        """
        Gets or sets the playing status of the game.

        Returns:
            GamePlayingStatus: The playing status of the game.
        """
        return self._playing_status

    @playing_status.setter
    def playing_status(self, value: GamePlayingStatus) -> None:
        """
        Sets the playing status of the game.

        Args:
            value (GamePlayingStatus): The new playing status of the game.
        """
        self._playing_status = value

    def set_play_status(self) -> None:
        """
        Starts playing the game.
        """
        self._playing_status.play()

    def set_pause_status(self) -> None:
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
        return self._current_board

    @property
    def move_history(self) -> list[moveUci]:
        """
        Gets the history of move.

        Returns:
            list[chess.Move]: The history of move.
        """
        return self._move_history

    @property
    def fen_history(self) -> list[fen]:
        """
        Gets the history of fen.

        Returns:
            list[fen]: The history of fen.
        """
        return self._fen_history


class ObservableGame:
    """
    Represents an observable version of the Game object.
    """

    game: Game
    mailboxes_display: list[queue.Queue[IsDataclass]]

    # function that will be called by the observable game when the board is updated, which should query
    # at least one player to compute a move
    move_functions: list[MoveFunction]

    def __init__(self, game: Game) -> None:
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

    def register_display(self, mailbox: queue.Queue[IsDataclass]) -> None:
        """
        Registers a mailbox for displaying the board.

        Args:
            mailbox (queue.Queue[IsDataclass]): The mailbox for board to be displayed.
        """
        self.mailboxes_display.append(mailbox)

    def register_player(self, move_function: MoveFunction) -> None:
        """
        Registers a player to compute a move.

        Args:
            move_function (MoveFunction): The function to be called to compute a move.
        """
        self.move_functions.append(move_function)

    def play_move(self, move: moveKey) -> None:
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
    def playing_status(self, new_status: GamePlayingStatus) -> None:
        """
        Sets the playing status of the game.

        Args:
            new_status (GamePlayingStatus): The new playing status of the game.
        """
        self.game.playing_status = new_status
        raise Exception(
            "problem no notificaiton implemented. Maybe this function is deadcode?"
        )

    def set_play_status(self) -> None:
        """
        Starts playing the game.
        """
        print("start playing")
        self.game.set_play_status()
        self.notify_status()
        self.notify_display()

    def set_pause_status(self) -> None:
        """
        Pauses the game.
        """
        self.game.set_pause_status()
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
            print(
                "sending board to display",
                self.game.board.fen,
                self.game.board.move_history_stack,
            )

            message: BoardMessage = BoardMessage(
                fen_plus_moves=self.game.board.into_fen_plus_history()
            )
            mailbox.put(item=message)

    def query_move_from_players(self) -> None:
        """
        Notifies the players to ask for a move.
        """
        if not self.game.board.is_game_over():
            move_function: MoveFunction
            for move_function in self.move_functions:
                if not self.game.board.is_game_over():
                    merged_seed: int | None = unique_int_from_list(
                        [self.game._seed, self.game.board.ply()]
                    )
                    if merged_seed is not None:
                        move_function(
                            fen_plus_history=self.game.board.into_fen_plus_history(),
                            seed_int=merged_seed,
                        )

    def notify_status(self) -> None:
        """
        Notifies the status mailboxes with the updated game status.
        """
        print("notify game", self.game.playing_status.status)

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

    @property
    def move_history(self) -> list[moveUci]:
        """
        Gets the history of move.

        Returns:
            list[chess.Move]: The history of move.
        """
        return self.game.move_history

    @property
    def fen_history(self) -> list[fen]:
        """
        Gets the history of fen.

        Returns:
            list[fen]: The history of fen.
        """
        return self.game.fen_history
