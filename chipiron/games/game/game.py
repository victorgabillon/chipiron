"""
Module for the Game class.
"""

from typing import TYPE_CHECKING, Annotated

from valanga import StateTag, TurnState
from valanga.game import ActionKey, Seed

from chipiron.displays.gui_protocol import Scope
from chipiron.players.communications.player_request_encoder import PlayerRequestEncoder
from chipiron.players.factory_higher_level import MoveFunction
from chipiron.utils.communication.gui_encoder import GuiEncoder
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.small_tools import unique_int_from_list

from .game_playing_status import GamePlayingStatus, PlayingStatus

if TYPE_CHECKING:
    from chipiron.players.communications.player_message import PlayerRequest
    from chipiron.displays.gui_protocol import UpdatePayload

type Ply = Annotated[
    int, "The number of one turn taken by one of the players in the game so far"
]


class Game[StateT: TurnState = TurnState]:
    """
    Class representing a game of chess.

    Note:
        A Game and a Board can look similar as a Board can (but not necessarily) include a record of previous moves.
        Board should be more lightweight than Game. Boards are more specific to a special position
        (but can include history) while Game is more related to the entire game and the process generating it.
        Game needs the original FEN, all the moves, the seed to generate and maybe more.
    """

    _playing_status: GamePlayingStatus  # todo should this be here? looks related to gui
    _current_state: StateT
    _seed: Seed | None
    _state_tag_history: list[StateTag]
    _action_history: list[ActionKey]
    _ply: Ply

    # list of boards object to implement rewind function without having to necessarily code it in the Board object.
    # this let the board object a bit more lightweight to speed up the Monte Carlo tree search
    _state_history: list[StateT]

    def __init__(
        self, state: StateT, playing_status: GamePlayingStatus, seed_: Seed = 0
    ):
        """
        Initializes the Game object.

        Args:
            board (BoardChi): The chess board.
            playing_status (GamePlayingStatus): The playing status of the game.
            seed_ (seed): The seed for random number generation.
        """
        self._current_state = state
        self._playing_status = playing_status
        self._seed = seed_
        self._state_tag_history = [state.tag]
        self._action_history = []
        state_copy: StateT = state.copy(stack=True)
        self._state_history = [state_copy]
        self._ply = 0

    @property
    def ply(self) -> Ply:
        """
        Gets the number of turns taken so far in the game.

        Returns:
            Ply: The number of turns taken so far in the game.
        """
        return self._ply

    @property
    def seed(self) -> Seed | None:
        """
        Gets the seed for random number generation.

        Returns:
            seed: The seed for random number generation.
        """
        return self._seed

    def play_move(self, action: ActionKey) -> None:
        """
        Plays a move on the chess board.

        Args:
            action (ActionKey): The action to be played.

        Raises:
            AssertionError: If the action is not valid or the game status is not play.
        """
        if self._playing_status.is_play():
            assert action in [i for i in self._current_state.branch_keys]

            self._action_history.append(
                self._current_state.branch_name_from_key(key=action)
            )

            self._current_state.step(branch_key=action)

            self._state_tag_history.append(self._current_state.tag)
            current_state_copy: StateT = self._current_state.copy(
                stack=True, deep_copy_legal_moves=True
            )
            self._state_history.append(current_state_copy)
            self._ply += 1
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
            if len(self._state_history) > 1:
                del self._state_history[-1]
                self._current_state = self._state_history[-1].copy(
                    stack=True, deep_copy_legal_moves=True
                )
                self._ply -= 1
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
    def state(self) -> StateT:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self._current_state

    @property
    def action_history(self) -> list[ActionKey]:
        """
        Gets the history of move.

        Returns:
            list[chess.Move]: The history of move.
        """
        return self._action_history

    @property
    def state_tag_history(self) -> list[StateTag]:
        """
        Gets the history of fen.

        Returns:
            list[Fen]: The history of fen.
        """
        return self._state_tag_history


class ObservableGame[StateT: TurnState = TurnState]:
    """
    Represents an observable version of the Game object.
    """

    game: Game[StateT]
    player_encoder: PlayerRequestEncoder[StateT]
    gui_encoder: GuiEncoder[StateT]
    scope: Scope
    mailboxes_display: list[GuiPublisher]

    # function that will be called by the observable game when the board is updated, which should query
    # at least one player to compute a move
    move_functions: list[MoveFunction]

    def __init__(
        self, game: Game[StateT], gui_encoder: GuiEncoder[StateT], scope: Scope, player_encoder: PlayerRequestEncoder[StateT]
    ) -> None:
        """
        Initializes the ObservableGame object.
        Args:
            game (Game): The game object to be observed.
            encoder (GuiEncoder): The GUI encoder for the game.
            scope (Scope): Routing scope for this game.
        """
        self.game = game
        self.gui_encoder = gui_encoder
        self.player_encoder = player_encoder
        self.scope = scope

        self.mailboxes_display = []  # mailboxes for board to be displayed
        self.move_functions = []  # mailboxes for board to be played
        # the difference between the two is that board can be modified without asking the player to play
        # (for instance when using the button back)

    def register_display(self, mailbox: GuiPublisher) -> None:
        """
        Registers a mailbox for displaying the board.

        Args:
            mailbox (GuiPublisher): The mailbox for board to be displayed.
        """
        self.mailboxes_display.append(mailbox)

    def register_player(self, move_function: MoveFunction) -> None:
        """
        Registers a player to compute a move.

        Args:
            move_function (MoveFunction): The function to be called to compute a move.
        """
        self.move_functions.append(move_function)

    def play_move(self, action: ActionKey) -> None:
        """
        Plays a move on the chess board.

        Args:
            action (ActionKey): The action to be played.
        """
        self.game.play_move(action)
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
        raise NotImplementedError(
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
        # Build payload from domain state
        payload = self.gui_encoder.make_state_payload(
            state=self.game.state,
            seed=self.game.seed,
        )
        for pub in self.mailboxes_display:
            chipiron_logger.debug(
                "Sending board to display - FEN: %s, Move history: %s",
                self.game.state.tag,
                self.game.action_history,
            )
            pub.publish(payload)

    def query_move_from_players(self) -> None:
        """
        Notifies the players to ask for a move.
        """
        if self.game.state.is_game_over():
            return

        merged_seed: int | None = unique_int_from_list([self.game.seed, self.game.ply])
        if merged_seed is None:
            merged_seed = int(self.game.ply)  # deterministic fallback
        request: PlayerRequest = self.player_encoder.make_move_request(
            state=self.game.state, seed=merged_seed, scope=self.scope
        )

        move_function: MoveFunction
        for move_function in self.move_functions:
            if self.game.state.is_game_over():
                break
            move_function(request)

    def notify_status(self) -> None:
        """
        Notifies the status mailboxes with the updated game status.
        """
        chipiron_logger.debug("notify game %s", self.game.playing_status.status)

        # Map internal playing status to transport enum
        status = PlayingStatus.PLAY if self.game.is_play() else PlayingStatus.PAUSE

        payload: UpdatePayload = self.gui_encoder.make_status_payload(status=status)

        for mailbox in self.mailboxes_display:
            mailbox.publish(payload)

    @property
    def state(self) -> StateT:
        """
        Gets the chess board.

        Returns:
            BoardChi: The chess board.
        """
        return self.game.state

    @property
    def action_history(self) -> list[ActionKey]:
        """
        Gets the history of move.

        Returns:
            list[chess.Move]: The history of move.
        """
        return self.game.action_history

    @property
    def state_tag_history(self) -> list[StateTag]:
        """
        Gets the history of fen.

        Returns:
            list[Fen]: The history of fen.
        """
        return self.game.state_tag_history

    @property
    def ply(self) -> Ply:
        """
        Gets the number of turns taken so far in the game.

        Returns:
            Ply: The number of turns taken so far in the game.
        """
        return self.game.ply
