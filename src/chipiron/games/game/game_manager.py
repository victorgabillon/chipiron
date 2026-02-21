"""Module in charge of managing the game. It is the main class that will be used to play a game."""

import os
import queue
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal, assert_never

import yaml
from atomheart.move_factory import MoveFactory
from valanga import BranchKey, Color, StateTag, TurnState
from valanga.evaluations import StateEvaluation
from valanga.game import ActionKey

import chipiron.players as players_m
from chipiron.match.domain_events import ActionApplied, IllegalAction, MatchOver, NeedAction
from chipiron.displays.gui_protocol import (
    CmdBackOneMove,
    CmdSetStatus,
    GuiCommand,
    HumanActionChosen,
    Scope,
)
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.boardevaluators.board_evaluator import IGameStateEvaluator
from chipiron.players.communications.player_message import (
    EvMove,
    EvProgress,
    PlayerEvent,
)
from chipiron.utils import MyPath
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.dataclass import custom_asdict_factory
from chipiron.utils.logger import chipiron_logger

from .final_game_result import FinalGameResult, GameReport
from .game import ObservableGame, Ply
from .game_args import GameArgs
from .game_rules import (
    GameOutcome,
    GameRules,
    OutcomeKind,
    OutcomeSource,
    outcome_to_final_game_result,
)
from .progress_collector import PlayerProgressCollectorP

if TYPE_CHECKING:
    from chipiron.match.match_controller import MatchController



class GameManager[StateT: TurnState = TurnState]:
    """Object in charge of playing one game."""

    # The game object that is managed
    game: ObservableGame[StateT]

    # Evaluators that just evaluates the boards but are not players (just spectators) for display info of who is winning
    # according to them
    display_state_evaluator: IGameStateEvaluator[StateT]

    # folder to log results
    output_folder_path: MyPath | None
    path_to_store_result: MyPath | None

    # args of the Game
    args: GameArgs

    # Dictionary mapping colors to player names?
    player_color_to_id: dict[Color, str]

    # A Queue for receiving messages from other process or functions such as players or Gui
    main_thread_mailbox: queue.Queue[MainMailboxMessage]

    # The list of players (lifecycle handles)
    players: list[players_m.PlayerHandle]

    # A move factory
    move_factory: MoveFactory

    # an object for collecting how advances each player is in its thinking/computation of the moves
    progress_collector: PlayerProgressCollectorP

    # Game-specific rules adapter
    rules: GameRules[StateT]

    # Correlation id for synchronous request/reply orchestration
    _request_id: int
    _scope: Scope | None

    def __init__(
        self,
        game: ObservableGame[StateT],
        display_state_evaluator: IGameStateEvaluator[StateT],
        output_folder_path: MyPath | None,
        args: GameArgs,
        player_color_to_id: dict[Color, str],
        main_thread_mailbox: queue.Queue[MainMailboxMessage],
        players: list[players_m.PlayerHandle],
        move_factory: MoveFactory,
        progress_collector: PlayerProgressCollectorP,
        rules: GameRules[StateT],
    ) -> None:
        """Initialize the GameManager Class. If the args, and players are not given a value it is set to None,.

        waiting for the set methods to be called. This is done like this so that the players can be changed
        (often swapped) easily.

        Args:
            game (ObservableGame): The observable game object.
            display_board_evaluator (IGameBoardEvaluator): The board evaluator to display an independent evaluation.
            output_folder_path (path | None): The output folder path or None.
            args (GameArgs): The game arguments.
            player_color_to_id (dict[chess.Color, str]): The dictionary mapping player color to player ID.
            main_thread_mailbox (queue.Queue[MainMailboxMessage]): The main thread mailbox.
            players (list[players_m.PlayerProcess | players_m.GamePlayer]): The list of players.

        Returns:
            None

        """
        self.game = game
        self.path_to_store_result = (
            os.path.join(output_folder_path, "games")
            if output_folder_path is not None
            else None
        )
        self.display_state_evaluator = display_state_evaluator
        self.args = args
        self.player_color_to_id = player_color_to_id
        self.main_thread_mailbox = main_thread_mailbox
        self.players = players
        self.move_factory = move_factory
        self.progress_collector = progress_collector
        self.rules = rules
        self._request_id = 0
        self._scope = None
    def start_match_sync(self, scope: Scope) -> list[object]:
        """Start a match synchronously and request the first action."""
        self._scope = scope
        self._request_id = 0
        state = self.game.state
        return [
            NeedAction(
                scope=scope,
                color=state.turn,
                request_id=self._request_id,
                state=state,
            )
        ]

    def propose_action_sync(
        self,
        scope: Scope,
        color: Color,
        request_id: int,
        action: BranchKey,
    ) -> list[object]:
        """Apply a proposed action synchronously and emit domain events."""
        if self._scope != scope:
            return []

        if request_id != self._request_id:
            return []

        state = self.game.state

        if color != state.turn:
            return []

        try:
            transition = self.game.dynamics.step(state=state, action=action)
        except Exception as exc:  # noqa: BLE001 - preserve reducer-level error payload
            return [
                IllegalAction(scope, color, request_id, action, str(exc)),
                NeedAction(scope, state.turn, self._request_id, state),
            ]

        action_name = self.game.dynamics.action_name(state, action)
        self.game.apply_transition(transition=transition, action_name=action_name)

        out: list[object] = [
            ActionApplied(scope, color, request_id, action, transition)
        ]

        is_over = getattr(transition, "is_over", False)
        over_event = getattr(transition, "over_event", None)
        if is_over or transition.next_state.is_game_over():
            out.append(MatchOver(scope, transition.next_state, over_event))
            return out

        self._request_id += 1

        out.append(
            NeedAction(
                scope,
                transition.next_state.turn,
                self._request_id,
                transition.next_state,
            )
        )

        return out

    def external_eval(self) -> tuple[StateEvaluation | None, StateEvaluation]:
        """Evaluate the game board using the display board evaluator.

        Returns:
            tuple[StateEvaluation | None, StateEvaluation]: A tuple containing the evaluation scores.

        """
        return self.display_state_evaluator.evaluate(self.game.state)

    def play_one_move(self, action: ActionKey) -> None:
        """Play one move in the game.

        Args:
            action (ActionKey): The action to be played.

        """
        self.game.play_move(action)

    def rewind_one_move(self) -> None:
        """Rewinds the game by one move.

        This method rewinds the game by one move, undoing the last move made.

        Returns:
            None

        """
        self.game.rewind_one_move()

    def play_one_game(self, controller: "MatchController") -> GameReport:
        """Play one game.

        Returns:
            GameReport: The report of the game.

        """
        color_names = {
            Color.WHITE: "White",
            Color.BLACK: "Black",
        }

        # sending the current board to the gui
        self.game.notify_display()

        # match controller is the single async orchestration hub
        if self.game.is_play():
            controller.start()

        while True:
            state = self.game.state
            ply: Ply = self.game.ply
            chipiron_logger.info(
                "Half Move: %s playing status %s",
                ply,
                self.game.playing_status.status,
            )
            color_to_move: Color = state.turn
            color_of_player_to_move_str = color_names[color_to_move]
            chipiron_logger.info(
                "%s (%s) to play now...",
                color_of_player_to_move_str,
                self.player_color_to_id[color_to_move],
            )

            # waiting for a message
            mail = self.main_thread_mailbox.get()
            if isinstance(mail, GuiCommand):
                if self._ignore_if_stale_scope(mail.scope):
                    continue
                match mail.payload:
                    case HumanActionChosen():
                        controller.handle_human_action(mail.payload)
                    case _:
                        self._handle_gui_command(mail)
            elif isinstance(mail, PlayerEvent):
                if self._ignore_if_stale_scope(mail.scope):
                    continue
                match mail.payload:
                    case EvMove():
                        controller.handle_player_action(mail.payload)
                    case _:
                        self._handle_player_event(mail)
            else:
                assert_never(mail)

            state = self.game.state
            is_terminal = self.rules.outcome(state) is not None
            if is_terminal or not self.game_continue_conditions():
                if is_terminal:
                    chipiron_logger.info("The game is over")
                if not self.game_continue_conditions():
                    chipiron_logger.info("Game continuation not met")
                break
            chipiron_logger.info("Not game over at %s", state)

        self.tell_results()
        self.terminate_processes()
        chipiron_logger.info("End play_one_game")

        game_results: FinalGameResult = self.simple_results()

        game_report: GameReport = GameReport(
            final_game_result=game_results,
            action_history=list(self.game.action_history),
            state_tag_history=self.game.state_tag_history,
        )
        return game_report

    def _ignore_if_stale_scope(self, scope: object) -> bool:
        if scope != self.game.scope:
            chipiron_logger.debug(
                "Ignoring stale message scope=%s (current=%s)",
                scope,
                self.game.scope,
            )
            return True
        return False

    def _handle_gui_command(self, message: GuiCommand) -> None:
        state: StateT = self.game.state

        match message.payload:
            case CmdSetStatus():
                if message.payload.status == PlayingStatus.PLAY:
                    self.game.set_play_status()
                elif message.payload.status == PlayingStatus.PAUSE:
                    self.game.set_pause_status()
                else:
                    chipiron_logger.warning(  # type: ignore[unreachable]
                        "Unhandled PlayingStatus: %s", message.payload.status
                    )

            case CmdBackOneMove():
                self.game.set_pause_status()
                self.rewind_one_move()

            case _:
                chipiron_logger.warning(  # type: ignore[unreachable]
                    "Unhandled GuiCommand payload in file %s: %r",
                    __name__,
                    message.payload,
                )
                return

    def _handle_player_event(self, message: PlayerEvent) -> None:
        match message.payload:
            case EvMove():
                chipiron_logger.debug("EvMove is orchestrated by MatchController; ignoring in GameManager")
                return
            case EvProgress():
                # forward to your progress collector
                if message.payload.player_color == Color.WHITE:
                    self.progress_collector.progress_white(
                        value=message.payload.progress_percent
                    )
                else:
                    self.progress_collector.progress_black(
                        value=message.payload.progress_percent
                    )
            case _:
                chipiron_logger.warning(  # type: ignore[unreachable]
                    "Unhandled PlayerEvent payload in file %s: %r",
                    __name__,
                    message.payload,
                )
                return

    def game_continue_conditions(self) -> bool:
        """Check the conditions for continuing the game.

        Returns:
            bool: True if the game should continue, False otherwise.

        """
        ply: Ply = self.game.ply
        continue_bool: bool = True
        if self.args.max_half_moves is not None and ply >= self.args.max_half_moves:
            continue_bool = False

        return continue_bool

    def print_to_file(self, game_report: GameReport, idx: int = 0) -> None:
        """Print the moves of the game to a yaml file and a more human-readable text file.

        Args:
            game_report(GameReport): a game report to be printed
            idx (int): The index to include in the file name (default is 0).

        Returns:
            None

        """
        # TODO: probably the txt file should be a valid PGN file : https://en.wikipedia.org/wiki/Portable_Game_Notation
        if self.path_to_store_result is not None:
            path_file: MyPath = (
                f"{self.path_to_store_result}_{idx}_W:{self.player_color_to_id[Color.WHITE]}"
                f"-vs-B:{self.player_color_to_id[Color.BLACK]}"
            )
            path_file_obj = f"{path_file}_game_report.yaml"
            path_file_txt = f"{path_file}.txt"
            with open(path_file_txt, "a", encoding="utf-8") as file_text:
                move_1 = None
                for counter, move in enumerate(self.game.action_history):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        if move_1 is not None:
                            file_text.write(str(move_1) + " " + str(move) + "\n")
            with open(path_file_obj, "w", encoding="utf-8") as file:
                yaml.dump(
                    asdict(game_report, dict_factory=custom_asdict_factory),
                    file,
                    default_flow_style=False,
                )

    def _handle_move_attempt(
        self,
        *,
        source: Literal["player", "gui"],
        branch_name: str,  # stable transport action name (for chess, this is UCI)
        corresponding_state_tag: StateTag,
        player_name: str,
        color_to_play: Color,
        evaluation: StateEvaluation | None,  # your real type
    ) -> None:
        """Single move-application path used by BOTH GUI and players.

        Keeps the stricter checks/logging from the player path.
        """
        state: StateT = self.game.state

        chipiron_logger.info(
            "[%s] MOVE ATTEMPT: move=%s player=%s color_to_play=%s is_play=%s current_tag=%s msg_tag=%s",
            source,
            branch_name,
            player_name,
            color_to_play,
            self.game.playing_status.is_play(),
            state.tag,
            corresponding_state_tag,
        )

        # --- check 1: scope is handled outside (caller ensures correct scope) ---

        # --- check 2: current position must match ---
        fen_ok = corresponding_state_tag == state.tag

        # --- check 3: must be in PLAY mode ---
        play_ok = self.game.playing_status.is_play()

        # --- check 4: the sender must match "who should play" ---
        expected_player = self.player_color_to_id[state.turn]
        player_ok = player_name == expected_player

        if not (fen_ok and play_ok and player_ok):
            chipiron_logger.info(
                "[%s] MOVE REJECTED: fen_ok=%s play_ok=%s player_ok=%s expected_player=%s",
                source,
                fen_ok,
                play_ok,
                player_ok,
                expected_player,
            )
            return

        # Ensure legal moves generated
        try:
            self.game.dynamics.legal_actions(state).get_all()
        except (RuntimeError, ValueError) as exc:
            chipiron_logger.info(
                "[%s] MOVE REJECTED: failed to generate legal moves",
                source,
                exc_info=exc,
            )
            return

        # Convert name->ActionKey
        try:
            action_key = self.game.dynamics.action_from_name(state, branch_name)
        except (KeyError, ValueError) as exc:
            chipiron_logger.info(
                "[%s] MOVE REJECTED: invalid branch_name=%s",
                source,
                branch_name,
                exc_info=exc,
            )
            return

        chipiron_logger.info("[%s] MOVE ACCEPTED: %s", source, branch_name)

        self.play_one_move(action_key)

        # optional: store evaluation
        if evaluation is not None:
            self.display_state_evaluator.add_evaluation(
                player_color=color_to_play,
                evaluation=evaluation,
            )


    def tell_results(self) -> None:
        """Print the results of the game based on the current state of the board.

        Returns:
            None

        """
        state = self.game.state
        outcome = self.rules.outcome(state)
        if outcome is None:
            chipiron_logger.info("No terminal outcome available to report.")
        else:
            chipiron_logger.info(self.rules.pretty_result(state, outcome))
        assessment = self.rules.assessment(state)
        if assessment is not None:
            chipiron_logger.info(self.rules.pretty_assessment(state, assessment))

    def simple_results(self) -> FinalGameResult:
        """Determine the final result of the game based on the current state of the board.

        Returns:
            FinalGameResult: The final result of the game.

        """
        state = self.game.state
        outcome = self.rules.outcome(state)
        if outcome is None:
            outcome = GameOutcome(
                kind=OutcomeKind.UNKNOWN,
                reason="no_terminal_outcome",
                source=OutcomeSource.TERMINAL,
            )
        return outcome_to_final_game_result(outcome)

    def terminate_processes(self) -> None:
        """Terminates all player processes and stops the associated threads.

        This method iterates over the list of players and terminates any player processes found.
        If a player process is found, it is terminated and the associated thread is stopped.

        Note:
        - This method assumes that the `players` attribute is a list of `PlayerProcess` or `GamePlayer` objects.

        Returns:
        None

        """
        for player in self.players:
            player.close()
