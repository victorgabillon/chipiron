"""
Module in charge of managing the game. It is the main class that will be used to play a game.
"""

import os
import queue
from dataclasses import asdict
from typing import TypeAlias

import yaml
from atomheart.move_factory import MoveFactory
from valanga import Color, StateEvaluation, StateTag, TurnState
from valanga.game import ActionKey

import chipiron.players as players_m
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.boardevaluators.board_evaluator import IGameStateEvaluator
from chipiron.players.boardevaluators.table_base.factory import AnySyzygyTable
from chipiron.players.communications.player_message import (
    EvMove,
    EvProgress,
    PlayerEvent,
)
from chipiron.utils import path
from chipiron.utils.communication.gui_messages.gui_messages import (
    CmdBackOneMove,
    CmdHumanMoveUci,
    CmdSetStatus,
    GuiCommand,
)
from chipiron.utils.dataclass import IsDataclass, custom_asdict_factory
from chipiron.utils.logger import chipiron_logger

from .final_game_result import FinalGameResult, GameReport
from .game import ObservableGame, Ply
from .game_args import GameArgs
from .progress_collector import PlayerProgressCollectorP

MainMailboxMessage: TypeAlias = GuiCommand | PlayerEvent


class GameManager[StateT: TurnState = TurnState]:
    """
    Object in charge of playing one game
    """

    # The game object that is managed
    game: ObservableGame[StateT]

    # A SyzygyTable
    syzygy: AnySyzygyTable | None

    # Evaluators that just evaluates the boards but are not players (just spectators) for display info of who is winning
    # according to them
    display_state_evaluator: IGameStateEvaluator[StateT]

    # folder to log results
    output_folder_path: path | None

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

    def __init__(
        self,
        game: ObservableGame[StateT],
        syzygy: AnySyzygyTable | None,
        display_state_evaluator: IGameStateEvaluator[StateT],
        output_folder_path: path | None,
        args: GameArgs,
        player_color_to_id: dict[Color, str],
        main_thread_mailbox: queue.Queue[MainMailboxMessage],
        players: list[players_m.PlayerHandle],
        move_factory: MoveFactory,
        progress_collector: PlayerProgressCollectorP,
    ) -> None:
        """
        Constructor for the GameManager Class. If the args, and players are not given a value it is set to None,
        waiting for the set methods to be called. This is done like this so that the players can be changed
        (often swapped) easily.

        Args:
            game (ObservableGame): The observable game object.
            syzygy (AnySyzygyTable | None): The syzygy table object or None.
            display_board_evaluator (IGameBoardEvaluator): The board evaluator to display an independent evaluation.
            output_folder_path (path | None): The output folder path or None.
            args (GameArgs): The game arguments.
            player_color_to_id (dict[chess.Color, str]): The dictionary mapping player color to player ID.
            main_thread_mailbox (queue.Queue[IsDataclass]): The main thread mailbox.
            players (list[players_m.PlayerProcess | players_m.GamePlayer]): The list of players.

        Returns:
            None
        """
        self.game = game
        self.syzygy = syzygy
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

    def external_eval(self) -> tuple[float | None, float]:
        """Evaluates the game board using the display board evaluator.

        Returns:
            tuple[float, float]: A tuple containing the evaluation scores.
        """
        return self.display_state_evaluator.evaluate(self.game.state)

    def play_one_move(self, action: ActionKey) -> None:
        """Play one move in the game.

        Args:
            action (ActionKey): The action to be played.
        """
        self.game.play_move(action)
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.state):
            chipiron_logger.info(
                "Theoretically finished with value for white: %s",
                self.syzygy.string_result(self.game.state),
            )

    def rewind_one_move(self) -> None:
        """
        Rewinds the game by one move.

        This method rewinds the game by one move, undoing the last move made.
        If the game has a Syzygy tablebase loaded and the current board position is in the tablebase,
        it prints the theoretically finished value for white.

        Returns:
            None
        """
        self.game.rewind_one_move()
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.state):
            chipiron_logger.info(
                "Theoretically finished with value for white: %s",
                self.syzygy.string_result(self.game.state),
            )

    def play_one_game(self) -> GameReport:
        """
        Play one game.

        Returns:
            GameReport: The report of the game.
        """

        color_names = ["Black", "White"]

        # sending the current board to the gui
        self.game.notify_display()

        # sending the current board to the player and asking for a move
        if self.game.is_play():
            self.game.query_move_from_players()

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
            self.processing_mail(mail)

            state = self.game.state
            if state.is_game_over() or not self.game_continue_conditions():
                if state.is_game_over():
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
            action_history=[move for move in self.game.action_history],
            state_tag_history=self.game.state_tag_history,
        )
        return game_report

    def processing_mail(self, message: IsDataclass) -> None:
        """
        Process the incoming mail message.

        Args:
            message (IsDataclass): The incoming mail message.

        Returns:
            None
        """

        state: TurnState = self.game.state

        match message:
            case GuiCommand():
                if message.scope != self.game.scope:
                    chipiron_logger.debug(
                        "Ignoring stale GuiCommand for scope=%s (current=%s)",
                        message.scope,
                        self.game.scope,
                    )
                    return

                match message.payload:
                    case CmdSetStatus():
                        if message.payload.status == PlayingStatus.PLAY:
                            self.game.set_play_status()
                            self.game.query_move_from_players()
                        elif message.payload.status == PlayingStatus.PAUSE:
                            self.game.set_pause_status()
                        else:
                            chipiron_logger.warning(
                                "Unhandled PlayingStatus: %s", message.payload.status
                            )

                    case CmdBackOneMove():
                        self.game.set_pause_status()
                        self.rewind_one_move()

                    case CmdHumanMoveUci():
                        # Convert GUI move into the SAME move pipeline as players
                        ev = EvMove(
                            branch_name=message.payload.move_uci,
                            corresponding_state_tag=message.payload.corresponding_fen,
                            player_name=self.player_color_to_id[
                                state.turn
                            ],  # or message.payload.player_name if you have it
                            color_to_play=state.turn,
                            evaluation=None,
                        )
                        self._handle_move_attempt(
                            source="gui",
                            branch_name=ev.branch_name,
                            corresponding_state_tag=ev.corresponding_state_tag,
                            player_name=ev.player_name,
                            color_to_play=ev.color_to_play,
                            evaluation=ev.evaluation,
                        )

                    case _:
                        raise ValueError(
                            f"Unhandled GuiCommand payload in file {__name__}: {message.payload!r}"
                        )

            case PlayerEvent():
                if message.scope != self.game.scope:
                    chipiron_logger.debug(
                        "Ignoring stale PlayerEvent for scope=%s (current=%s)",
                        message.scope,
                        self.game.scope,
                    )
                    return

                match message.payload:
                    case EvMove():
                        self._handle_move_attempt(
                            source="player",
                            branch_name=message.payload.branch_name,
                            corresponding_state_tag=message.payload.corresponding_state_tag,
                            player_name=message.payload.player_name,
                            color_to_play=message.payload.color_to_play,
                            evaluation=message.payload.evaluation,
                        )
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
                        raise ValueError(
                            f"Unhandled PlayerEvent payload in {__name__}: {message.payload!r}"
                        )

            case _:
                raise ValueError(f"Unexpected message type in {__name__}: {message!r}")

    def game_continue_conditions(self) -> bool:
        """
        Checks the conditions for continuing the game.

        Returns:
            bool: True if the game should continue, False otherwise.
        """
        ply: Ply = self.game.ply
        continue_bool: bool = True
        if self.args.max_half_moves is not None and ply >= self.args.max_half_moves:
            continue_bool = False

        return continue_bool

    def print_to_file(self, game_report: GameReport, idx: int = 0) -> None:
        """
        Print the moves of the game to a yaml file and a more human-readable text file.

        Args:
            game_report(GameReport): a game report to be printed
            idx (int): The index to include in the file name (default is 0).

        Returns:
            None
        """
        # todo probably the txt file should be a valid PGN file : https://en.wikipedia.org/wiki/Portable_Game_Notation
        if self.path_to_store_result is not None:
            path_file: path = (
                f"{self.path_to_store_result}_{idx}_W:{self.player_color_to_id[Color.WHITE]}"
                f"-vs-B:{self.player_color_to_id[Color.BLACK]}"
            )
            path_file_obj = f"{path_file}_game_report.yaml"
            path_file_txt = f"{path_file}.txt"
            with open(path_file_txt, "a", encoding="utf-8") as the_fileText:
                move_1 = None
                for counter, move in enumerate(self.game.action_history):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        if move_1 is not None:
                            the_fileText.write(str(move_1) + " " + str(move) + "\n")
            with open(path_file_obj, "w", encoding="utf-8") as file:
                yaml.dump(
                    asdict(game_report, dict_factory=custom_asdict_factory),
                    file,
                    default_flow_style=False,
                )

    def _handle_move_attempt(
        self,
        *,
        source: str,  # "player" or "gui"
        branch_name: str,  # stable transport action name (for chess, this is UCI)
        corresponding_state_tag: StateTag,  # StateTag/Fen
        player_name: str,
        color_to_play: Color,
        evaluation: StateEvaluation | None,  # your real type
    ) -> None:
        """
        Single move-application path used by BOTH GUI and players.
        Keeps the stricter checks/logging from the player path.
        """
        state: TurnState = self.game.state

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
        state.branch_keys.get_all()

        # Convert name->ActionKey
        action_key: ActionKey = state.branch_key_from_name(name=branch_name)

        chipiron_logger.info("[%s] MOVE ACCEPTED: %s", source, branch_name)

        self.play_one_move(action_key)

        # optional: store evaluation
        if evaluation is not None:
            self.display_state_evaluator.add_evaluation(
                player_color=color_to_play,
                evaluation=evaluation,
            )

        # query next move if still playing and game not over
        if self.game.is_play() and not self.game.state.is_game_over():
            self.game.query_move_from_players()

    def tell_results(self) -> None:
        """
        Prints the results of the game based on the current state of the board.

        The method checks various conditions on the board and prints the corresponding result.
        It also checks for specific conditions like syzygy, fivefold repetition, seventy-five moves,
        insufficient material, stalemate, and checkmate.

        Returns:
            None
        """
        state = self.game.state
        if self.syzygy is not None and self.syzygy.fast_in_table(state):
            chipiron_logger.info(
                "Syzygy: Theoretical value for white %s",
                self.syzygy.string_result(state),
            )
        state.tell_result()

    def simple_results(self) -> FinalGameResult:
        """
        Determines the final result of the game based on the current state of the board.

        Returns:
            FinalGameResult: The final result of the game.
        """
        board = self.game.state

        res: FinalGameResult | None = None
        result: str = board.result(claim_draw=True)
        if result == "*":
            if self.syzygy is None or not self.syzygy.fast_in_table(board):
                # useful when a game is stopped
                # before the end, for instance for debugging and profiling
                res = FinalGameResult.DRAW  # arbitrary meaningless choice
                # raise ValueError(f'Problem with figuring our game results in {__name__}')
            else:
                raise ValueError(
                    "this case is not coded atm think of what is the right thing to do here!"
                )
        else:
            match result:
                case "1/2-1/2":
                    res = FinalGameResult.DRAW
                case "0-1":
                    res = FinalGameResult.WIN_FOR_BLACK
                case "1-0":
                    res = FinalGameResult.WIN_FOR_WHITE
                case other:
                    raise ValueError(
                        f"unexpected result value {other} in game manager/simple_results"
                    )

        assert isinstance(res, FinalGameResult)
        return res

    def terminate_processes(self) -> None:
        """
        Terminates all player processes and stops the associated threads.

        This method iterates over the list of players and terminates any player processes found.
        If a player process is found, it is terminated and the associated thread is stopped.

        Note:
        - This method assumes that the `players` attribute is a list of `PlayerProcess` or `GamePlayer` objects.

        Returns:
        None
        """
        for player in self.players:
            player.close()
