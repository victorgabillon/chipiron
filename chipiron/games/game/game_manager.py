"""
Module in charge of managing the game. It is the main class that will be used to play a game.
"""

import os
import queue
from dataclasses import asdict
from typing import Any

import chess
import yaml

import chipiron.players as players_m
from chipiron.environments import HalfMove
from chipiron.environments.chess.board.iboard import IBoard
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.environments.chess.move_factory import MoveFactory
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.boardevaluators.board_evaluator import IGameBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.utils import path
from chipiron.utils.communication.gui_messages import (
    BackMessage,
    GameStatusMessage,
    PlayerProgressMessage,
)
from chipiron.utils.communication.player_game_messages import MoveMessage
from chipiron.utils.dataclass import IsDataclass, custom_asdict_factory
from chipiron.utils.logger import chipiron_logger

from .final_game_result import FinalGameResult, GameReport
from .game import ObservableGame
from .game_args import GameArgs
from .progress_collector import PlayerProgressCollectorP


class GameManager:
    """
    Object in charge of playing one game
    """

    # The game object that is managed
    game: ObservableGame

    # A SyzygyTable
    syzygy: SyzygyTable[Any] | None

    # Evaluators that just evaluates the boards but are not players (just spectators) for display info of who is winning
    # according to them
    display_board_evaluator: IGameBoardEvaluator

    # folder to log results
    output_folder_path: path | None

    # args of the Game
    args: GameArgs

    # Dictionary mapping colors to player names?
    player_color_to_id: dict[chess.Color, str]

    # A Queue for receiving messages from other process or functions such as players or Gui
    main_thread_mailbox: queue.Queue[IsDataclass]

    # The list of players
    players: list[players_m.PlayerProcess | players_m.GamePlayer]

    # A move factory
    move_factory: MoveFactory

    # an object for collecting how advances each player is in its thinking/computation of the moves
    progress_collector: PlayerProgressCollectorP

    def __init__(
        self,
        game: ObservableGame,
        syzygy: SyzygyTable[Any] | None,
        display_board_evaluator: IGameBoardEvaluator,
        output_folder_path: path | None,
        args: GameArgs,
        player_color_to_id: dict[chess.Color, str],
        main_thread_mailbox: queue.Queue[IsDataclass],
        players: list[players_m.PlayerProcess | players_m.GamePlayer],
        move_factory: MoveFactory,
        progress_collector: PlayerProgressCollectorP,
    ) -> None:
        """
        Constructor for the GameManager Class. If the args, and players are not given a value it is set to None,
        waiting for the set methods to be called. This is done like this so that the players can be changed
        (often swapped) easily.

        Args:
            game (ObservableGame): The observable game object.
            syzygy (SyzygyTable | None): The syzygy table object or None.
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
        self.display_board_evaluator = display_board_evaluator
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
        return self.display_board_evaluator.evaluate(self.game.board)

    def play_one_move(self, move: moveKey) -> None:
        """Play one move in the game.

        Args:
            move (chess.Move): The move to be played.
        """
        self.game.play_move(move)
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.board):
            chipiron_logger.info(
                f"Theoretically finished with value for white: {self.syzygy.string_result(self.game.board)}",
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
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.board):
            chipiron_logger.info(
                f"Theoretically finished with value for white: {self.syzygy.string_result(self.game.board)}",
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

            board = self.game.board
            half_move: HalfMove = board.ply()
            chipiron_logger.info(
                f"Half Move: {half_move} playing status {self.game.playing_status.status} "
            )
            color_to_move: chess.Color = board.turn
            color_of_player_to_move_str = color_names[color_to_move]
            chipiron_logger.info(
                f"{color_of_player_to_move_str} ({self.player_color_to_id[color_to_move]}) to play now..."
            )

            # waiting for a message
            mail = self.main_thread_mailbox.get()
            self.processing_mail(mail)

            board = self.game.board
            if board.is_game_over() or not self.game_continue_conditions():
                if board.is_game_over():
                    chipiron_logger.info("The game is other")
                if not self.game_continue_conditions():
                    chipiron_logger.info("Game continuation not met")
                break
            else:
                chipiron_logger.info(f"Not game over at {board}")

        self.tell_results()
        self.terminate_processes()
        chipiron_logger.info("End play_one_game")

        game_results: FinalGameResult = self.simple_results()

        game_report: GameReport = GameReport(
            final_game_result=game_results,
            move_history=[move for move in self.game.move_history],
            fen_history=self.game.fen_history,
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

        board: IBoard = self.game.board

        match message:
            case MoveMessage():
                chipiron_logger.info(
                    "=====================MOVE MESSAGE RECEIVED============"
                )
                move_message: MoveMessage = message
                # play the move
                move_uci: moveUci = move_message.move

                chipiron_logger.info(
                    f"Game Manager: Receiving the move uci {move_uci} {self.game.playing_status} {board.fen}"
                )
                if (
                    move_message.corresponding_board == board.fen
                    and self.game.playing_status.is_play()
                    and message.player_name == self.player_color_to_id[board.turn]
                ):

                    board.legal_moves.get_all()  # make sure the board has generated the legal moves

                    move_key: moveKey = board.get_move_key_from_uci(move_uci=move_uci)

                    chipiron_logger.info(
                        f"Game Manager: Play a move {move_uci} at {board} {self.game.board.fen}"
                    )
                    # move: IMove = self.move_factory(move_uci=move_uci, board=board)
                    self.play_one_move(move_key)
                    chipiron_logger.info(
                        f"Game Manager: Now board is  {self.game.board}"
                    )

                    eval_sto, eval_chi = self.external_eval()
                    chipiron_logger.info(
                        f"Stockfish evaluation:{eval_sto} and chipiron eval{eval_chi}"
                    )
                    # Print the board
                    chipiron_logger.info(board.print_chess_board())

                    # sending the current board to the player  and asking for a move
                    if self.game.is_play():
                        self.game.query_move_from_players()

                else:
                    chipiron_logger.info(
                        f"the move is rejected because one of the following is false \n"
                        f" move_message.corresponding_board == board.fen{move_message.corresponding_board == board.fen} \n"
                        f"self.game.playing_status.is_play() {self.game.playing_status.is_play()}\n"
                        f"message.player_name == self.player_color_to_id[board.turn] {message.player_name == self.player_color_to_id[board.turn]}"
                    )
                    chipiron_logger.info(
                        f"{message.player_name},{self.player_color_to_id[board.turn]}"
                    )
                if message.evaluation is not None:
                    self.display_board_evaluator.add_evaluation(
                        player_color=message.color_to_play,
                        evaluation=message.evaluation,
                    )
            case PlayerProgressMessage():
                player_progress_message: PlayerProgressMessage = message
                if player_progress_message.player_color == chess.WHITE:
                    self.progress_collector.progress_white(
                        value=player_progress_message.progress_percent
                    )
                if player_progress_message.player_color == chess.BLACK:
                    self.progress_collector.progress_black(
                        value=player_progress_message.progress_percent
                    )
            case GameStatusMessage():
                game_status_message: GameStatusMessage = message
                # update game status
                if game_status_message.status == PlayingStatus.PLAY:
                    self.game.set_play_status()
                    self.game.query_move_from_players()

                if game_status_message.status == PlayingStatus.PAUSE:
                    self.game.set_pause_status()
            case BackMessage():
                self.game.set_pause_status()
                self.rewind_one_move()

            case other:
                raise ValueError(
                    f"type of message received is not recognized {other} in file {__name__}"
                )

    def game_continue_conditions(self) -> bool:
        """
        Checks the conditions for continuing the game.

        Returns:
            bool: True if the game should continue, False otherwise.
        """
        half_move: HalfMove = self.game.board.ply()
        continue_bool: bool = True
        if (
            self.args.max_half_moves is not None
            and half_move >= self.args.max_half_moves
        ):
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
                f"{self.path_to_store_result}_{idx}_W:{self.player_color_to_id[chess.WHITE]}"
                f"-vs-B:{self.player_color_to_id[chess.BLACK]}"
            )
            path_file_obj = f"{path_file}_game_report.yaml"
            path_file_txt = f"{path_file}.txt"
            with open(path_file_txt, "a") as the_fileText:
                for counter, move in enumerate(self.game.move_history):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + " " + str(move) + "\n")
            with open(path_file_obj, "w") as file:
                yaml.dump(
                    asdict(game_report, dict_factory=custom_asdict_factory),
                    file,
                    default_flow_style=False,
                )

    def tell_results(self) -> None:
        """
        Prints the results of the game based on the current state of the board.

        The method checks various conditions on the board and prints the corresponding result.
        It also checks for specific conditions like syzygy, fivefold repetition, seventy-five moves,
        insufficient material, stalemate, and checkmate.

        Returns:
            None
        """
        board = self.game.board
        if self.syzygy is not None and self.syzygy.fast_in_table(board):
            chipiron_logger.info(
                f"Syzygy: Theoretical value for white {self.syzygy.string_result(board)}"
            )
        board.tell_result()

    def simple_results(self) -> FinalGameResult:
        """
        Determines the final result of the game based on the current state of the board.

        Returns:
            FinalGameResult: The final result of the game.
        """
        board = self.game.board

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
                    raise Exception(
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
        player: players_m.PlayerProcess | players_m.GamePlayer
        for player in self.players:
            if isinstance(player, players_m.PlayerProcess):
                player.terminate()
                chipiron_logger.info("Stopping the thread")
                chipiron_logger.info("Thread stopped")
