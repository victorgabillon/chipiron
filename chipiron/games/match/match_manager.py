"""
Module in charge of playing one match
"""

import os
import queue

import chess

from chipiron.environments.chess.move import moveUci
from chipiron.games.game.final_game_result import GameReport
from chipiron.games.game.game_args import GameArgs
from chipiron.games.game.game_args_factory import GameArgsFactory
from chipiron.games.game.game_manager import GameManager
from chipiron.games.game.game_manager_factory import GameManagerFactory
from chipiron.games.match.match_results import IMatchResults, MatchReport, MatchResults
from chipiron.games.match.match_results_factory import MatchResultsFactory
from chipiron.games.match.observable_match_result import ObservableMatchResults
from chipiron.players import PlayerFactoryArgs
from chipiron.utils import path, seed
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger


class MatchManager:
    """
    Object in charge of playing one match

    Args:
        player_one_id (str): The ID of player one.
        player_two_id (str): The ID of player two.
        game_manager_factory (GameManagerFactory): The factory for creating game managers.
        game_args_factory (GameArgsFactory): The factory for creating game arguments.
        match_results_factory (MatchResultsFactory): The factory for creating match results.
        output_folder_path (path | None, optional): The path to the output folder. Defaults to None.
    """

    def __init__(
        self,
        player_one_id: str,
        player_two_id: str,
        game_manager_factory: GameManagerFactory,
        game_args_factory: GameArgsFactory,
        match_results_factory: MatchResultsFactory,
        output_folder_path: path | None = None,
    ) -> None:
        """Initialize a MatchManager object.

        Args:
            player_one_id (str): The ID of player one.
            player_two_id (str): The ID of player two.
            game_manager_factory (GameManagerFactory): The factory object for creating game managers.
            game_args_factory (GameArgsFactory): The factory object for creating game arguments.
            match_results_factory (MatchResultsFactory): The factory object for creating match results.
            output_folder_path (path | None, optional): The path to the output folder. Defaults to None.
        """
        self.player_one_id = player_one_id
        self.player_two_id = player_two_id
        self.game_manager_factory = game_manager_factory
        self.output_folder_path = output_folder_path
        self.match_results_factory = match_results_factory
        self.game_args_factory = game_args_factory
        self.print_info()

    def print_info(self) -> None:
        """Prints the information about the players in the match.

        This method prints the IDs of player one and player two.

        Parameters:
            None

        Returns:
            None
        """
        chipiron_logger.info(
            f"player one is {self.player_one_id}",
        )
        chipiron_logger.info(
            f"player two is {self.player_two_id}",
        )

    def play_one_match(self) -> MatchReport:
        """Plays one match and returns the match report.

        This method plays a single match, which consists of multiple games. It generates game arguments,
        plays each game, updates the match results, and saves the match report to a file.

        Returns:
            MatchReport: The report of the match, including the move history and match results.
        """
        chipiron_logger.info("Playing the match")

        # creating object for reporting the result of the match and the move history
        match_results: IMatchResults = self.match_results_factory.create()
        match_move_history: dict[int, list[moveUci]] = {}

        # Main loop of playing various games
        game_number: int = 0
        while not self.game_args_factory.is_match_finished():
            args_game: GameArgs
            player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs]
            game_seed: seed | None
            player_color_to_factory_args, args_game, game_seed = (
                self.game_args_factory.generate_game_args(game_number)
            )

            assert game_seed is not None
            # Play one game
            game_report: GameReport = self.play_one_game(
                player_color_to_factory_args=player_color_to_factory_args,
                args_game=args_game,
                game_number=game_number,
                game_seed=game_seed,
            )

            # Update the reporting of the ongoing match with the report of the finished game
            match_results.add_result_one_game(
                white_player_name_id=player_color_to_factory_args[
                    chess.WHITE
                ].player_args.name,
                game_result=game_report.final_game_result,
            )
            match_move_history[game_number] = game_report.move_history

            # ad hoc waiting time in case we play against a human and the game is finished
            # (so that the human as the time to view the final position before the automatic start of a new game)
            if player_color_to_factory_args[chess.WHITE].player_args.is_human():
                import time

                time.sleep(30)

            game_number += 1

        chipiron_logger.info(match_results)
        self.print_stats_to_file(match_results=match_results)

        # setting  officially the game to finished state (some subscribers might receive this event as a message,
        # when a gui is present it might action it to close itself)
        match_results.finish()

        if not isinstance(match_results, MatchResults):
            assert isinstance(match_results, ObservableMatchResults)
            match_results = match_results.match_results
        assert isinstance(match_results, MatchResults)
        match_report: MatchReport = MatchReport(
            match_move_history=match_move_history, match_results=match_results
        )

        self.save_match_report_to_file(match_report)

        return match_report

    def play_one_game(
        self,
        player_color_to_factory_args: dict[chess.Color, PlayerFactoryArgs],
        args_game: GameArgs,
        game_number: int,
        game_seed: seed,
    ) -> GameReport:
        """Plays one game and returns the game report.

        Args:
            player_color_to_factory_args (dict[chess.Color, PlayerFactoryArgs]): A dictionary mapping player colors to their factory arguments.
            args_game (GameArgs): The arguments for the game.
            game_number (int): The number of the game.
            game_seed (seed): The seed for the game.

        Returns:
            GameReport: The report of the game.
        """
        game_manager: GameManager = self.game_manager_factory.create(
            args_game_manager=args_game,
            player_color_to_factory_args=player_color_to_factory_args,
            game_seed=game_seed,
        )
        game_report: GameReport = game_manager.play_one_game()
        game_manager.print_to_file(idx=game_number, game_report=game_report)

        return game_report

    def print_stats_to_file(self, match_results: IMatchResults) -> None:
        """Prints the match statistics to a file.

        Args:
            match_results (IMatchResults): The match results object containing the statistics.
        """
        if self.output_folder_path is not None:
            path_file: path = os.path.join(self.output_folder_path, "gameStats.txt")
            with open(path_file, "a") as the_file:
                the_file.write(str(match_results))

    def save_match_report_to_file(self, match_report: MatchReport) -> None:
        """Save the match report to a file.

        Args:
            match_report (MatchReport): The match report to be saved.
        """
        if self.output_folder_path is not None:
            ...
            # path_file: path = os.path.join(self.output_folder_path, 'match_report.obj')
            # with open(path_file, 'wb') as the_file:
            # print('tt', type(match_report))
            # pickle.dump(match_report, the_file)

    def subscribe(self, subscriber: queue.Queue[IsDataclass]) -> None:
        """Subscribe a subscriber to receive updates from the match manager.

        Args:
            subscriber (queue.Queue[IsDataclass]): The subscriber to be added to the list of subscribers.
        """
        self.game_manager_factory.subscribe(subscriber)
        self.match_results_factory.subscribe(subscriber)
