import chess
from chipiron.games.game.game_manager_factory import GameManagerFactory
from chipiron.games.game.game_manager import GameManager, GameReport
from chipiron.games.match.match_results import MatchResults, MatchReport
from chipiron.games.game.game_args import GameArgs
from chipiron.players import Player
from chipiron.games.match.match_results_factory import MatchResultsFactory
from chipiron.utils import path
import pickle
import os
from chipiron.utils import seed


class MatchManager:
    """
    Objet in charge of playing one match
    """

    def __init__(self,
                 player_one_id: str,
                 player_two_id: str,
                 game_manager_factory: GameManagerFactory,
                 game_args_factory,
                 match_results_factory: MatchResultsFactory,
                 output_folder_path=None):
        self.player_one_id = player_one_id
        self.player_two_id = player_two_id
        self.game_manager_factory = game_manager_factory
        self.output_folder_path = output_folder_path
        self.match_results_factory = match_results_factory
        self.game_args_factory = game_args_factory
        self.print_info()

    def print_info(self):
        print('player one is ', self.player_one_id)
        print('player two is ', self.player_two_id)

    def play_one_match(
            self
    ) -> MatchReport:
        """Playing one game"""
        print('Playing the match')

        # creating object for reporting the result of the match and the move history
        match_results: MatchResults = self.match_results_factory.create()
        match_move_history: dict[int, list[chess.Move]] = {}

        # Main loop of playing various games
        game_number: int = 0
        while not self.game_args_factory.is_match_finished():
            args_game: GameArgs
            player_color_to_player: dict[chess.Color, Player]
            game_seed: seed
            player_color_to_player, args_game, game_seed = self.game_args_factory.generate_game_args(game_number)

            # Play one game
            game_report: GameReport = self.play_one_game(
                player_color_to_player=player_color_to_player,
                args_game=args_game,
                game_number=game_number,
                game_seed=game_seed
            )

            # Update the reporting of the ongoing match with the report of the finished game
            match_results.add_result_one_game(
                white_player_name_id=player_color_to_player[chess.WHITE].id,
                game_result=game_report.final_game_result
            )
            match_move_history[game_number] = game_report.move_history

            # ad hoc waiting time in case we play against a human and the game is finished
            # (so that the human as the time to view the final position before the automatic start of a new game)
            if player_color_to_player[chess.WHITE].id == 'Human' or player_color_to_player[chess.BLACK].id == 'Human':
                import time
                time.sleep(30)

            game_number += 1

        print(match_results)
        self.print_stats_to_file(match_results=match_results)

        # setting  officially the game to finished state (some subscribers might receive this event as a message,
        # when a gui is present it might action it to close itself)
        match_results.finish()

        match_report: MatchReport = MatchReport(
            match_move_history=match_move_history,
            match_results=match_results
        )

        self.save_match_report_to_file(match_report)

        return match_report

    def play_one_game(
            self,
            player_color_to_player: dict[chess.Color, Player],
            args_game: GameArgs,
            game_number: int,
            game_seed: seed
    ) -> GameReport:
        game_manager: GameManager = self.game_manager_factory.create(
            args_game_manager=args_game,
            player_color_to_player=player_color_to_player,
            game_seed=game_seed
        )
        game_report: GameReport = game_manager.play_one_game()
        game_manager.print_to_file(idx=game_number)

        return game_report

    def print_stats_to_file(
            self,
            match_results: MatchResults
    ) -> None:
        if self.output_folder_path is not None:
            path_file: path = os.path.join(self.output_folder_path, 'gameStats.txt')
            with open(path_file, 'a') as the_file:
                the_file.write(str(match_results))

    def save_match_report_to_file(
            self,
            match_report: MatchReport
    ) -> None:
        if self.output_folder_path is not None:
            path_file: path = os.path.join(self.output_folder_path, 'match_report.obj')
            with open(path_file, 'wb') as the_file:
                pickle.dump(match_report, the_file)

    def subscribe(self, subscriber):
        self.game_manager_factory.subscribe(subscriber)
        self.match_results_factory.subscribe(subscriber)
