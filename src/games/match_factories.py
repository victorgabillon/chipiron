import chess
from src.games.match_manager import MatchManager
from src.games.game_manager_factory import GameManagerFactory
from src.games.math_results import MatchResults, ObservableMatchResults
from src.players.factory import create_player
import yaml
from src.players.boardevaluators.table_base.syzygy import SyzygyTable
from src.players.boardevaluators.factory import ObservableBoardEvaluatorFactory
import random
from src.extra_tools.small_tools import unique_int_from_list


class MatchManagerFactory:
    def __init__(self,
                 args_match: dict,
                 args_player_one: dict,
                 args_player_two: dict,
                 syzygy_table: object,
                 output_folder_path: object,
                 seed: int,
                 main_thread_mailbox: object) -> None:
        self.output_folder_path = output_folder_path
        self.player_one_name = args_player_one['name']
        self.player_two_name = args_player_two['name']

        game_manager_board_evaluator_factory = ObservableBoardEvaluatorFactory()

        self.game_manager_factory = GameManagerFactory(syzygy_table, game_manager_board_evaluator_factory,
                                                             output_folder_path, main_thread_mailbox)

        self.match_results_factory = MatchResultsFactory(self.player_one_name, self.player_two_name)
        self.game_args_factory = GameArgsFactory(args_match, args_player_one, args_player_two, seed)

    def create(self) -> MatchManager:
        match_manager: MatchManager
        match_manager = MatchManager(self.player_one_name,
                                     self.player_two_name,
                                     self.game_manager_factory,
                                     self.game_args_factory,
                                     self.match_results_factory,
                                     self.output_folder_path)

        return match_manager

    def subscribe(self, subscriber):
        self.game_manager_factory.subscribe(subscriber)
        self.match_results_factory.subscribe(subscriber)


class MatchResultsFactory:
    def __init__(self, player_one_name, player_two_name):
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscribers = []

    def create(self):
        match_result = MatchResults(self.player_one_name, self.player_two_name)
        if self.subscribers:
            match_result = ObservableMatchResults(match_result)
            for subscriber in self.subscribers:
                match_result.subscribe(subscriber)
        return match_result

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)


class GameArgsFactory:
    # TODO MAYBE CHANGE THE NAME, ALSO MIGHT BE SPLIT IN TWO (players and rules)?
    """
    The GameArgsFactory creates the players and decides the rules.
    So far quite simple
    This class is supposed to be dependent of Match-related classes (contrarily to the GameArgsFactory)

    """

    def __init__(self, args_match, args_player_one, args_player_two, seed):
        self.args_match = args_match
        self.seed = seed
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        with open('data/settings/GameSettings/' + args_match['game_setting_file'], 'r') as file_game:
            self.args_game = yaml.load(file_game, Loader=yaml.FullLoader)
        self.game_number = 0

    def generate_game_args(self, game_number):
        print('args_game', self.args_game)

        # Creating the players
        syzygy_table = SyzygyTable('')
        random_generator = random.Random(unique_int_from_list([self.seed, game_number]))
        player_one = create_player(self.args_player_one, syzygy_table, random_generator)
        player_two = create_player(self.args_player_two, syzygy_table, random_generator)

        if game_number < self.args_match['number_of_games_player_one_white']:
            player_color_to_player = {chess.WHITE: player_one, chess.BLACK: player_two}
        else:
            player_color_to_player = {chess.WHITE: player_two, chess.BLACK: player_one}
        self.game_number += 1

        return player_color_to_player, self.args_game

    def is_match_finished(self):
        return self.game_number >= self.args_match['number_of_games_player_one_white'] + self.args_match[
            'number_of_games_player_one_black']
