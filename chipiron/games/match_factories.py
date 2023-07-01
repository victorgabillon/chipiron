import chess
from chipiron.games.match_manager import MatchManager
from chipiron.games.game_manager_factory import GameManagerFactory
from chipiron.games.match_results_factory import MatchResultsFactory
from chipiron.players.factory import create_player
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.boardevaluators.factory import ObservableBoardEvaluatorFactory
import random
from chipiron.extra_tools.small_tools import unique_int_from_list
import queue


class MatchManagerFactory:

    def __init__(self,
                 args_match: dict,
                 args_player_one: dict,
                 args_player_two: dict,
                 syzygy_table: SyzygyTable,
                 output_folder_path: str,
                 seed: int,
                 main_thread_mailbox: queue.Queue,
                 args_game: dict) -> None:
        self.output_folder_path = output_folder_path
        self.player_one_name: str = args_player_one['name']
        self.player_two_name: str = args_player_two['name']

        game_manager_board_evaluator_factory: ObservableBoardEvaluatorFactory = ObservableBoardEvaluatorFactory()

        self.game_manager_factory: GameManagerFactory = GameManagerFactory(
            syzygy_table=syzygy_table,
            game_manager_board_evaluator_factory=game_manager_board_evaluator_factory,
            output_folder_path=output_folder_path,
            main_thread_mailbox=main_thread_mailbox
        )

        self.match_results_factory: MatchResultsFactory = MatchResultsFactory(
            player_one_name=self.player_one_name,
            player_two_name=self.player_two_name
        )

        self.game_args_factory: GameArgsFactory = GameArgsFactory(args_match,
                                                                  args_player_one,
                                                                  args_player_two,
                                                                  seed,
                                                                  args_game)

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


class GameArgsFactory:
    # TODO MAYBE CHANGE THE NAME, ALSO MIGHT BE SPLIT IN TWO (players and rules)?
    """
    The GameArgsFactory creates the players and decides the rules.
    So far quite simple
    This class is supposed to be dependent of Match-related classes (contrarily to the GameArgsFactory)

    """

    def __init__(self,
                 args_match,
                 args_player_one,
                 args_player_two,
                 seed: int,
                 args_game):
        self.args_match = args_match
        self.seed = seed
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.game_number = 0

    def generate_game_args(self, game_number):
        print('args_game', self.args_game)

        # Creating the players
        syzygy_table = SyzygyTable('')
        merged_seed = unique_int_from_list([self.seed, game_number])
        random_generator: random.Random = random.Random(merged_seed)
        player_one = create_player(args=self.args_player_one,
                                   syzygy=syzygy_table,
                                   random_generator=random_generator)
        player_two = create_player(args=self.args_player_two,
                                   syzygy=syzygy_table,
                                   random_generator=random_generator)

        if game_number < self.args_match['number_of_games_player_one_white']:
            player_color_to_player = {chess.WHITE: player_one, chess.BLACK: player_two}
        else:
            player_color_to_player = {chess.WHITE: player_two, chess.BLACK: player_one}
        self.game_number += 1

        return player_color_to_player, self.args_game

    def is_match_finished(self):
        return self.game_number >= self.args_match['number_of_games_player_one_white'] + self.args_match[
            'number_of_games_player_one_black']
