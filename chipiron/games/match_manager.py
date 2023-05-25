import chess
from chipiron.games.game_manager_factory import GameManagerFactory
from chipiron.games.game_manager import GameManager
from chipiron.games.math_results import MatchResults


class MatchManager:
    """
    Objet in charge of playing one game
    """

    def __init__(self,
                 player_one_id: str,
                 player_two_id: str,
                 game_manager_factory: GameManagerFactory,
                 game_args_factory,
                 match_results_factory,
                 output_folder_path=None):
        assert (player_one_id != player_two_id)
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

    def play_one_match(self) -> MatchResults:
        print('Playing the match')
        match_results = self.match_results_factory.create()

        game_number = 0
        while not self.game_args_factory.is_match_finished():
            player_color_to_player, args_game = self.game_args_factory.generate_game_args(game_number)
            game_results = self.play_one_game(player_color_to_player, args_game, game_number)
            match_results.add_result_one_game(white_player_name_id=player_color_to_player[chess.WHITE].id,
                                              game_result=game_results)
            game_number += 1

        print(match_results)
        self.print_stats_to_file(match_results)
        return match_results

    def play_one_game(self, player_color_to_player, args_game, game_number):
        game_manager: GameManager = self.game_manager_factory.create(args_game, player_color_to_player)
        game_results = game_manager.play_one_game()
        game_manager.print_to_file(idx=game_number)
        return game_results

    def print_stats_to_file(self, match_results):
        if self.output_folder_path is not None:
            path_file = self.output_folder_path + '/gameStats.txt'
            with open(path_file, 'a') as the_file:
                the_file.write(str(match_results))
