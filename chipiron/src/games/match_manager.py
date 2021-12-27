import yaml
from src.games.game_manager import GameManager


# todo a wrapper for chess.white chess.black


class MatchManager:
    """
    Objet in charge of playing one game
    """


    def __init__(self, args_match, player_one, player_two, game_manager, output_folder_path=None):

        assert(player_one.player_name_id != player_two.player_name_id)
        self.args_match = args_match
        self.player_one = player_one
        self.player_two = player_two
        self.game_manager = game_manager
        self.output_folder_path = output_folder_path

        with open('chipiron/data/settings/GameSettings/' + self.args_match['game_setting_file'], 'r') as file_game:
            self.args_game = yaml.load(file_game, Loader=yaml.FullLoader)
            print(self.args_game)

        self.match_results = MatchResults(self.player_one, self.player_two)

        self.print_info()

    def print_info(self):
        print('player one is ', self.player_one.player_name_id)
        print('player two is ', self.player_two.player_name_id)

    def play_one_match(self):
        print('Playing the Match')
        self.game_manager.set(self.args_game, self.player_one, self.player_two)
        for game_number in range(self.args_match['number_of_games_player_one_white']):
            game_result_p1w = self.game_manager.play_one_game()
            self.match_results.add_result_one_game(white_player_name_id=self.player_one.player_name_id,
                                                   game_result=game_result_p1w)
            self.game_manager.print_to_file(idx=game_number)

        self.game_manager.swap_players()
        for game_number in range(self.args_match['number_of_games_player_one_black']):
            game_result_p2w = self.game_manager.play_one_game()
            self.match_results.add_result_one_game(white_player_name_id=self.player_two.player_name_id,
                                                   game_result=game_result_p2w)
            self.game_manager.print_to_file(idx=game_number)

        self.print_stats_to_screen()
        self.print_stats_to_file()
        return self.match_results

    def print_stats_to_screen(self):
        print(self.match_results)

    def print_stats_to_file(self):
        if self.output_folder_path is not None:
            path_file = self.output_folder_path + '/gameStats.txt'
            with open(path_file, 'a') as the_file:
                the_file.write(str(self.match_results))


class MatchResults:

    def __init__(self,  player_one, player_two):
        self.player_one_name_id = player_one.player_name_id
        self.player_two_name_id = player_two.player_name_id
        self.number_of_games = 0
        self.player_one_is_white_white_wins = 0
        self.player_one_is_white_black_wins = 0
        self.player_one_is_white_draws = 0
        self.player_two_is_white_white_wins = 0
        self.player_two_is_white_black_wins = 0
        self.player_two_is_white_draws = 0

    def get_player_one_wins(self):
        return self.player_one_is_white_white_wins + self.player_two_is_white_black_wins

    def get_player_two_wins(self):
        return self.player_one_is_white_black_wins + self.player_two_is_white_white_wins

    def get_draws(self):
        return self.player_one_is_white_draws + self.player_two_is_white_draws

    def get_simple_result(self):
        return self.get_player_one_wins(), self.get_player_two_wins(), self.get_draws()

    def add_result_one_game(self, white_player_name_id, game_result):
        self.number_of_games += 1
        if white_player_name_id == self.player_one_name_id:
            if game_result == GameManager.WIN_FOR_WHITE:
                self.player_one_is_white_white_wins += 1
            elif game_result == GameManager.WIN_FOR_BLACK:
                self.player_one_is_white_black_wins += 1
            elif game_result == GameManager.DRAW:
                self.player_one_is_white_draws += 1
            else:
                pass
                # raise Exception('#')
        elif white_player_name_id == self.player_two_name_id:
            if game_result == GameManager.WIN_FOR_WHITE:
                self.player_two_is_white_white_wins += 1
            elif game_result == GameManager.WIN_FOR_BLACK:
                self.player_two_is_white_black_wins += 1
            elif game_result == GameManager.DRAW:
                self.player_two_is_white_draws += 1
            else:
                raise Exception('!')
        else:
            raise Exception('?')

    def __str__(self):
        str_ = 'Main result: ' + self.player_one_name_id + ' wins ' + str(self.get_player_one_wins()) + ' '
        str_ += self.player_two_name_id + ' wins ' + str(self.get_player_two_wins())
        str_ += ' draws ' + str(self.get_draws()) + '\n'

        str_ += self.player_one_name_id + ' with white: '
        str_ += 'Wins ' + str(self.player_one_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_two_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws) + '\n'

        str_ += self.player_two_name_id + ' with white: '
        str_ += 'Wins ' + str(self.player_two_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_one_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        return str_
