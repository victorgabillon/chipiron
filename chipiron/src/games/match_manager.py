import chess
from src.games.game_manager import GameManager
import copy


class MatchManager:
    """
    Objet in charge of playing one game
    """

    def __init__(self,
                 player_one_id,
                 player_two_id,
                 game_manager_factory,
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

    def play_one_match(self):
        print('Playing the match')
        match_results = self.match_results_factory.create()

        game_number = 0
        while not self.game_args_factory.is_match_finished():
            player_color_to_id, args_game = self.game_args_factory.generate_game_args(game_number)
            game_results = self.launch_game(player_color_to_id, args_game, game_number)
            match_results.add_result_one_game(white_player_name_id=player_color_to_id[chess.WHITE],
                                              game_result=game_results)
            game_number += 1

        print(match_results)
        self.print_stats_to_file(match_results)
        return match_results

    def launch_game(self, player_color_to_id, args_game, game_number):
        game_manager = self.game_manager_factory.create(args_game, player_color_to_id)
        game_results = game_manager.play_one_game()
        game_manager.print_to_file(idx=game_number)
        game_manager.terminate_threads()  # TODO should this line be inside the play one game ?
        return game_results

    def print_stats_to_file(self, match_results):
        if self.output_folder_path is not None:
            path_file = self.output_folder_path + '/gameStats.txt'
            with open(path_file, 'a') as the_file:
                the_file.write(str(match_results))


class MatchResults:

    def __init__(self, player_one_id, player_two_id):
        self.player_one_name_id = player_one_id
        self.player_two_name_id = player_two_id
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


class ObservableMatchResults:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info
    def __init__(self, match_result):
        self.match_result = match_result
        self.mailboxes = []

    def subscribe(self, mailbox):
        self.mailboxes.append(mailbox)

    def copy_match_result(self):
        match_result_copy = copy.deepcopy(self.match_result)
        return match_result_copy

    # wrapped function
    def add_result_one_game(self, white_player_name_id, game_result):
        self.match_result.add_result_one_game(white_player_name_id, game_result)
        self.notify_new_results()

    def notify_new_results(self):
        for mailbox in self.mailboxes:
            match_result_copy = self.copy_match_result()
            mailbox.put({'type': 'match_results', 'match_results': match_result_copy})

    # forwarding
    def get_simple_result(self):
        return self.match_result.get_player_one_wins(), \
               self.match_result.get_player_two_wins(), \
               self.match_result.get_draws()
