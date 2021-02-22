from games.play_one_game import PlayOneGame
import yaml
from games.game import Game


# todo a wrapper for chess.white chess.black


class PlayOneMatch:
    """
    Objet in charge of playing one game
    """

    PLAYERS_ID = [PLAYER_ONE_ID, PLAYER_TWO_ID] = range(2)

    def __init__(self, args_math, player_one, player_two, chess_simulator, syzygy, path_directory=None):

        self.chess_simulator = chess_simulator
        self.syzygy = syzygy
        self.argsMath = args_math
        self.number_of_games_player_one_white = args_math['number_of_games_player_one_white']
        self.number_of_games_player_one_black = args_math['number_of_games_player_one_black']
        self.path_directory = path_directory
        self.game_setting_file = args_math['game_setting_file']

        with open('runs/GameSettings/' + self.game_setting_file, 'r') as fileGame:
            self.argsGame = yaml.load(fileGame, Loader=yaml.FullLoader)
            print(self.argsGame)

        self.player_one = player_one
        self.player_two = player_two

        self.match_results = MatchResults(self.PLAYER_ONE_ID, self.PLAYER_TWO_ID, self.player_one, self.player_two)

        self.print_info()

    def print_info(self):
        print('player one is ', self.player_one.player_name)
        print('player two is ', self.player_two.player_name)

    def play_the_match(self):

        for game_number in range(self.number_of_games_player_one_white):
            print('game number', game_number, 'white:', self.player_one.player_name, 'black:',
                  self.player_two.player_name )
            if self.path_directory is not None:
                path_to_store_result = self.path_directory + '/games/Game' + str(game_number)
            else:
                path_to_store_result = None
            self.play_one_game = PlayOneGame(self.argsGame, self.player_one, self.player_two, self.chess_simulator,
                                             self.syzygy, path_to_store_result)

            game_result_p1w = self.play_one_game.play_the_game()
            self.match_results.add_result_one_game(who_is_white=self.PLAYER_ONE_ID,
                                                   game_result=game_result_p1w)

        for game_number in range(self.number_of_games_player_one_black):
            print('game P2 in White', game_number)
            if self.path_directory is not None:
                path_to_store_result = self.path_directory + '/games/Game' + str(game_number)
            else:
                path_to_store_result = None
            self.play_one_game = PlayOneGame(self.argsGame, self.player_two, self.player_one, self.chess_simulator,
                                             self.syzygy, path_to_store_result)

            game_result_p2w = self.play_one_game.play_the_game()
            self.match_results.add_result_one_game(who_is_white=self.PLAYER_TWO_ID,
                                                   game_result=game_result_p2w)

        self.print_stats_to_screen()
        self.print_stats_to_file()
        return self.match_results.get_simple_result()

    def print_stats_to_screen(self):
        print(self.match_results)

    def print_stats_to_file(self):
        if self.path_directory is not None:
            path_file = self.path_directory + '/gameStats.txt'
            with open(path_file, 'a') as the_file:
                the_file.write(str(self.match_results))


class MatchResults:

    def __init__(self, player_one_id, player_two_id, player_one, player_two):
        self.player_one = player_one
        self.player_two = player_two
        self.number_of_games = 0
        self.player_one_id = player_one_id
        self.player_two_id = player_two_id
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

    def add_result_one_game(self, who_is_white, game_result):
        self.number_of_games += 1
        if who_is_white == self.player_one_id:
            if game_result == Game.WIN_FOR_WHITE:
                self.player_one_is_white_white_wins += 1
            elif game_result == Game.WIN_FOR_BLACK:
                self.player_one_is_white_black_wins += 1
            elif game_result == Game.DRAW:
                self.player_one_is_white_draws += 1
            else:
                pass
                # raise Exception('#')
        elif who_is_white == self.player_two_id:
            if game_result == Game.WIN_FOR_WHITE:
                self.player_two_is_white_white_wins += 1
            elif game_result == Game.WIN_FOR_BLACK:
                self.player_two_is_white_black_wins += 1
            elif game_result == Game.DRAW:
                self.player_two_is_white_draws += 1
            else:
                raise Exception('!')
        else:
            raise Exception('?')

    def __str__(self):
        str_ = 'Main result: ' + self.player_one.player_name + ' wins ' + str(self.get_player_one_wins()) + ' '
        str_ += self.player_two.player_name + ' wins ' + str(self.get_player_two_wins())
        str_ += ' draws ' + str(self.get_draws()) + '\n'

        str_ += self.player_one.player_name + ' with white: '
        str_ += 'Wins ' + str(self.player_one_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_two_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws) + '\n'

        str_ += self.player_two.player_name + ' with white: '
        str_ += 'Wins ' + str(self.player_two_is_white_white_wins)
        str_ += ', Losses ' + str(self.player_two_is_white_black_wins)
        str_ += ', Draws ' + str(self.player_two_is_white_draws)
        str_ += '\n           with black: '
        str_ += 'Wins ' + str(self.player_one_is_white_black_wins)
        str_ += ', Losses ' + str(self.player_one_is_white_white_wins)
        str_ += ', Draws ' + str(self.player_one_is_white_draws)
        return str_
