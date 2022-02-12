import chess
from src.players.game_player import GamePlayer
from src.players.boardevaluators.stockfish_evaluation import Stockfish
from src.games.game_manager import GameManager
from src.games.match_manager import MatchManager, MatchResults, ObservableMatchResults
from src.players.factory import create_player, launch_player_process
import yaml
from src.players.boardevaluators.table_base.syzygy import SyzygyTable
from src.chessenvironment.boards.factory import create_board
import random
from src.extra_tools.small_tools import unique_int_from_list


class MatchManagerFactory:
    def __init__(self, args_match, args_player_one, args_player_two, syzygy_table, output_folder_path,
                 seed, main_thread_mailbox):
        self.output_folder_path = output_folder_path
        self.player_one_name = args_player_one['name']
        self.player_two_name = args_player_two['name']

        game_manager_board_evaluator_factory = BoardEvaluator2Factory()

        self.game_manager_factory = GameManagerFactory(syzygy_table, game_manager_board_evaluator_factory,
                                                       output_folder_path, main_thread_mailbox
                                                       )

        self.match_results_factory = MatchResultsFactory(self.player_one_name, self.player_two_name)
        self.game_args_factory = GameArgsFactory(args_match, args_player_one, args_player_two, seed)

    def create(self):
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


# TODO THERE IS ALREADY A class BoardEvaluator that was done before should we merge or name differently? atm I will note the one for gui display with a 2
class BoardEvaluator2Factory:
    def __init__(self):
        self.subscribers = []

    def create(self):
        board_evaluator = BoardEvaluator2()
        if self.subscribers:

            board_evaluator = ObservableBoardEvaluator2(board_evaluator)
            for subscriber in self.subscribers:
                board_evaluator.subscribe(subscriber)
        return board_evaluator

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)


class ObservableBoardEvaluator2:
    # TODO see if it is possible and desirable to  make a general Observable wrapper that goes all that automatically
    # as i do the same for board and game info
    def __init__(self, board_evaluator):
        self.board_evaluator = board_evaluator
        self.mailboxes = []

    def subscribe(self, mailbox):
        self.mailboxes.append(mailbox)

    # wrapped function
    def get_evaluation(self, board):
        evaluation = self.board_evaluator.get_evaluation(board)
        self.notify_new_results(evaluation)
        return evaluation

    def notify_new_results(self, evaluation):
        for mailbox in self.mailboxes:
            mailbox.put({'type': 'evaluation', 'evaluation': evaluation})

    # forwarding


class BoardEvaluator2:
    def __init__(self):
        self.evaluation = Stockfish()

    def get_evaluation(self, board):
        evaluation = self.evaluation.score(board)
        return evaluation


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
        with open('chipiron/data/settings/GameSettings/' + args_match['game_setting_file'], 'r') as file_game:
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


class GameManagerFactory:
    """
    The GameManagerFactory creates GameManager once the players and rules have been decided.
    Calling create ask for the creation of a GameManager depending on args and players.
    This class is supposed to be independent of Match-related classes (contrarily to the GameArgsFactory)
    """

    def __init__(self, syzygy_table, game_manager_board_evaluator_factory, output_folder_path, main_thread_mailbox):
        self.syzygy_table = syzygy_table
        self.output_folder_path = output_folder_path
        self.game_manager_board_evaluator_factory = game_manager_board_evaluator_factory
        self.main_thread_mailbox = main_thread_mailbox
        self.subscribers = []

    def create(self, args_game_manager, player_color_to_player):
        # maybe this factory is overkill at the moment but might be
        # useful if the logic of game generation gets more complex

        board = create_board(self.subscribers)
        if self.subscribers:
            player_color_to_id = {color: player.id for color, player in player_color_to_player.items()}
            for subscriber in self.subscribers:
                subscriber.put({'type': 'players_color_to_id', 'players_color_to_id': player_color_to_id})
        board_evaluator = self.game_manager_board_evaluator_factory.create()

        while not self.main_thread_mailbox.empty():
            self.main_thread_mailbox.get()

        player_processes = []
        # Creating and launching the player threads
        for player_color in chess.COLORS:
            player = player_color_to_player[player_color]
            game_player = GamePlayer(player, player_color)
            if player.id != 'Human':  # TODO COULD WE DO BETTER ? maybe with the null object
                player_process = launch_player_process(game_player, board, self.main_thread_mailbox)
                player_processes.append(player_process)

        game_manager = GameManager(board,
                                   self.syzygy_table,
                                   board_evaluator,
                                   self.output_folder_path,
                                   args_game_manager,
                                   player_color_to_id,
                                   self.main_thread_mailbox,
                                   player_processes)

        return game_manager

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
        self.game_manager_board_evaluator_factory.subscribers.append(subscriber)
