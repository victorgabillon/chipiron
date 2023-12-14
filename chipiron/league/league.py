import shutil
import datetime
import yaml
import os
from chipiron.utils.small_tools import yaml_fetch_args_in_file
from sortedcollections import ValueSortedDict
from collections import deque
import matplotlib.pyplot as plt
import chipiron as ch
from chipiron.games.match_factories import create_match_manager
from utils import path


def new_player_joins(player):
    league_folder = 'chipiron/runs/league/league_10000/new_players/'
    player_filename = league_folder + '/player' + str(datetime.datetime.now()) + '.yaml'
    with open(player_filename, 'w') as out_file:
        out_file.write(yaml.dump(player.arg))


class League:
    K = 10
    ELO_HISTORY_LENGTH = 500

    def __init__(self, foldername, random_generator):
        print('init league from folder: ', foldername)
        self.folder_league = foldername
        self.players_elo = ValueSortedDict()
        self.players_args = {}
        self.players_number_of_games_played = {}

        self.id_for_next_player = 0
        self.check_for_players()
        self.random_generator = random_generator

    def check_for_players(self):
        path: str = os.path.join(self.folder_league, 'new_players/')
        if os.path.exists(path):
            files = os.listdir(path)
            if len(files) > 0:
                for file in files:
                    path_file = os.path.join(self.folder_league, 'new_players/', file)
                    self.new_player_joins(path_file)

    def new_player_joins(self, file_player):
        print('adding player:', file_player)
        with open(file_player, 'r') as filePlayerOne:
            args_player = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player)

        args_player['name'] = args_player['name'] + '_' + str(self.id_for_next_player)
        self.id_for_next_player += 1

        self.players_elo[args_player['name']] = deque([1200], maxlen=self.ELO_HISTORY_LENGTH)
        self.players_args[args_player['name']] = args_player
        self.players_number_of_games_played[args_player['name']] = 0

        shutil.move(file_player, self.folder_league + '/current_players')

        print('elo', self.players_elo)
        print('args', self.players_args)

    def run(self):
        self.print_info()
        self.update_elo_graph()

        args_player_one, args_player_two = self.select_two_players()

        #  play
        path_match_setting: str = 'data/settings/OneMatch/setting_jime.yaml'
        args_match: dict = yaml_fetch_args_in_file(path_file=path_match_setting)

        file_path: path = 'data/settings/GameSettings/setting_navo.yaml'
        with open(file_path, 'r', encoding="utf-8") as file_game:
            args_game: dict = yaml.load(file_game, Loader=yaml.FullLoader)

        match_manager: ch.game.MatchManager = create_match_manager(
            args_match=args_match,
            args_player_one=args_player_one,
            args_player_two=args_player_two,
            output_folder_path=None,
            seed=None,
            args_game=args_game,
            gui=False
        )

        match_results = match_manager.play_one_match()
        with open(self.folder_league + '/log_results.txt', 'a') as log_file:
            log_file.write(
                f'{args_player_one["name"]} vs {args_player_two["name"]}: {match_results.get_player_one_wins()}-{match_results.get_player_two_wins()}-{match_results.get_draws()}\n')
        self.players_number_of_games_played[args_player_one['name']] += 1
        self.players_number_of_games_played[args_player_two['name']] += 1

        # tobecodedupdate
        self.update_elo(match_results)

    def update_elo(self, match_results):
        # coded for one single game!!
        player_one_name_id = match_results.player_one_name_id
        elo_player_one = self.players_elo[player_one_name_id][0]
        power_player_one = 10 ** (elo_player_one / 400)
        player_two_name_id = match_results.player_two_name_id
        elo_player_two = self.players_elo[player_two_name_id][0]
        power_player_two = 10 ** (elo_player_two / 400)
        Eone = power_player_one / (power_player_one + power_player_two)
        Etwo = power_player_two / (power_player_one + power_player_two)

        Perf_one = match_results.get_player_one_wins() + match_results.get_draws() / 2.
        Perf_two = match_results.get_player_two_wins() + match_results.get_draws() / 2.

        print(elo_player_one, self.players_elo[player_one_name_id])
        old_elo_player_one = elo_player_one
        old_elo_player_two = elo_player_two
        increment_one = self.K * (Perf_one - Eone)
        increment_two = self.K * (Perf_two - Etwo)
        new_elo_one = old_elo_player_one + increment_one
        new_elo_two = old_elo_player_two + increment_two
        self.players_elo[player_one_name_id].appendleft(new_elo_one)
        self.players_elo[player_two_name_id].appendleft(new_elo_two)

        for player in self.players_elo:
            if player != player_one_name_id and player != player_two_name_id:
                self.players_elo[player].appendleft(self.players_elo[player][0])

        with open(self.folder_league + '/log_results.txt', 'a') as log_file:
            log_file.write(
                f'{player_one_name_id} increments its elo by {increment_one}: {old_elo_player_one} -> {new_elo_one}\n')
            log_file.write(
                f'{player_two_name_id} increments its elo by {increment_two}: {old_elo_player_two} -> {new_elo_two}\n')

        self.update_elo_graph()

    def update_elo_graph(self):
        plt.clf()
        for player_name, elo in self.players_elo.items():
            elo.reverse()
            plt.plot(elo, label=player_name)
            elo.reverse()
        # plt.axis([0, 6, 0, 20])
        plt.legend()
        plt.savefig(self.folder_league + '/elo.plot.svg', format='svg')

    def select_two_players(self):
        if len(self.players_args) < 2:
            raise ValueError(
                'Not enough players in the league. To add players put the yaml files in the folder "new players"')
        picked = self.random_generator.sample(list(self.players_args.values()), k=2)
        print('picked', picked)
        return picked[0], picked[1]

    def save(self):
        pass

    def check_to_discard_bad_player(self, player):
        pass

    def print_info(self):
        print('print info league')
        print('players', self.players_elo)
