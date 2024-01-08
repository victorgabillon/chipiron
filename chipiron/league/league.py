import shutil
import datetime
import yaml
import os
from sortedcollections import ValueSortedDict
from collections import deque
import matplotlib.pyplot as plt
import chipiron as ch
from chipiron.games.match.match_factories import create_match_manager
import chipiron.games.game as game
import chipiron.games.match as match
from chipiron.games.match.utils import fetch_match_games_args_convert_and_save
from chipiron.players.utils import fetch_player_args_convert_and_save
from dataclasses import dataclass, field
import chipiron.players as players
import random
from memory_profiler import profile


def new_player_joins(player):
    league_folder = 'chipiron/runs/league/league_10000/new_players/'
    player_filename = league_folder + '/player' + str(datetime.datetime.now()) + '.yaml'
    with open(player_filename, 'w') as out_file:
        out_file.write(yaml.dump(player.arg))


@dataclass(slots=True)
class League:
    folder_league: str
    players_elo: ValueSortedDict = field(default_factory=ValueSortedDict)
    players_args: dict[str, players.PlayerArgs] = field(default_factory=dict)
    players_number_of_games_played: dict[str, int] = field(default_factory=dict)
    id_for_next_player: int = 0
    K: int = 10
    ELO_HISTORY_LENGTH: int = 500

    def __post_init__(self):
        print(f'init league from folder: {self.folder_league}')
        self.check_for_players()

    def check_for_players(self):
        path: str = os.path.join(self.folder_league, 'new_players/')
        if os.path.exists(path):
            files = os.listdir(path)
            if len(files) > 0:
                for file in files:
                    path_file = os.path.join(self.folder_league, 'new_players/', file)
                    self.new_player_joins(path_file)

    def new_player_joins(
            self,
            file_player: str | bytes | os.PathLike
    ) -> None:

        print('adding player:', file_player)
        args_player: players.PlayerArgs = fetch_player_args_convert_and_save(
            file_name_player=file_player,
            from_data_folder=False)
        print(args_player)

        args_player.name = f'{args_player.name}_{self.id_for_next_player}'
        self.id_for_next_player += 1

        self.players_elo[args_player.name] = deque([1200], maxlen=self.ELO_HISTORY_LENGTH)
        self.players_args[args_player.name] = args_player
        self.players_number_of_games_played[args_player.name] = 0

        shutil.move(file_player, os.path.join(self.folder_league, 'current_players'))

        print('elo', self.players_elo)
        print('args', self.players_args)

    @profile
    def run(self) -> None:
        self.print_info()
        self.update_elo_graph()

        args_player_one, args_player_two = self.select_two_players()

        #  play
        file_match_setting: str = 'setting_jime.yaml'

        # Recovering args from yaml file for match and game and merging with extra args and converting
        # to standardized dataclass
        match_args: match.MatchArgs
        game_args: game.GameArgs
        match_args, game_args = fetch_match_games_args_convert_and_save(
            file_name_match_setting=file_match_setting,
        )

        match_manager: ch.game.MatchManager = create_match_manager(
            args_match=match_args,
            args_player_one=args_player_one,
            args_player_two=args_player_two,
            args_game=game_args
        )

        match_results = match_manager.play_one_match()
        with open(self.folder_league + '/log_results.txt', 'a') as log_file:
            log_file.write(
                f'{args_player_one.name} vs {args_player_two.name}: {match_results.get_player_one_wins()}-{match_results.get_player_two_wins()}-{match_results.get_draws()}\n')
        self.players_number_of_games_played[args_player_one.name] += 1
        self.players_number_of_games_played[args_player_two.name] += 1

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
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        for player_name, elo in self.players_elo.items():
            elo.reverse()
            ax.plot(elo, label=player_name)
            elo.reverse()
        # plt.axis([0, 6, 0, 20])
        plt.legend()
        plt.savefig(self.folder_league + '/elo.plot.svg', format='svg')

    def select_two_players(self):
        if len(self.players_args) < 2:
            raise ValueError(
                'Not enough players in the league. To add players put the yaml files in the folder "new players"')
        picked = random.sample(list(self.players_args.values()), k=2)
        print('picked', picked)
        return picked[0], picked[1]

    def save(self):
        pass

    def check_to_discard_bad_player(self, player):
        pass

    def print_info(self):
        print('print info league')
        print('players', self.players_elo)
