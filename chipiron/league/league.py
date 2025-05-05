"""
This module defines the `League` class, which represents a league of players in a game.
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from sortedcollections import ValueSortedDict

import chipiron as ch
import chipiron.games.game as game
import chipiron.games.match as match
import chipiron.players as players
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager
from chipiron.games.match.match_results import MatchReport, MatchResults
from chipiron.games.match.MatchTag import MatchConfigTag
from chipiron.games.match.utils import fetch_match_games_args_convert_and_save
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.utils.small_tools import mkdir_if_not_existing, path


@dataclass(slots=True)
class League:
    """
    Represents a league of players in a game.

    Attributes:
        folder_league (str): The folder path where the league is located.
        seed (int): The seed value for random number generation.
        players_elo (ValueSortedDict): A dictionary that stores the Elo ratings of the players.
        players_args (dict[str, players.PlayerArgs]): A dictionary that stores the arguments of the players.
        players_number_of_games_played (dict[str, int]): A dictionary that stores the number of games played by each player.
        id_for_next_player (int): The ID for the next player to join the league.
        K (int): The K-factor used in the Elo rating system.
        ELO_HISTORY_LENGTH (int): The length of Elo rating history to keep for each player.
        games_already_played (int): The number of games already played in the league.
    """

    folder_league: str
    seed: int
    players_elo: ValueSortedDict = field(default_factory=ValueSortedDict)
    players_args: dict[str, players.PlayerArgs] = field(default_factory=dict)
    players_number_of_games_played: dict[str, int] = field(default_factory=dict)
    id_for_next_player: int = 0
    K: int = 10
    ELO_HISTORY_LENGTH: int = 500
    games_already_played: int = 0

    def __post_init__(self) -> None:
        """
        Initializes the league object.

        This method is called after the object is created and initializes the necessary attributes.
        It also creates the required folders for logging and storing game data.
        """
        print(f"init league from folder: {self.folder_league}")
        self.check_for_players()
        path_logs_folder: path = os.path.join(self.folder_league, "logs")
        mkdir_if_not_existing(path_logs_folder)
        path_logs_games_folder: path = os.path.join(path_logs_folder, "games")
        mkdir_if_not_existing(path_logs_games_folder)

    def check_for_players(self) -> None:
        """
        Checks for new players joining the league.

        This method checks if there are any new player files in the 'new_players' folder and adds them to the league.
        """
        path: str = os.path.join(self.folder_league, "new_players/")
        if os.path.exists(path):
            files = os.listdir(path)
            if len(files) > 0:
                for file in files:
                    path_file = os.path.join(self.folder_league, "new_players/", file)
                    self.new_player_joins(path_file)

    def new_player_joins(self, file_player: str) -> None:
        """
        Adds a new player to the league.

        Args:
            file_player (str | os.PathLike[str]): The path to the player file.

        This method adds a new player to the league by reading the player file, assigning a unique ID to the player,
        and updating the necessary attributes.
        """
        print("adding player:", file_player)

        player_tag = PlayerConfigTag(
            file_player
        )  # probably wrong / to be fixed when reusing the leagu not a big issue
        args_player: players.PlayerArgs = player_tag.get_players_args()

        print(args_player)

        args_player.name = f"{args_player.name}_{self.id_for_next_player}"
        self.id_for_next_player += 1

        self.players_elo[args_player.name] = deque(
            [1200], maxlen=self.ELO_HISTORY_LENGTH
        )
        self.players_args[args_player.name] = args_player
        self.players_number_of_games_played[args_player.name] = 0

        current_player_folder: path = os.path.join(
            self.folder_league, "current_players"
        )
        mkdir_if_not_existing(current_player_folder)
        shutil.move(file_player, current_player_folder)

        print("elo", self.players_elo)
        print("args", self.players_args)

    def run(self) -> None:
        """
        Runs a game in the league.

        This method selects two players from the league, plays a game between them, updates the Elo ratings,
        and saves the results.
        """
        self.print_info()
        self.update_elo_graph()

        args_player_one, args_player_two = self.select_two_players()

        #  play
        file_match_setting: str = "setting_jime.yaml"

        # Recovering args from yaml file for match and game and merging with extra args and converting
        # to standardized dataclass
        match_args: match.MatchSettingsArgs
        game_args: game.GameArgs
        match_args, game_args = fetch_match_games_args_convert_and_save(
            match_args=MatchArgs(
                player_one=PlayerConfigTag.CHIPIRON,
                player_two=PlayerConfigTag.CHIPIRON,
                match_setting=MatchConfigTag(
                    file_match_setting
                ).get_match_settings_args(),  # probibly to fix as weell and this is a ditry fix to create a martch args please imrpove!!
            )
        )

        path_logs_game_folder: path = os.path.join(
            self.folder_league, f"logs/games/game{self.games_already_played}"
        )
        mkdir_if_not_existing(path_logs_game_folder)
        path_logs_game_folder_temp: path = os.path.join(
            self.folder_league, f"logs/games/game{self.games_already_played}/games"
        )
        mkdir_if_not_existing(path_logs_game_folder_temp)

        match_seed = self.seed + self.games_already_played
        match_manager: ch.game.MatchManager = create_match_manager(
            args_match=match_args,
            args_player_one=args_player_one,
            args_player_two=args_player_two,
            args_game=game_args,
            seed=match_seed,
            output_folder_path=path_logs_game_folder,
            implementation_args=ImplementationArgs(),
        )

        # Play the match
        match_report: MatchReport = match_manager.play_one_match()

        # Logs the results
        path_logs_file: path = os.path.join(self.folder_league, "logs/log_results.txt")
        with open(path_logs_file, "a") as log_file:
            log_file.write(
                f"Game #{self.games_already_played} || "
                f"{args_player_one.name} vs {args_player_two.name}: {match_report.match_results.get_player_one_wins()}-"
                f"{match_report.match_results.get_player_two_wins()}-{match_report.match_results.get_draws()}"
                f" with seed {match_seed}\n"
            )
        self.players_number_of_games_played[args_player_one.name] += 1
        self.players_number_of_games_played[args_player_two.name] += 1

        # update the ELO
        self.update_elo(match_report.match_results, path_logs_file)

        self.games_already_played += 1

    def update_elo(self, match_results: MatchResults, path_logs_file: path) -> None:
        """
        Updates the Elo ratings of the players based on the match results.

        Args:
            match_results (MatchResults): The results of the match.
            path_logs_file (path): The path to the log file.

        This method calculates the Elo rating changes for the players based on the match results and updates their ratings.
        It also logs the changes in the log file.
        """
        # coded for one single game!!
        player_one_name_id = match_results.player_one_name_id
        elo_player_one = self.players_elo[player_one_name_id][0]
        power_player_one = 10 ** (elo_player_one / 400)
        player_two_name_id = match_results.player_two_name_id
        elo_player_two = self.players_elo[player_two_name_id][0]
        power_player_two = 10 ** (elo_player_two / 400)
        Eone = power_player_one / (power_player_one + power_player_two)
        Etwo = power_player_two / (power_player_one + power_player_two)

        Perf_one = match_results.get_player_one_wins() + match_results.get_draws() / 2.0
        Perf_two = match_results.get_player_two_wins() + match_results.get_draws() / 2.0

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

        with open(path_logs_file, "a") as log_file:
            log_file.write(
                f"{player_one_name_id} increments its elo by {increment_one}: {old_elo_player_one} -> {new_elo_one}\n"
            )
            log_file.write(
                f"{player_two_name_id} increments its elo by {increment_two}: {old_elo_player_two} -> {new_elo_two}\n"
            )

        self.update_elo_graph()

    def update_elo_graph(self) -> None:
        """
        Updates the Elo rating graph.

        This method updates the Elo rating graph by plotting the Elo ratings of all players in the league.
        """
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        for player_name, elo in self.players_elo.items():
            elo.reverse()
            ax.plot(elo, label=player_name)
            elo.reverse()
        # plt.axis([0, 6, 0, 20])
        plt.legend()
        plt.savefig(self.folder_league + "/elo.plot.svg", format="svg")

    def select_two_players(self) -> tuple[players.PlayerArgs, players.PlayerArgs]:
        """
        Selects two players from the league.

        Returns:
            tuple[players.PlayerArgs, players.PlayerArgs]: A tuple containing the arguments of the two selected players.

        This method randomly selects two players from the league to participate in a game.
        """
        if len(self.players_args) < 2:
            raise ValueError(
                'Not enough players in the league. To add players put the yaml files in the folder "new players"'
            )
        picked = random.sample(list(self.players_args.values()), k=2)
        print("picked", picked)
        return picked[0], picked[1]

    def save(self) -> None:
        """
        Saves the league.

        This method saves the league by serializing its state to a file.
        """
        pass

    def print_info(self) -> None:
        """
        Prints information about the league.

        This method prints information about the league, such as the players and their Elo ratings.
        """
        print("print info league")
        print("players", self.players_elo)
