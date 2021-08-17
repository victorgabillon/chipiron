import pickle
import datetime
import yaml
import os
from src.games.match_manager import MatchManager
import random


def new_player_joins(player):
    league_folder = 'chipiron/data/league/league_10000/new_players/'
    player_filename = league_folder + '/player' + str(datetime.datetime.now()) + '.yaml'
    with open(player_filename, 'w') as out_file:
        out_file.write(yaml.dump(player.arg))


class League:

    def __init__(self, foldername):
        print('init league from folder: ', foldername)
        self.folder_league = foldername
        try:
            with (open(foldername + '/players.pickle', "rb")) as openfile:
                self.players_elo = pickle.load(openfile)
        except:
            print('45')
            self.players_elo = []

        print('players', self.players_elo)
        self.check_for_players()

    def check_for_players(self):
        files = os.listdir(self.folder_league + '/new_players/')
        if len(files) > 0:
            for file in files:
                self.new_player_joins(self.folder_league + '/new_players/' + file)

    def new_player_joins(self, file_player):
        print('adding player:', file_player)
        with open(file_player, 'r') as filePlayerOne:
            args_player = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player)

        self.players_elo.append([1200 + 100 * random.random(), args_player])
        self.players_elo.sort()
        os.remove(file_player)
        print(self.players_elo)


    def run(self):
        player_one, player_two = self.select_two_players()

        # tobecoded play
        path_match_setting = 'runs/OneMatch/setting_duda.yaml'
        with open(path_match_setting, 'r') as fileMatch:
            args_match = yaml.load(fileMatch, Loader=yaml.FullLoader)
        play = MatchManager(args_match, player_one, player_two, chess_simulator, syzygy, pathDirectory)
        play.play_the_match()

        # tobecodedupdate

        # save
        pickle.dump(self.players_elo, open(self.folder_league + '/players.pickle', "wb"))

    def select_two_players(self):
        return 0, 3

    def save(self):
        pass

    def check_to_discard_bad_player(self, player):
        pass
