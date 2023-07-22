import random
from chipiron.players.factory import create_player
from scripts.script import Script
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
import yaml
from chipiron.environments.chess.board import create_board
import cProfile
import time
import io
from pstats import SortKey
import pstats
import os


class BaseTreeExplorationScript(Script):

    def __init__(self):
        super().__init__()

    def run(self):
        syzygy = SyzygyTable('')

        profile = cProfile.Profile()
        profile.enable()

        file_name_player_one = 'RecurZipfBase3.yaml'
        file_name_player_one = 'Uniform.yaml'

        path_player_one = os.path.join('data/players/player_config', file_name_player_one)

        with open(path_player_one, 'r') as filePlayerOne:
            args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player_one)

        random_generator = random.Random()
        player = create_player(args=args_player_one, syzygy=syzygy, random_generator=random_generator)

        board = create_board()
        player.select_move(board=board)

        print(f'--- {time.time() - self.start_time} seconds ---')
        profile.disable()
        string_io = io.StringIO()
        sort_by = SortKey.CUMULATIVE
        stats = pstats.Stats(profile, stream=string_io).sort_stats(sort_by)
        stats.print_stats()
        print(string_io.getvalue())
