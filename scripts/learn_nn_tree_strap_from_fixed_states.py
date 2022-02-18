from scripts.script import Script
from src.players.factory import create_player
import yaml
from src.players.boardevaluators.table_base.syzygy import SyzygyTable

class LearnNNTreeStrap(Script):

    def __init__(self):
        super().__init__()
        file_name_player_one = 'RecurZipfBase.yaml'
        path_player_one = 'chipiron/runs/players/' + file_name_player_one

        with open(path_player_one, 'r') as filePlayerOne:
            args_player_one = yaml.load(filePlayerOne, Loader=yaml.FullLoader)
            print(args_player_one)

        syzygy = SyzygyTable()
        self.player_one = create_player(args_player_one, syzygy)

    def run(self):
        print('run')

